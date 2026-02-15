"""
Filter MOFs and BBs by size (code adapted from MOFDiff) and split into train/val/test sets.
"""
import hydra
import pickle
import torch
import rootutils
import numpy as np
from pathlib import Path
from omegaconf import DictConfig

from src.preprocess.bwdb.base import BaseLMDBProcessor
from src.utils.data_utils import (
    lattice_params_to_matrix_torch,
    frac_to_cart_coords,
    frac2cart,
    compute_distance_matrix,
)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class FilterMOF(BaseLMDBProcessor):
    def __init__(self, cfg: DictConfig):
        process_cfg = cfg.preprocess
        filter_cfg = cfg.preprocess.filter_and_split
        # Initialize the base class with the relevant config sections
        super().__init__(process_cfg, filter_cfg)

        # Set specific attributes for this step
        self.split_dir = Path(filter_cfg.split_dir)
        self.max_bbs = filter_cfg.max_bbs
        self.max_atoms = filter_cfg.max_atoms
        self.max_cps = filter_cfg.max_cps
        self.prop_list = filter_cfg.prop_list

    @staticmethod
    def bb_criterion(bb, max_atoms=200, max_cps=20):
        try:
            bb.num_cps = bb.is_anchor.long().sum()
            if (bb.num_atoms > max_atoms) or (bb.num_cps > max_cps):
                return None, False

            cart_coords = frac_to_cart_coords(
                bb.frac_coords, bb.lengths, bb.angles, bb.num_atoms
            )
            pdist = torch.cdist(cart_coords, cart_coords).fill_diagonal_(5.0)

            # Detect BBs with problematic bond info
            edge_index = bb.edge_index
            j, i = edge_index
            bond_dist = (cart_coords[i] - cart_coords[j]).pow(2).sum(dim=-1).sqrt()

            success = (
                pdist.min() > 0.25
                and bond_dist.max() < 5.0
                and (bb.num_atoms <= max_atoms)
                and (bb.num_cps <= max_cps)
            )
            return success
        except Exception:
            return False

    @staticmethod
    def mof_criterion(mof, max_bbs=20):
        try:
            if (
                mof.num_components > max_bbs
                or mof.y.isnan().sum() > 0
                or mof.y.isinf().sum() > 0
            ):
                return False
            cell = lattice_params_to_matrix_torch(mof.lengths, mof.angles).squeeze()
            distances = compute_distance_matrix(
                cell, frac2cart(mof.cg_frac_coords, cell)
            ).fill_diagonal_(5.0)
            return (
                (not (distances < 1.0).any())
                and mof.num_components <= max_bbs
                and mof.y.isnan().sum() == 0
                and mof.y.isinf().sum() == 0
            )
        except Exception:
            return False

    def _get_paths_and_limit(self, split: str):
        """Override base method for custom pathing logic."""
        # Source is a single LMDB, not split-specific
        src_path = self.lmdb_dir / f"{self.input_lmdb_suffix}.lmdb"
        
        # Destination path is nested under the task directory
        output_dir = self.lmdb_dir / self.task
        output_dir.mkdir(parents=True, exist_ok=True)
        dest_path = output_dir / f"{self.output_lmdb_suffix}_{split}.lmdb"
        
        # This step does not use a data limit
        return src_path, dest_path, None
    
    def process_one(self, idx, value):
        try:
            # Load data
            data = pickle.loads(value)

            # Add properties
            data['y'] = torch.tensor([data.prop_dict[prop] for prop in self.prop_list], dtype=torch.float32).view(1, -1)

            # Check criteria
            mof_ok = self.mof_criterion(data, self.max_bbs)
            bbs_ok = all(self.bb_criterion(bb, self.max_atoms, self.max_cps) for bb in data.bbs)

            if mof_ok and bbs_ok:
                return idx, pickle.dumps(data)
            else:
                return idx, None
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            return idx, None


@hydra.main(version_base=None, config_path="../../configs", config_name="preprocess.yaml")
def main(cfg: DictConfig):
    filter = FilterMOF(cfg=cfg)

    # Set splits
    if cfg.preprocess.task == "gen":
        splits = ["train", "val"]
    elif cfg.preprocess.task == "csp":
        splits = ["train", "val", "test"]
    else:
        raise ValueError(f"Unknown task: {cfg.preprocess.task}")
    
    # Process
    for split in splits[::-1]:
        # Load indices for current split
        split_file = filter.split_dir / filter.task / f"{split}_split.txt"
        print(f"INFO:: Loading split indices from {split_file}")
        split_idx = np.loadtxt(split_file, dtype=int)

        filter.process(split=split, keys_to_read=split_idx)

if __name__ == "__main__":
    main()