"""
Extract key features from the raw data.
"""
import hydra
import pickle
import torch
import rootutils
import warnings
import numpy as np

from omegaconf import DictConfig
from rdkit import RDLogger
from openbabel import pybel
from pymatgen.core import Lattice

from src.preprocess.bwdb.base import BaseLMDBProcessor
from src.utils.data_utils import frac_to_cart_coords
from src.utils.mol_utils import bb_to_rdkit_mols, compute_3d_conformer

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class ExtractFeatures(BaseLMDBProcessor):
    def __init__(self, cfg: DictConfig):
        process_cfg = cfg.preprocess
        extract_cfg = process_cfg.extract_feats
        super().__init__(process_cfg, extract_cfg)

    @staticmethod
    def get_cart_coords(data, c_rotmat=None):
        """
        Returns:
            cart_coords: numpy array of shape (n_atoms, 3)
        """
        def _get_cart_coords_from_bb(bb):
            cart_coords = frac_to_cart_coords(
                bb.frac_coords, 
                bb.lengths,
                bb.angles,
                bb.num_atoms
            )
            return cart_coords

        # Gather ground-truth Cartesian coordinates
        cart_coords = [_get_cart_coords_from_bb(bb) for bb in data.pyg_mols]
        cart_coords = torch.cat(cart_coords, dim=0) # [num_atoms, 3]
        if c_rotmat is not None:
            cart_coords = cart_coords @ c_rotmat.T

        return cart_coords

    @staticmethod
    def get_canonical_rotmat(lattice):
        """
        Returns:
            rotmat: numpy array of shape (3, 3)
                Rotation matrix to apply to Cartesian coordinates
            lattice_standard: pymatgen.core.lattice.Lattice
                Standardized lattice
        """
        lattice_standard = Lattice.from_parameters(*lattice.parameters)
        rotmat = np.linalg.inv(lattice_standard.matrix) @ lattice.matrix
        
        return rotmat, lattice_standard

    def suppress_warnings(self):
        warnings.filterwarnings("ignore")
        RDLogger.DisableLog('rdApp.*')
        pybel.ob.obErrorLog.SetOutputLevel(0)
    
    def transform_and_validate(self, data):
        # Normalize the lattice structure
        lattice = Lattice(data.cell).get_niggli_reduced_lattice()
        c_rotmat, canonical_lattice = self.get_canonical_rotmat(lattice)

        # Assemble core features into a dictionary
        pyg_mols = data.pyg_mols
        feats = {
            'cart_coords': self.get_cart_coords(data, c_rotmat).float(),
            'frac_coords': torch.cat([mol.frac_coords for mol in pyg_mols]).float(),
            'atom_types': torch.cat([mol.atom_types for mol in pyg_mols]).int(),
            'lattice': torch.tensor(canonical_lattice.parameters).float(),
            'cell': torch.tensor(canonical_lattice.matrix).float(),
            'bb_num_vec': torch.tensor([mol.num_atoms for mol in pyg_mols]).int(),
            'props': data.y # [1, prop_dim]
        }

        # Convert to RDKit molecules for chemical validation
        bb_mols, is_metal = bb_to_rdkit_mols(
            feats['atom_types'], feats['cart_coords'], feats['bb_num_vec']
        )
        feats.update({'bb_mols': bb_mols, 'is_metal': is_metal})

        # Generate 3D conformers from scratch, failing if any molecule is invalid
        for bb_idx, mol in enumerate(bb_mols):
            if is_metal[bb_idx]:
                continue # Will be loaded later in the pipeline
            else:
                mol.RemoveAllConformers()
                if not compute_3d_conformer(mol):
                    return None

        return feats

    def process_one(self, idx, value):
        self._suppress_warnings()
        try:
            data = pickle.loads(value)
            processed_feats = self._transform_and_validate(data)

            # Return None if validation failed, otherwise return the pickled features
            if processed_feats is None:
                return idx, None
            
            return idx, pickle.dumps(processed_feats)
        
        except Exception as e:
            # Catch any unexpected errors during processing
            print(f"Error processing index {idx}: {e}")
            return idx, None


@hydra.main(version_base=None, config_path="../../configs", config_name="preprocess.yaml")
def main(cfg: DictConfig):
    extractor = ExtractFeatures(cfg=cfg)

    # Set splits
    if cfg.preprocess.task == "gen":
        splits = ["train", "val"]
    elif cfg.preprocess.task == "csp":
        splits = ["train", "val", "test"]
    else:
        raise ValueError(f"Unknown task: {cfg.preprocess.task}")
    
    # Process
    for split in splits[::-1]:
        extractor.process(split=split)

if __name__ == "__main__":
    main()