"""
Creates a library of average conformers for metal-containing building blocks.
"""
import copy
import time
import pickle
import rootutils
import hydra
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from joblib import Parallel, delayed
from collections import defaultdict
from omegaconf import DictConfig

from src.preprocess.bwdb.base import BaseLMDBProcessor
from src.utils.lmdb_utils import read_lmdb, get_all_keys
from src.utils.mol_utils import is_metal_bb

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class CreateMetalLibrary:
    def __init__(self, cfg: DictConfig):
        process_cfg = cfg.preprocess
        metal_cfg = process_cfg.metal_library
        
        self.task = process_cfg.task
        self.lmdb_dir = Path(process_cfg.lmdb_dir)
        self.metal_dir = Path(metal_cfg.metal_dir)
        self.metal_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_cpus = metal_cfg.num_cpus
        self.input_lmdb_suffix = metal_cfg.input_lmdb_suffix

    @staticmethod
    def _extract_metal_bbs(value):
        """Extracts metal building blocks from a single MOF entry."""
        metal_mols = []
        feats = pickle.loads(value)
        for mol in feats['bb_mols']:
            if is_metal_bb(mol):
                metal_mols.append(mol)
        return metal_mols

    @staticmethod
    def _get_average_conformer(mol_list):
        """Aligns a list of molecules and computes their average structure."""
        ref_mol = mol_list[0]
        avg_mol = copy.deepcopy(ref_mol)
        avg_coords = np.zeros((avg_mol.GetNumAtoms(), 3))
        
        for mol in mol_list:
            rmsd, trans_mat, atom_map = AllChem.GetBestAlignmentTransform(prbMol=mol, refMol=ref_mol)
            
            # Apply transformation
            coords = mol.GetConformer().GetPositions()
            coords = np.append(coords, np.ones((coords.shape[0], 1)), axis=1)
            new_coords = coords.dot(trans_mat.T)[:, :3]
            
            # Reorder atoms to match reference
            order = [prb_idx for prb_idx, ref_idx in sorted(atom_map, key=lambda x: x[1])]
            reordered_coords = new_coords[order, :]
            avg_coords += reordered_coords
            
        avg_mol.GetConformer().SetPositions(avg_coords / len(mol_list))
        return avg_mol

    def run(self, split='train'):
        """Main method to create and save the metal BB library."""
        print(f"--- Creating Metal Library from '{split}' split for task '{self.task}' ---")
        start_time = time.time()

        src_path = self.lmdb_dir / self.task / f"{self.input_lmdb_suffix}_{split}.lmdb"
        
        # Read all values from the source LMDB
        values = []
        with read_lmdb(src_path) as env, env.begin() as txn:
            keys = get_all_keys(txn)
            for key in tqdm(keys, desc="Reading data"):
                values.append(txn.get(key))

        # Extract all metal BBs in parallel
        results = Parallel(n_jobs=self.num_cpus)(
            delayed(self._extract_metal_bbs)(v) for v in tqdm(values, desc="Extracting Metal BBs")
        )
        all_metal_mols = [mol for sublist in results for mol in sublist]

        # Group metal BBs by their SMILES string
        metal_mol_dict = defaultdict(list)
        for mol in all_metal_mols:
            smiles = Chem.MolToSmiles(mol, canonical=True)
            metal_mol_dict[smiles].append(mol)
        print(f"INFO:: Found {len(metal_mol_dict)} unique metal building block types.")

        # Create the average conformer for each unique BB
        metal_bb_library = {}
        for smi, mol_list in tqdm(metal_mol_dict.items(), desc="Creating average conformers"):
            if len(mol_list) > 1:
                avg_mol = self._get_average_conformer(mol_list)
                metal_bb_library[smi] = avg_mol
            else:
                metal_bb_library[smi] = mol_list[0]

        # Save the library to a pickle file
        save_path = self.metal_dir / f"metal_lib_{self.task}.pkl"
        print(f"INFO:: Saving metal library to {save_path}")
        with open(save_path, 'wb') as f:
            pickle.dump(metal_bb_library, f)

        print(f"INFO:: Library creation took {time.time() - start_time:.2f} seconds.")


class FilterByMetalLibrary(BaseLMDBProcessor):
    def __init__(self, cfg: DictConfig):
        process_cfg = cfg.preprocess
        metal_cfg = process_cfg.metal_library
        super().__init__(process_cfg, metal_cfg)

        # Load the pre-computed library of RDKit Mol objects
        library_path = Path(metal_cfg.metal_dir) / f"metal_lib_{self.task}.pkl"
        print(f"INFO:: Loading metal library from {library_path}")
        with open(library_path, 'rb') as f:
            self.metal_bb_library = pickle.load(f)
        
        self.valid_metal_smiles = set(self.metal_bb_library.keys())
        print(f"INFO:: Loaded {len(self.valid_metal_smiles)} valid metal BBs.")
        
        # Set the output suffix for the filtered files
        self.output_lmdb_suffix = metal_cfg.output_lmdb_suffix

    def process_one(self, idx, value):
        """
        Filters a MOF and replaces its metal BBs with standard conformers from the library.
        """
        try:
            feats = pickle.loads(value)
            new_bb_mols = []

            # Loop directly through each RDKit Mol object in the MOF
            for bb_idx, mol in enumerate(feats['bb_mols']):
                if feats['is_metal'][bb_idx]:
                    # This is a metal BB
                    smi = Chem.MolToSmiles(mol, canonical=True)

                    # Filter: Check if the metal BB is in our library
                    if smi not in self.valid_metal_smiles:
                        return idx, None

                    # Replace: Get the standard Mol object from the library
                    standard_mol = self.metal_bb_library[smi]
                    new_bb_mols.append(standard_mol)
                else:
                    # This is an organic BB, keep the original Mol object
                    new_bb_mols.append(mol)

            # Update the feats dictionary with the new list of Mol objects
            feats['bb_mols'] = new_bb_mols

            # Return the modified and re-pickled data
            return idx, pickle.dumps(feats)

        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            return idx, None
        

@hydra.main(version_base=None, config_path="../../configs", config_name="preprocess.yaml")
def main(cfg: DictConfig):
    # Create the metal library from the training data
    library_creator = CreateMetalLibrary(cfg=cfg)
    library_creator.run(split="train")

    # Filter other splits using the created library
    library_filter = FilterByMetalLibrary(cfg=cfg)
    
    if cfg.preprocess.task == "gen":
        splits_to_filter = ["train", "val"]
    elif cfg.preprocess.task == "csp":
        splits_to_filter = ["train", "val", "test"]
    else:
        raise ValueError(f"Unknown task: {cfg.preprocess.task}")

    for split in splits_to_filter:
        library_filter.process(split=split)

if __name__ == "__main__":
    main()