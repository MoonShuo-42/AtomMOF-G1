"""
Creates a library of average conformers for metal-containing building blocks.
Then replaces metal building blocks in the dataset with their average conformers.
"""
import copy
import argparse
import json
import random
import numpy as np
import lmdb
import pickle
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from collections import defaultdict
from typing import List, Dict, Any

from src.utils.lmdb_utils import read_lmdb, write_lmdb, get_all_keys
from src.data.types import (
    Fragment, DecomposedMOF, MetalNodeVariant, MetalLibrary
)

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol


def collect_node_fragments(value: bytes) -> List[Fragment]:
    """Extracts all metal building blocks from a single MOF entry."""
    node_fragments = []
    mof: DecomposedMOF = pickle.loads(value)
    for idx, fragment in enumerate(mof.fragments):
        if fragment.block_type == "NODE":
            node_fragments.append(fragment)
    return node_fragments


def group_fragments_by_smiles(all_node_fragments: List[Fragment]) -> MetalLibrary:
    """Groups metal building blocks by their canonical SMILES."""
    node_fragment_dict = defaultdict(list)
    for fragment in all_node_fragments:
        variant = MetalNodeVariant.from_fragment(fragment)
        node_fragment_dict[fragment.canonical_smiles].append(variant)
    return node_fragment_dict


# def get_average_conformer(mol_list: List[Mol]) -> Mol:
#     """Aligns a list of molecules and computes their average structure."""
#     ref_mol = mol_list[0]
#     avg_mol = copy.deepcopy(ref_mol)
#     avg_coords = np.zeros((avg_mol.GetNumAtoms(), 3))
    
#     for mol in tqdm(mol_list, desc="Aligning conformers"):
#         rmsd, trans_mat, atom_map = AllChem.GetBestAlignmentTransform(prbMol=mol, refMol=ref_mol)
        
#         # Apply transformation
#         coords = mol.GetConformer().GetPositions()
#         coords = np.append(coords, np.ones((coords.shape[0], 1)), axis=1)
#         new_coords = coords.dot(trans_mat.T)[:, :3]
        
#         # Reorder atoms to match reference
#         order = [prb_idx for prb_idx, ref_idx in sorted(atom_map, key=lambda x: x[1])]
#         reordered_coords = new_coords[order, :]
#         avg_coords += reordered_coords
        
#     avg_mol.GetConformer().SetPositions(avg_coords / len(mol_list))
#     return avg_mol


# def create_metal_library(metal_mol_dict: Dict[str, List[Mol]], max_conformers: int, num_cpus: int) -> Dict[str, Mol]:
#     """Creates a library of average conformers for each unique metal building block."""

#     def _process_entry(item: Tuple[str, List[Mol]]) -> Tuple[str, Mol]:
#         smi, mol_list = item
#         if len(mol_list) > 1:
#             mol_list = random.sample(mol_list, max_conformers) if len(mol_list) > max_conformers else mol_list
#             avg_mol = get_average_conformer(mol_list)
#         else:
#             avg_mol = mol_list[0]
#         return smi, avg_mol

#     results = p_map(_process_entry, metal_mol_dict.items(), num_cpus=num_cpus, desc="Creating metal library")
#     metal_library = {smi: avg_mol for smi, avg_mol in results}
#     return metal_library


def load_lmdb_to_dict(block_lmdb_path: Path) -> Dict[str, bytes]:
    """Loads an LMDB into a dictionary."""
    data_dict = {}
    env = read_lmdb(block_lmdb_path)
    with env.begin() as txn:
        all_keys = get_all_keys(txn)
        for key_bytes in all_keys:
            value = txn.get(key_bytes)
            key_str = key_bytes.decode('ascii')
            data_dict[key_str] = value
    env.close()
    return data_dict


def save_as_pickle(obj: Any, path: Path):
    """Saves an object as a pickle file."""
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def main(args):
    # Load LMDB
    block_lmdb_path = Path(args.block_lmdb_dir) / "building_blocks.lmdb"
    data_dict = load_lmdb_to_dict(block_lmdb_path)

    # Step 1: Extract all metal building blocks
    all_node_fragments = Parallel(n_jobs=args.num_cpus)(
        delayed(collect_node_fragments)(v) for v in tqdm(data_dict.values(), desc="Collecting metal blocks")
    )
    all_node_fragments = [frag for sublist in all_node_fragments for frag in sublist]

    # Step 2: Group metal building blocks by SMILES
    node_fragment_dict = group_fragments_by_smiles(all_node_fragments)
    print(f"INFO:: Found {len(node_fragment_dict)} unique metal building block types.")

    # Step 3: Create average conformer library
    # metal_bb_library = create_metal_library(metal_mol_dict, args.max_conformers, args.num_cpus)
    
    # Save dict and library
    output_dir = block_lmdb_path.parents[3] / "metal"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save dictionary
    split = block_lmdb_path.parents[2].name
    dict_path = output_dir / f"metal_dict_{split}.pkl"
    save_as_pickle(node_fragment_dict, dict_path)
    print(f"INFO:: Saved metal building block dictionary to {dict_path}")

    # Save library
    # lib_path = output_dir / f"metal_lib_{split}.pkl"
    # save_as_pickle(metal_bb_library, lib_path)
    # print(f"INFO:: Saved metal building block library to {lib_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--block_lmdb_dir", type=str, required=True, default="/path/to/blocks/")
    parser.add_argument("--num_cpus", type=int, default=32)
    parser.add_argument("--max_conformers", type=int, default=5000)
    args = parser.parse_args()
    main(args)