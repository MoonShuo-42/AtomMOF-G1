"""
Convert BWDB dataset into Structure
"""
import pickle
import argparse
from pathlib import Path
from joblib import Parallel, delayed
from typing import Dict, Tuple, Literal

import numpy as np
from rdkit import Chem
from tqdm import tqdm

from src.data import const
from src.data.types import Block, Structure
from src.utils.lmdb_utils import read_lmdb, write_lmdb, get_all_keys


def load_lmdb_to_dict(lmdb_path: Path) -> Dict[str, bytes]:
    """Loads an LMDB into a dictionary."""
    data_dict = {}
    with read_lmdb(lmdb_path) as env:
        with env.begin() as txn:
            all_keys = get_all_keys(txn)
            for key_bytes in tqdm(all_keys, desc="Loading LMDB"):
                value = txn.get(key_bytes)
                key_str = key_bytes.decode('ascii')
                data_dict[key_str] = value
    return data_dict


def write_dict_to_lmdb(data_dict: Dict[str, bytes], lmdb_path: Path):
    """Writes a dictionary to an LMDB."""
    with write_lmdb(lmdb_path) as env:
        with env.begin(write=True) as txn:
            for key, value in data_dict.items():
                key_bytes = f"{key}".encode('ascii')
                txn.put(key_bytes, value)
    print(f"Data written to LMDB at {lmdb_path}")


def rdkit_mol_to_block(mol: Chem.Mol, mol_type: Literal['NODE', 'LINKER', 'SOLVENT', 'ADSORBATE']) -> Block:
    """Convert an RDKit Mol object to a Block dataclass."""
    # Set unk chiral id
    unk_chirality_tags = const.chirality_type_ids[const.unk_chirality_type]

    cart_coords = mol.GetConformer().GetPositions()
    atomic_numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
    formal_charges = np.array([atom.GetFormalCharge() for atom in mol.GetAtoms()])
    chirality_tags = np.array([const.chirality_type_ids.get(atom.GetChiralTag().name, unk_chirality_tags) for atom in mol.GetAtoms()])

    bond_indices_list = []
    bond_types_list = []
    for bond in mol.GetBonds():
        start_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_types = const.bond_type_ids.get(str(bond.GetBondType()), 0)
        bond_indices_list.append((start_idx, end_idx))
        bond_types_list.append(bond_types)
    
    if len(bond_indices_list) > 0:
        bond_indices = np.array(bond_indices_list).T  # Shape: (2, num_bonds)
        bond_types = np.array(bond_types_list)        # Shape: (num_bonds,)
    else: # No bonds
        bond_indices = np.empty((2, 0), dtype=int)
        bond_types = np.empty((0,), dtype=int)

    return Block(
        cart_coords=cart_coords,
        atomic_numbers=atomic_numbers,
        formal_charges=formal_charges,
        chirality_tags=chirality_tags,
        bond_indices=bond_indices,
        bond_types=bond_types,
        block_type=mol_type,
    )


def convert_data_to_structure(key: str, value: bytes) -> Tuple[str, bytes]:
    """Convert raw data bytes to Structure object."""
    data = pickle.loads(value)
    block_list = []
    for idx, mol in enumerate(data['bb_mols']):
        is_metal = data['is_metal'][idx]
        mol_type = 'NODE' if is_metal else 'LINKER'
        block = rdkit_mol_to_block(mol, mol_type)
        block_list.append(block)

    structure = Structure(
        blocks=block_list,
        cart_coords=data['cart_coords'].numpy(),
        frac_coords=data['frac_coords'].numpy(),
        lattice=data['lattice'].numpy(),
        cell=data['cell'].numpy()
    )
    return key, pickle.dumps(structure)


def main(args):
    lmdb_path = Path(args.lmdb_path)

    # Load LMDB
    data_dict = load_lmdb_to_dict(lmdb_path)

    # Convert to Structure objects in parallel
    results = Parallel(n_jobs=args.num_cpus)(
        delayed(convert_data_to_structure)(key, value) for key, value in tqdm(data_dict.items())
    )
    structure_dict = dict(results)

    # Save as LMDB
    name = lmdb_path.stem.split('_')[0]
    split = lmdb_path.stem.split('_')[1]
    output_path = lmdb_path.parent / f"{name}_structure_{split}.lmdb"
    write_dict_to_lmdb(structure_dict, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_path', type=str, required=True)
    parser.add_argument('--num_cpus', type=int, default=1)
    args = parser.parse_args()
    main(args)