"""
Finalizes the dataset by generating local building blocks and assembling
them into Structure objects with ground-truth targets.
"""
import json
import pickle
import warnings
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
from rdkit import RDLogger
from openbabel import pybel
from p_tqdm import p_imap
from tqdm import tqdm

from src.data.types import DecomposedMOF, MOFStructure, Block
from src.data.generator import NodeGenerator, RDKitBlockGenerator
from src.utils.lmdb_utils import read_lmdb, write_lmdb, get_all_keys


_NODE_GENERATOR: Optional[NodeGenerator] = None
_RDKIT_GENERATOR: Optional[RDKitBlockGenerator] = None


def suppress_warnings():
    warnings.filterwarnings("ignore")
    RDLogger.DisableLog('rdApp.*')
    pybel.ob.obErrorLog.SetOutputLevel(0)


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


def cart_to_frac(cart_coords: np.ndarray, cell: np.ndarray) -> np.ndarray:
    return np.linalg.solve(cell.T, cart_coords.T).T


def _get_generators(metal_library_path: str):
    """
    Lazy loader for generators (Alternative to 'initializer').
    Since p_tqdm doesn't accept an initializer arg easily, we use this 
    to ensure we only load the library once per worker.
    """
    global _NODE_GENERATOR, _RDKIT_GENERATOR
    
    if _NODE_GENERATOR is None:
        _NODE_GENERATOR = NodeGenerator(metal_library_path)

    if _RDKIT_GENERATOR is None:
        _RDKIT_GENERATOR = RDKitBlockGenerator()

    suppress_warnings()
        
    return _NODE_GENERATOR, _RDKIT_GENERATOR


def process_single_entry_wrapper(args_tuple):
    """
    Unpacks arguments because p_umap passes a single item at a time.
    """
    return process_single_entry(*args_tuple)


def process_single_entry(
    key: str, 
    serialized_data: bytes,
    metal_library_path: str,
) -> Tuple[str, Union[MOFStructure, str], bool]:
    """
    Processes a single DecomposedMOF into a Structure.
    """
    node_generator, rdkit_generator = _get_generators(metal_library_path)

    try:
        mof: DecomposedMOF = pickle.loads(serialized_data)
        
        # Generate blocks
        blocks: List[Block] = []
        for fragment in mof.fragments:
            if fragment.block_type == "NODE":
                block = node_generator.generate_block_from_fragment(fragment)
            else:
                block = rdkit_generator.generate_block_from_fragment(fragment)
            
            if block is None:
                return key, f"Failed to generate block for fragment type {fragment.block_type}", True
            
            blocks.append(block)

        # Collect coordinates and atomic numbers
        contiguous_cart_coords = np.concatenate([f.cart_coords_contiguous for f in mof.fragments], axis=0)
        original_cart_coords = np.concatenate([f.cart_coords_original for f in mof.fragments], axis=0)
        frac_coords = cart_to_frac(original_cart_coords, mof.cell)
        atomic_numbers = np.concatenate([fragment.atomic_numbers for fragment in mof.fragments], axis=0)

        assert sum([len(b.cart_coords) for b in blocks]) == contiguous_cart_coords.shape[0], \
            "Mismatch in number of atoms between blocks and total structure."
        
        structure = MOFStructure(
            blocks=blocks,
            atomic_numbers=atomic_numbers,
            cart_coords=contiguous_cart_coords,
            frac_coords=frac_coords,
            original_cart_coords=original_cart_coords,
            lattice=mof.lattice,
            cell=mof.cell,
            info=mof.info,
        )
        
        return key, structure, False

    except Exception as e:
        return key, str(e), True


def main(args):
    print(f"--- Dataset Construction Setup ---")
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Metal Library: {args.metal_library_path}")
    print(f"Workers: {args.num_cpus}")
    print(f"----------------------------------")
    
    dataset_dir = Path(args.dataset_dir)
    blocks_lmdb_path = dataset_dir / "blocks" / "building_blocks.lmdb"
    
    print("Reading input fragments from LMDB...")
    data_dict = load_lmdb_to_dict(blocks_lmdb_path)
    print(f"Found {len(data_dict)} mofs to process.")

    tasks = [
        (key, val, str(args.metal_library_path)) 
        for key, val in data_dict.items()
    ]

    successful_structures: Dict[str, MOFStructure] = {}
    failed_structures: Dict[str, str] = {}

    results_iterator = p_imap(
        process_single_entry_wrapper, 
        tasks, 
        num_cpus=args.num_cpus,
        desc="Generating Structures"
    )

    for key, data, error in results_iterator:
        if error:
            failed_structures[key] = data
        else:
            successful_structures[key] = data

    print(f"Successfully processed {len(successful_structures)}/{len(data_dict)} mofs.")

    # Save Outputs
    output_dir = dataset_dir / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_lmdb_path = output_dir / "structures.lmdb"
    if successful_structures:
        env = write_lmdb(output_lmdb_path)
        with env.begin(write=True) as txn:
            # We iterate normally here as data is already in memory
            for key, structure in tqdm(successful_structures.items(), desc="Writing to LMDB"):
                key_bytes = key.encode('ascii')
                value_bytes = pickle.dumps(structure)
                txn.put(key_bytes, value_bytes)
        env.close()
        print(f"Saved processed structures to {output_lmdb_path}")

    if failed_structures:
        failure_path = output_dir / "generation_failures.json"
        with open(failure_path, "w") as f:
            json.dump(failed_structures, f, indent=4)
        print(f"{len(failed_structures)} / {len(data_dict)} failures detected during processing.")
        print(f"Saved failure log to {failure_path}")
    else:
        print("No failures detected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate final Structures for training.")
    parser.add_argument("--dataset_dir", type=str, required=True, default="/path/to/dataset_dir/")
    parser.add_argument("--metal_library_path", type=str, required=True, default="/path/to/metal_library.pkl")
    parser.add_argument("--num_cpus", type=int, default=32)
    
    args = parser.parse_args()
    main(args)