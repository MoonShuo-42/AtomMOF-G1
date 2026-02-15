import os
import argparse
import random
import copy
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple

from src.utils.lmdb_utils import read_lmdb, get_all_keys, write_lmdb
from src.utils.mol_utils import convert_xyz_to_canonical_smiles
from src.data.types import Block, MetalNodeVariant, Structure


def replace_node_with_variant(node_block: Block, variant: MetalNodeVariant) -> Block:
    """
    Creates a copy of the block and updates it with variant data.
    """
    new_block = copy.deepcopy(node_block)
    
    # Update properties
    new_block.atomic_numbers = variant.atomic_numbers
    new_block.cart_coords = variant.cart_coords
    
    return new_block


def randomize_coordinates(node_block: Block) -> Block:
    """
    Randomly initializes coordinates for nodes missing from the library.
    Using standard normal distribution scaled down slightly.
    """
    new_block = copy.deepcopy(node_block)
    
    noise = np.random.randn(*node_block.cart_coords.shape)
    new_block.cart_coords = noise.astype(np.float32)
    
    return new_block

def process_structure(
    structure: Structure, 
    key: str, 
    metal_library: Dict[str, List[MetalNodeVariant]],
    missing_log: List[str]
) -> Tuple[Structure, bool]:
    """
    Iterates through blocks in a structure and replaces NODE blocks.
    """
    new_blocks_list = []
    is_fully_replaced = True # Assumes success unless a node is missing

    for idx, block in enumerate(structure.blocks):
        # ----- Skip Non-Nodes -----
        if block.block_type != "NODE":
            new_blocks_list.append(block)
            continue

        # ----- Process Node Block -----
        # Convert to SMILES
        try:
            smi = convert_xyz_to_canonical_smiles(
                atom_types=block.atomic_numbers,
                positions=block.cart_coords,
                add_bonds=False
            )
        except Exception as e:
            smi = "INVALID_SMILES"

        # Check Library and Replace
        if smi in metal_library:
            # Sample a variant
            variant = random.choice(metal_library[smi])
            new_block = replace_node_with_variant(block, variant)
            new_blocks_list.append(new_block)
        else:
            # Failure Case: Log and Randomize
            is_fully_replaced = False

            # Log missing node
            log_entry = f"Key: {key} | Block Idx: {idx} | SMILES: {smi}"
            missing_log.append(log_entry)
            
            new_block = randomize_coordinates(block)
            new_blocks_list.append(new_block)

    new_structure = copy.deepcopy(structure)
    new_structure.blocks = new_blocks_list
    
    return new_structure, is_fully_replaced

def main(args):
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Setup Paths
    data_dir = Path(args.data_dir)
    metal_lib_path = data_dir / args.metal_lib_path
    input_lmdb_path = data_dir / args.input_lmdb
    output_lmdb_path = data_dir / args.output_lmdb
    log_path = data_dir / args.log_file

    print(f"INFO:: Loading Metal Library from {metal_lib_path}")
    with open(metal_lib_path, "rb") as f:
        metal_library = pickle.load(f)
    print(f"Loaded {len(metal_library)} unique node SMILES keys.")

    print(f"INFO:: Processing Data from {input_lmdb_path}")
    
    # Storage for processed data and logs
    new_data_list = []
    missing_log = []

    # Read LMDB
    env = read_lmdb(input_lmdb_path)

    fully_replaced_count = 0

    with env.begin() as txn:
        keys = get_all_keys(txn)
        print(f"INFO:: Found {len(keys)} structures.")
        
        for key in tqdm(keys, desc="Replacing Nodes"):
            # Deserialize
            raw_data = txn.get(key)
            if raw_data is None:
                continue
                
            structure: Structure = pickle.loads(raw_data)
            
            # Process
            new_structure, is_fully_replaced = process_structure(structure, key, metal_library, missing_log)
            new_data_list.append((key, new_structure))

            if is_fully_replaced:
                fully_replaced_count += 1

    env.close()

    # Save Output LMDB
    print(f"INFO:: Saving {len(new_data_list)} structures to {output_lmdb_path}.")
    dest_env = write_lmdb(output_lmdb_path)
    with dest_env.begin(write=True) as txn:
        for key, structure in tqdm(new_data_list, desc="Writing LMDB"):
            serialized_data = pickle.dumps(structure)
            txn.put(key, serialized_data)
    dest_env.close()

    # ----- Log Missing Nodes -----
    print(f"INFO:: {fully_replaced_count}/{len(new_data_list)} structures fully replaced without missing nodes.")
    if missing_log:
        print(f"INFO:: Saving failure log to {log_path}")
        with open(log_path, "w") as f:
            f.write("\n".join(missing_log))
    else:
        print("INFO:: Success: All nodes were found in the library.")

    print("\nProcessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/odac25")
    parser.add_argument("--metal_lib_path", type=str, default="metal/metal_dict_train.pkl")
    parser.add_argument("--input_lmdb", type=str, default="odac25_structure_test.lmdb")
    parser.add_argument("--output_lmdb", type=str, default="odac25_replaced_test.lmdb")
    parser.add_argument("--log_file", type=str, default="missing_nodes_log.txt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)