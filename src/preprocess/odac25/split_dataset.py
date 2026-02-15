import os
import shutil
import argparse
import lmdb
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple

from src.utils.lmdb_utils import read_lmdb, write_lmdb, get_all_keys


def write_data_to_lmdb(db_path: Path, data_list: List[Tuple[bytes, bytes]]):
    """
    Writes a list of (key, value) tuples to a new LMDB database.
    
    Args:
        db_path (Path): The path to the new LMDB database (directory).
        data_list (list): A list of (key_bytes, value_bytes) tuples.
    """
    env = write_lmdb(db_path)
    with env.begin(write=True) as txn:
        for key, value in tqdm(data_list, desc=f"Writing to {db_path.name}"):
            txn.put(key, value)
    env.close()

def main(dataset_dir: str, val_keys_path: str):
    data_dir = Path(dataset_dir)
    val_keys_path = Path(val_keys_path)

    # Load validation keys
    print(f"Loading validation keys from {val_keys_path.name}...")
    target_val_keys = set(np.load(val_keys_path))
    print(f"Loaded {len(target_val_keys)} keys for validation split.")

    # Define file paths
    orig_train_path = data_dir / "odac25_train.lmdb"
    orig_val_path = data_dir / "odac25_val.lmdb"
    new_test_path = data_dir / "odac25_test.lmdb"
    
    # Temporary paths for the new split
    tmp_train_path = data_dir / "odac25_train_TMP.lmdb"
    tmp_val_path = data_dir / "odac25_val_TMP.lmdb"

    # Rename validation set to test set
    if not orig_val_path.exists():
        print(f"Error: Original validation set not found at {orig_val_path}")
        print("Skipping rename step.")
    elif new_test_path.exists():
        print(f"Warning: Test set already exists at {new_test_path}")
        print("Skipping rename step.")
    else:
        orig_val_path.rename(new_test_path)
        print(f"Renamed: {orig_val_path.name} -> {new_test_path.name}")

    # Split original train set into new train and val sets
    if not orig_train_path.exists():
        print(f"Error: Original training set not found at {orig_train_path}")
        sys.exit(1)
        
    if tmp_train_path.exists() or tmp_val_path.exists():
        print(f"Error: Temporary files already exist. Please remove them first:")
        print(f" - {tmp_train_path}")
        print(f" - {tmp_val_path}")
        sys.exit(1)

    # Read all data from the original training LMDB
    print(f"Reading all data from {orig_train_path.name}...")
    
    new_train_data = []
    new_val_data = []
    
    env = read_lmdb(orig_train_path)
    with env.begin() as txn:
        keys = get_all_keys(txn)
        for key in tqdm(keys, desc="Processing samples"):
            value = txn.get(key)
            key_int = int(key)

            if key_int in target_val_keys:
                new_val_data.append((key, value))
            else:
                new_train_data.append((key, value))
    env.close()
    
    total_samples = len(new_train_data) + len(new_val_data)
    print(f"Processed {total_samples} total samples.")
    print(f"Splitting into {len(new_train_data)} new train samples and {len(new_val_data)} new val samples.")
    
    # Sanity check
    if len(new_val_data) != len(target_val_keys):
        print(f"Warning: Found {len(new_val_data)} matches, but provided key file had {len(target_val_keys)} keys.")
        print("Some keys in the .npy file might not exist in the source LMDB.")

    # Write to temporary LMDBs
    write_data_to_lmdb(tmp_train_path, new_train_data)
    write_data_to_lmdb(tmp_val_path, new_val_data)

    # Delete original train LMDB
    try:
        shutil.rmtree(orig_train_path) if os.path.isdir(orig_train_path) else os.remove(orig_train_path)
        print(f"Removed original: {orig_train_path.name}")
    except OSError as e:
        print(f"Error removing {orig_train_path}: {e}")
        print("You may need to remove it manually.")
        sys.exit(1)

    # Rename temporary files to final names
    tmp_train_path.rename(orig_train_path)
    print(f"Renamed: {tmp_train_path.name} -> {orig_train_path.name}")
    
    tmp_val_path.rename(orig_val_path)
    print(f"Renamed: {tmp_val_path.name} -> {orig_val_path.name}")
    
    print("\n Split complete!")
    print(f"Test set:  {new_test_path.name}")
    print(f"Train set: {orig_train_path.name} ({len(new_train_data)} samples)")
    print(f"Valid set: {orig_val_path.name} ({len(new_val_data)} samples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--val_keys_path", type=str, required=True, default="/path/to/val_keys.npy")
    args = parser.parse_args()

    main(args.dataset_dir, args.val_keys_path)