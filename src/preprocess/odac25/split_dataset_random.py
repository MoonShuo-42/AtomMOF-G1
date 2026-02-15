import os
import shutil
import random
import argparse
import lmdb
import sys
from pathlib import Path
from tqdm import tqdm

from src.utils.lmdb_utils import read_lmdb, write_lmdb, get_all_keys


def write_data_to_lmdb(db_path, data_list):
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

def main(dataset_dir, val_split, seed):
    data_dir = Path(dataset_dir)

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
    all_data = []
    env = read_lmdb(orig_train_path)
    with env.begin() as txn:
        keys = get_all_keys(txn)
        for key in tqdm(keys, desc="Reading samples"):
            value = txn.get(key)
            all_data.append((key, value))
    env.close()
    
    total_samples = len(all_data)
    print(f"Read {total_samples} total samples.")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(all_data)

    val_count = int(total_samples * val_split)
    train_count = total_samples - val_count
    
    new_val_data = all_data[:val_count]
    new_train_data = all_data[val_count:]
    
    print(f"Splitting into {train_count} new train samples and {val_count} new val samples.")

    # Write to temporary LMDBs
    write_data_to_lmdb(tmp_train_path, new_train_data)
    write_data_to_lmdb(tmp_val_path, new_val_data)

    # Delete original train LMDB
    try:
        os.remove(orig_train_path)
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
    print(f"Train set: {orig_train_path.name} ({train_count} samples)")
    print(f"Valid set: {orig_val_path.name} ({val_count} samples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args.dataset_dir, args.val_split, args.seed)