import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from fairchem.core.datasets import AseDBDataset


def main(dataset_dir, output_filename):
    # Load the dataset
    print(f"Loading dataset from: {dataset_dir}")
    dataset = AseDBDataset({"src": dataset_dir})
    print(f"Successfully loaded dataset with {len(dataset)} total entries.")

    # Find the final frame for each trajectory
    final_frames = {}
    for i in tqdm(range(len(dataset)), desc="Processing entries"):
        atoms = dataset.get_atoms(i)
        info = atoms.info
        name = info.get('name')
        fid = info.get('fid')

        if name is None or fid is None:
            continue

        if name not in final_frames or fid > final_frames[name][0]:
            final_frames[name] = (fid, i)
    
    # Extract and sort the final indices
    final_indices = np.array(sorted([value[1] for value in final_frames.values()]), dtype=np.int64)
    print(f"Found {len(final_frames)} unique trajectories.")

    # Save the indices to a file
    output_file_path = Path(dataset_dir) / output_filename
    np.save(output_file_path, final_indices)

    print(f"Successfully saved {len(final_indices)}/{len(dataset)} indices to: {output_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--output_filename", type=str, default="final_frame_indices")
    args = parser.parse_args()
    
    main(args.dataset_dir, args.output_filename)