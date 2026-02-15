"""
Write cif file and extract mofid. Adapted from MOFDiff.
"""
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed

from ase.io import write
from ase.build import niggli_reduce
from mofid.run_mofid import cif2mofid
from fairchem.core.datasets import AseDBDataset


def save_mofid(cif_path, save_path):
    """
    Runs the MOFid algorithm on a single CIF file and saves the results.

    Args:
        cif_path (Path): Path to the input CIF file.
        save_path (Path): Directory where MOFid results should be saved.
    """
    cif_id = cif_path.stem
    output_path = save_path / cif_id
    if (output_path / "orig_mol.cif").exists():
        # Skip if already processed
        return

    output_path.mkdir(exist_ok=True)
    processing_cif_path = cif_path

    cif2mofid(processing_cif_path, output_path)


def main(dataset_dir, num_workers, use_niggli_reduce):
    print(f"--- Processing Setup ---")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Number of workers: {num_workers}")
    print(f"Niggli reduction: {'Enabled' if use_niggli_reduce else 'Disabled'}")
    print(f"------------------------")

    dataset_dir = Path(dataset_dir)

    # Load dataset
    print("Loading dataset...")
    dataset = AseDBDataset({'src': str(dataset_dir)})
    final_frame_indices = np.load(dataset_dir / "final_frame_indices.npy")

    # Save as cif
    dir_suffix = "_niggli" if use_niggli_reduce else ""
    cif_dir = dataset_dir / f"cif{dir_suffix}"
    cif_dir.mkdir(exist_ok=True)
    print(f"Saving {len(final_frame_indices)} structures to CIF files in {cif_dir}...")

    for i in tqdm(final_frame_indices, desc="Saving CIF files"):
        atoms = dataset.get_atoms(i)
        if use_niggli_reduce:
            niggli_reduce(atoms)

        cif_filename = cif_dir / f"ase_index_{i}.cif"

        # NOTE: Only use framework atoms
        atoms_mof = atoms[atoms.get_tags() == 0] # exclude adsorbates 
        if not cif_filename.exists():
            write(str(cif_filename), atoms_mof, format="cif")

    # Extract mofid
    mofid_dir = dataset_dir / f"mofid{dir_suffix}"
    mofid_dir.mkdir(exist_ok=True)
    print(f"Extracting MOFIDs to {mofid_dir}...")

    cif_files = sorted(list(cif_dir.glob("*.cif")))

    def process_one(cif_file):
        material_id = cif_file.stem
        try:
            save_mofid(cif_file, mofid_dir)
            return material_id, False # No error
        except Exception as e:
            print(f"Error processing {material_id}: {e}")
            return material_id, True # Error occurred
    
    results = Parallel(n_jobs=num_workers)(
        delayed(process_one)(cif_file) for cif_file in tqdm(cif_files, desc="Processing CIF files")
    )

    # Report failures
    failed_ids = [material_id for material_id, error in results if error]
    if failed_ids:
        print(f"\nProcessing failed for {len(failed_ids)} structures.")
        failed_log_path = mofid_dir / "failed_ids.txt"
        with open(failed_log_path, "w") as f:
            f.write("\n".join(failed_ids))
        print(f"List of failed IDs saved to: {failed_log_path}")
    else:
        print("\nSuccessfully processed all structures.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--no_niggli_reduce", dest='use_niggli_reduce', action='store_false')
    args = parser.parse_args()
    
    main(args.dataset_dir, args.num_workers, args.use_niggli_reduce)