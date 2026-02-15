"""
Check MOF validity. Filter out MOFs that do not meet the criteria.
"""
import time
import json
import hydra
import pickle
import rootutils
import warnings
from tqdm import tqdm
from pathlib import Path
from collections import Counter
from joblib import Parallel, delayed
from omegaconf import DictConfig
from pymatgen.core import Structure

from src.preprocess.bwdb.base import BaseLMDBProcessor
from src.utils.lmdb_utils import read_lmdb
from src.utils.mofchecker_utils import is_mof_valid

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class CheckMOFValidity(BaseLMDBProcessor):
    def __init__(self, cfg: DictConfig):
        process_cfg = cfg.preprocess
        mof_validity_cfg = process_cfg.mof_validity

        # Initialize the base class
        super().__init__(process_cfg, mof_validity_cfg)

    def process_one(self, idx, value):
        """
        Checks a single MOF for validity.

        Returns:
            A tuple containing two elements:
            1. (idx, value or None): The data to be written to the new LMDB.
            2. status (str): The outcome of the check ('success', 'invalid', 'exception').
        """
        warnings.filterwarnings("ignore")
        try:
            feats = pickle.loads(value)
            structure = Structure(
                coords=feats['frac_coords'],
                species=feats['atom_types'],
                lattice=feats['cell'],
                coords_are_cartesian=False,
            )
            valid, _ = is_mof_valid(structure)
            if valid:
                return (idx, value), 'success'
            else:
                return (idx, None), 'invalid'
        except Exception:
            return (idx, None), 'exception'

    def process(self, split="train"):
        """
        Overrides the base 'process' method to handle custom status tracking.
        """
        print(f"--- Starting validity check for '{split}' split for task '{self.task}' ---")
        start_time = time.time()

        # Use the base class helper to get paths
        src_path, dest_path, _ = self._get_paths_and_limit(split)

        # Total number of source entries for stats and progress bar
        with read_lmdb(src_path) as src_env:
            total_entries = src_env.stat()['entries']

        # Get a data iterator
        data_iterator = self._read_data(src_path)

        # Run the parallel processing job
        results = Parallel(n_jobs=self.num_cpus)(
            delayed(self.process_one)(idx, value)
            for idx, value in tqdm(data_iterator, desc="Processing", total=total_entries)
        )

        # Unpack results and count statuses
        # 'results' is a list of [((idx, value), status), ...]
        writable_data = [res[0] for res in results]
        status_list = [res[1] for res in results]
        status_counts = Counter(status_list)
        print(f"INFO:: Status counts: {dict(status_counts)}")

        # Save failure counts to JSON
        stats_path = dest_path.parent / f"mofchecker_failure_counts_{split}.json"
        with open(stats_path, "w") as f:
            json.dump(status_counts, f, indent=2)
        print(f"INFO:: Failure counts saved to {stats_path}")

        # Write the filtered data
        num_written = self._write_data(dest_path, writable_data)

        # Report final statistics
        self._report_stats(split, total_entries, num_written, start_time, dest_path.parent)


@hydra.main(version_base=None, config_path="../../configs", config_name="preprocess.yaml")
def main(cfg: DictConfig):
    checker = CheckMOFValidity(cfg=cfg)

    if cfg.preprocess.task == "gen":
        splits = ["train", "val"]
    elif cfg.preprocess.task == "csp":
        splits = ["train", "val", "test"]
    else:
        raise ValueError(f"Unknown task: {cfg.preprocess.task}")
    
    for split in splits[::-1]:
        checker.process(split=split)

if __name__ == "__main__":
    main()