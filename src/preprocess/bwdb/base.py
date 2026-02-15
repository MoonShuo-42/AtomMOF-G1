"""
Abstract base class for preprocessing scripts.
"""
import time
import json
from abc import ABC, abstractmethod
from pathlib import Path

from tqdm import tqdm
from joblib import Parallel, delayed
from omegaconf import DictConfig

from src.utils.lmdb_utils import read_lmdb, write_lmdb, get_all_keys


class BaseLMDBProcessor(ABC):
    """
    Abstract base class for LMDB data processing.

    This class provides a template for reading data from a source LMDB,
    processing it in parallel, and writing the results to a destination LMdb.
    Subclasses must implement the `process_one` method to define the
    specific data transformation logic.
    """

    def __init__(self, process_cfg: DictConfig, step_cfg: DictConfig):
        self.task = process_cfg.task
        self.lmdb_dir = Path(process_cfg.lmdb_dir)

        self.num_cpus = step_cfg.num_cpus
        self.input_lmdb_suffix = step_cfg.input_lmdb_suffix
        self.output_lmdb_suffix = step_cfg.output_lmdb_suffix

    @abstractmethod
    def process_one(self, idx, value):
        """
        Process a single data entry. Must be implemented by subclasses.

        Args:
            idx (int or bytes): The key of the data entry.
            value (bytes): The pickled value of the data entry.

        Returns:
            A tuple of (idx, processed_value).
            `processed_value` should be bytes (pickled) or None if processing fails.
        """
        pass

    def _get_paths(self, split: str):
        """
        Determines source/destination paths.
        Can be overridden by subclasses for custom pathing logic.
        """
        output_dir = self.lmdb_dir / self.task
        output_dir.mkdir(parents=True, exist_ok=True)

        src_path = output_dir / f"{self.input_lmdb_suffix}_{split}.lmdb"
        dest_path = output_dir / f"{self.output_lmdb_suffix}_{split}.lmdb"
        return src_path, dest_path

    def _read_data(self, src_path, keys_to_read=None):
        """Reads data from the source LMDB, yielding key-value pairs."""
        print("INFO:: Reading source data...")
        with read_lmdb(src_path) as env:
            with env.begin() as txn:
                if keys_to_read is not None:
                    all_keys = [f"{key}".encode('ascii') for key in keys_to_read]
                    print(f"INFO:: Reading {len(all_keys)} specified entries.")
                else:
                    all_keys = get_all_keys(txn)

                for key_bytes in all_keys:
                    value = txn.get(key_bytes)
                    if value is None:
                        print(f"WARNING:: Key {key_bytes.decode('ascii', 'ignore')} not found, skipping.")
                        continue
                    
                    try:
                        idx = int(key_bytes.decode('ascii'))
                    except (UnicodeDecodeError, ValueError):
                        idx = key_bytes
                    
                    yield idx, value

    def _write_data(self, dest_path, results_iterator):
        """Writes processed data from an iterator to the destination LMDB."""
        print("INFO:: Writing processed features to new LMDB...")
        num_written = 0
        with write_lmdb(dest_path) as env:
            with env.begin(write=True) as txn:
                for idx, feats in results_iterator:
                    if feats is not None:
                        key_bytes = f"{idx}".encode('ascii')
                        txn.put(key_bytes, feats)
                        num_written += 1
        return num_written

    def _report_stats(self, split, num_processed, num_written, start_time, output_dir):
        """Calculates and saves processing statistics to a JSON file."""
        end_time = time.time()
        stats = {
            "samples_before": num_processed,
            "samples_after": num_written,
            "success_rate": (num_written / num_processed * 100) if num_processed > 0 else 0,
            "time_taken_seconds": end_time - start_time
        }
        stats_path = output_dir / f"{self.output_lmdb_suffix}_{split}_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        
        print("\n--- Processing Complete ---")
        print(f"Successfully processed and saved: {num_written}/{num_processed} entries.")
        print(f"Time taken: {end_time - start_time:.2f} seconds.")
        print(f"Stats saved to: {stats_path}")

    def process(self, split="train", keys_to_read=None):
        """
        Main processing pipeline.

        Args:
            split (str): The data split to process (e.g., 'train', 'val').
            keys_to_read (list, optional): A specific list of keys to read and process.
        """
        print(f"--- Starting processing for '{split}' split for task '{self.task}' ---")
        start_time = time.time()

        src_path, dest_path = self._get_paths(split)
        
        if keys_to_read is None:
            with read_lmdb(src_path) as src_env:
                num_src_entries = src_env.stat()['entries']
            total = num_src_entries
        else:
            total = len(keys_to_read)

        data_iterator = self._read_data(src_path, keys_to_read=keys_to_read)

        results = Parallel(n_jobs=self.num_cpus)(
            delayed(self.process_one)(idx, value) for idx, value in tqdm(data_iterator, desc="Processing", total=total)
        )

        num_written = self._write_data(dest_path, results)
        self._report_stats(split, total, num_written, start_time, dest_path.parent)