import pickle
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm
from lightning.pytorch.utilities import rank_zero_info

from src.data.feature.featurizer import Featurizer
from src.data.prior import PriorSampler, align_coords
from src.utils.lmdb_utils import get_all_keys, get_data, read_lmdb
from src.models.modules.utils import center_random_augmentation


class BWDBDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        lmdb_path: str, 
        featurizer: Featurizer, 
        prior_sampler: Optional[PriorSampler] = None,
        sample_limit: Optional[int] = None,
        max_num_atoms: Optional[int] = None,
        random_transform: bool = False,
        optimal_transport: bool = False,
    ):
        super().__init__()
        self.lmdb_path = Path(lmdb_path)
        self.featurizer = featurizer
        self.prior_sampler = prior_sampler
        self.sample_limit = sample_limit
        self.max_num_atoms = max_num_atoms
        self.random_transform = random_transform
        self.optimal_transport = optimal_transport
        self.split = Path(lmdb_path).stem.split('_')[-1]

        self.keys = []
        self.num_units = None
        self.env = None
        self.txn = None

        try:
            self._load_keys()
        except FileNotFoundError:
            print(f"Cache not found for {self.lmdb_path}. Generating it now...")
            self._generate_cache()
            self._load_keys()

    def _generate_cache(self):
        cache_path = self.lmdb_path.parent / f"{self.lmdb_path.stem}_metadata.pkl"

        if cache_path.exists():
            rank_zero_info(f"Cache already exists at {cache_path}. Skipping generation.")
            return
        
        metadata = {}
        env = read_lmdb(str(self.lmdb_path))
        with env.begin(write=False) as txn:
            keys = get_all_keys(txn)
            for key in tqdm(keys, desc="Scanning database"):
                data = get_data(txn, key)
                num_atoms = data.cart_coords.shape[0]
                metadata[key] = {'num_atoms': num_atoms}

        with open(cache_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        rank_zero_info(f"Cache generated and saved at {cache_path}.")
        env.close()
    
    def _load_keys(self):
        """Orchestrates loading, filtering, and limiting of dataset keys."""
        keys, num_units, total_keys = self._load_from_cache()
        
        self.keys = keys
        self.num_units = num_units
        
        self._log_initial_stats(total_keys)
        self._apply_sample_limit()

    def _load_from_cache(self):
        """
        Loads keys and atom counts from a pre-computed metadata cache.
        """
        cache_path = self.lmdb_path.parent / f"{self.lmdb_path.stem}_metadata.pkl"
        if not cache_path.exists():
            # The error message now points to the instance method
            raise FileNotFoundError(
                f"Metadata cache not found at '{cache_path}'.\n"
                f"Please run `your_dataset_instance.generate_cache()` first to create it."
            )
        
        rank_zero_info(f"Loading dataset index from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            metadata = pickle.load(f)

        total_keys = len(metadata)

        if self.max_num_atoms is None:
            all_keys = list(metadata.keys())
            all_atom_counts = [data['num_atoms'] for data in metadata.values()]
            return all_keys, all_atom_counts, total_keys

        filtered_keys = []
        atom_counts = []
        for key, data in metadata.items():
            if data['num_atoms'] <= self.max_num_atoms:
                filtered_keys.append(key)
                atom_counts.append(data['num_atoms'])
        
        return filtered_keys, atom_counts, total_keys

    def _log_initial_stats(self, total_keys):
        rank_zero_info(f"{total_keys} total datapoints found in {self.split} split.")
        
        if self.max_num_atoms is not None:
            rank_zero_info(f"Filtered to {len(self.keys)} samples with <= {self.max_num_atoms} atoms.")

    def _apply_sample_limit(self):
        """Applies the sample limit to the loaded keys and atom counts."""
        if self.sample_limit is None:
            return

        self.keys = self.keys[:self.sample_limit]
        if self.num_units is not None:
            self.num_units = self.num_units[:self.sample_limit]
            
        rank_zero_info(f"Limiting {self.split} split to {len(self.keys)} samples.")
    
    def _get_txn(self):
        """
        Initializes the LMDB environment and transaction for the current worker.
        This is called by __getitem__ to ensure each DataLoader worker has its own handle.
        """
        if self.txn is None:
            self.env = read_lmdb(str(self.lmdb_path))
            self.txn = self.env.begin(buffers=True, write=False)
        return self.txn

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> dict:
        """Retrieves and processes a single data item."""
        key = self.keys[idx]
        txn = self._get_txn()

        raw_data = get_data(txn, key)
        processed_data = self.featurizer.process(raw_data)

        # Center and augment data
        # TODO: Determine proper augmentation method
        atom_mask = torch.ones_like(processed_data['ref_element']).bool() # (num_atoms,)
        processed_data['cart_coords'] = center_random_augmentation(
            processed_data['cart_coords'][None],
            atom_mask[None],
            augmentation=self.random_transform,
            centering=True
        )[0]

        # Sample prior
        if self.prior_sampler is not None:
            prior_data = self.prior_sampler.sample(processed_data)
            processed_data.update(prior_data)

        # Align coords with optimal transport
        if self.optimal_transport:
            aligned_cart_coords = align_coords(
                coords_0=processed_data['cart_coords_0'],
                coords_1=processed_data['cart_coords'],
                atom_types=processed_data['ref_element'],
            )
            processed_data['cart_coords'] = aligned_cart_coords

        return processed_data

    def get_raw_data(self, idx: int) -> dict:
        """Retrieves the raw data without any processing."""
        key = self.keys[idx]
        txn = self._get_txn()

        raw_data = get_data(txn, key)
        return raw_data
    
    def _close_lmdb(self):
        if self.txn is not None:
            self.txn.abort()
            self.txn = None
        if self.env is not None:
            self.env.close()
            self.env = None

    def __del__(self):
        self._close_lmdb()