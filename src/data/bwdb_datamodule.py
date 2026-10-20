from pathlib import Path
from functools import partial
from typing import List, Optional, Dict

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from lightning.pytorch.utilities import rank_zero_info

from src.data.pad import pad_dim
from src.data.feature.featurizer import Featurizer
from src.data.bwdb_dataset import BWDBDataset
from src.data.sampler import DynamicBatchSampler, OverfitSampler
from src.data.prior.prior_sampler import PriorSampler


def collate_fn(batch: List[Dict[str, Tensor]], *, compact_batch=False, prepare_g1_metadata=False, prepare_g1_blocks=False) -> Dict[str, Tensor]:
    """
    Collates a batch of featurized Structure dictionaries for a Transformer.
    """
    if not batch:
        return {}

    # Define key types
    atom_keys_N = ['ref_element', 'ref_charge', 'ref_chirality', 'block_index', 'block_type']
    atom_keys_ND = ['ref_pos', 'cart_coords', 'frac_coords', 'cart_coords_0']
    atom_keys_NN = [] if compact_batch else ['atom_to_token', 'token_to_rep_atom']
    structure_keys = ['lattice', 'cell', 'lattice_0']
    
    # Define all output keys
    all_output_keys = (
        atom_keys_N + atom_keys_ND + atom_keys_NN + structure_keys +
        ['atom_pad_mask', 'bond_types'] + ([] if compact_batch else ['bond_indices'])
    )
    collated = {key: [] for key in all_output_keys}

    # Find max_atoms
    max_atoms = max(data['ref_element'].shape[0] for data in batch)
    
    # Iterate over each item in the batch
    for data in batch:
        num_atoms = data['ref_element'].shape[0]
        atom_pad_len = max_atoms - num_atoms

        ##### Create atom_pad_mask #####
        atom_pad_mask = pad_dim(torch.ones(num_atoms), dim=0, pad_len=atom_pad_len)
        collated['atom_pad_mask'].append(atom_pad_mask)

        ##### Create bond matrices #####
        if not compact_batch:
            bond_indices_matrix = torch.zeros((max_atoms, max_atoms), dtype=torch.long)
        bond_types_matrix = torch.zeros((max_atoms, max_atoms), dtype=torch.long)
        
        indices = data['bond_indices']
        types = data['bond_types']
        
        if types.shape[0] > 0:
            if not compact_batch:
                bond_indices_matrix[indices[0], indices[1]] = 1
                bond_indices_matrix[indices[1], indices[0]] = 1  # Symmetric
            
            bond_types_matrix[indices[0], indices[1]] = types
            bond_types_matrix[indices[1], indices[0]] = types  # Symmetric
        
        if not compact_batch:
            collated['bond_indices'].append(bond_indices_matrix)
        collated['bond_types'].append(bond_types_matrix)

        ##### Pad atom features: (N, ...) -> (max_N, ...) #####
        for key in atom_keys_N + atom_keys_ND:
            if key in data:
                tensor = data[key]
                padded_tensor = pad_dim(tensor, dim=0, pad_len=atom_pad_len)
                collated[key].append(padded_tensor)

        ##### Pad 2D square matrices: (N, N) -> (max_N, max_N) #####
        for key in atom_keys_NN:
            if key in data:
                tensor = data[key]
                padded_tensor = pad_dim(tensor, dim=0, pad_len=atom_pad_len)
                padded_tensor = pad_dim(padded_tensor, dim=1, pad_len=atom_pad_len)
                collated[key].append(padded_tensor)

        ##### No padding #####
        for key in structure_keys:
            if key in data:
                collated[key].append(data[key])
            
    # Stack all lists into the final batch
    final_batch = {}
    for key, tensor_list in collated.items():
        if tensor_list: # Only stack if the key was present in at least one item
            final_batch[key] = torch.stack(tensor_list, dim=0)

    if prepare_g1_metadata or prepare_g1_blocks:
        from src.data.execution_features import prepare_execution_layouts

        return prepare_execution_layouts(final_batch, prepare_blocks=prepare_g1_blocks)
    return final_batch


class BWDBDataModule(LightningDataModule):
    def __init__(
        self,
        lmdb_dir: str,
        lmdb_suffix: str,
        featurizer: Featurizer,
        prior_sampler: PriorSampler,
        optimal_transport: bool = False,
        sample_limit_dict: Dict[str, int] = None,
        max_num_atoms: Optional[int] = None,
        batch_size: Dict[str, int] = {"train": 32, "val": 32, "predict": 32},
        num_workers: int = 4,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        compact_batch: bool = False,
        prepare_g1_metadata: bool = False,
        prepare_g1_blocks: bool = False,
        sampler: Optional[str] = None,
        dynamic: Optional[Dict] = None,
        predict_split: str = 'val',
    ):
        super().__init__()
        self.lmdb_dir = lmdb_dir
        self.lmdb_suffix = lmdb_suffix
        self.featurizer = featurizer
        self.prior_sampler = prior_sampler
        self.optimal_transport = optimal_transport
        self.sample_limit_dict = sample_limit_dict or {}
        self.max_num_atoms = max_num_atoms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.compact_batch = compact_batch
        self.prepare_g1_metadata = prepare_g1_metadata
        self.prepare_g1_blocks = prepare_g1_blocks
        self._requested_g1_metadata = prepare_g1_metadata
        self._requested_g1_blocks = prepare_g1_blocks
        self.sampler = sampler
        self.dynamic = dynamic
        self.predict_split = predict_split

        self.splits = list(key for key in self.batch_size.keys() if key != 'predict')
        self._datasets = {}

    def configure_model_execution(self, *, g1_enabled: bool, execution):
        """Resolve CPU layouts from the constructed/loaded model before setup.

        Preserve explicit preparation requests when G1 is enabled. A disabled
        branch never prepares G1 tensors, even if a profile requests packing.
        Runtime model options also enable checkpoint-driven inference layouts.
        """
        self.prepare_g1_blocks = g1_enabled and (
            self._requested_g1_blocks or execution.get("packed_block_attention", False)
        )
        self.prepare_g1_metadata = g1_enabled and (
            self._requested_g1_metadata or self.prepare_g1_blocks
            or execution.get("prepared_layouts", False)
        )

    def setup(self, stage: Optional[str] = None):
        """Instantiate datasets for each split."""

        # Setup for fit (train/val)
        if stage in ('fit', 'validate', None):
            for split in self.splits:
                # Skip if already set up
                if split in self._datasets:
                    continue

                lmdb_path = Path(self.lmdb_dir) / f"{self.lmdb_suffix}_{split}.lmdb"
                sample_limit = self.sample_limit_dict.get(split)

                rank_zero_info(f"INFO:: Setting up '{split}' dataset from: {lmdb_path}")
                self._datasets[split] = BWDBDataset(
                    lmdb_path=lmdb_path,
                    featurizer=self.featurizer,
                    compact_batch=self.compact_batch,
                    prior_sampler=self.prior_sampler,
                    sample_limit=sample_limit,
                    max_num_atoms=self.max_num_atoms,
                    optimal_transport=self.optimal_transport, # TODO: Determine whether to apply only for training set
                )
        
        # Setup for predict
        if stage in ('predict', None):
            if 'predict' not in self._datasets:
                lmdb_path = Path(self.lmdb_dir) / f"{self.lmdb_suffix}_{self.predict_split}.lmdb"
                sample_limit = self.sample_limit_dict.get('predict')

                rank_zero_info(f"INFO:: Setting up 'predict' dataset using split '{self.predict_split}' from: {lmdb_path}")
                self._datasets['predict'] = BWDBDataset(
                    lmdb_path=lmdb_path,
                    featurizer=self.featurizer,
                    compact_batch=self.compact_batch,
                    sample_limit=sample_limit,
                    max_num_atoms=self.max_num_atoms,
                )

    def _build_dynamic_batch_sampler(self, dataset: BWDBDataset, is_train: bool = False):
        # Determine scaling method     
        if self.dynamic.get("scaling", "linear") == "linear":
            max_batch_units = self.dynamic['max_num_atoms']
            sort_key = lambda i: (dataset.num_units[i])
        elif self.dynamic.get("scaling") == "square":
            max_batch_units = self.dynamic['max_num_atoms_square']
            sort_key = lambda i: (dataset.num_units[i] ** 2)
        elif self.dynamic.get("scaling") == "cubic":
            max_batch_units = self.dynamic['max_num_atoms_cubic']
            sort_key = lambda i: (dataset.num_units[i] ** 3)
        else:
            raise ValueError(f"Unsupported scaling method: {self.dynamic.get('scaling')}")
        
        max_batch_size = self.dynamic.get("max_batch_size", None)
        num_buckets = self.dynamic.get("num_buckets", 100)
        shuffle_batches = self.dynamic.get("shuffle_batches", True)

        # Set sampler
        return DynamicBatchSampler(
            dataset=dataset,
            sampler=None, # NOTE: Lightning provides the sampler internally
            max_batch_units=max_batch_units,
            max_batch_size=max_batch_size,
            shuffle=is_train,
            shuffle_batches=shuffle_batches,
            skip_too_big=self.dynamic.get("skip_too_big", False),
            sort_key=sort_key,
            num_buckets=num_buckets,
            sync_ddp=is_train,
        )

    def _build_dataloader(self, split: str = 'train') -> DataLoader:
        dataset = self._datasets[split]
        batch_size = self.batch_size.get(split, 32)
        is_train = split == 'train'
        sampler = None
        batch_sampler = None

        if self.sampler == 'overfit' and is_train:
            sampler = OverfitSampler(num_samples=10000, dataset_len=len(dataset))
        elif self.sampler == 'dynamic':
            batch_sampler = self._build_dynamic_batch_sampler(dataset, is_train=is_train)

        return DataLoader(
            dataset,
            batch_size=batch_size if batch_sampler is None else 1,
            sampler=sampler,
            batch_sampler=batch_sampler,
            collate_fn=partial(collate_fn, compact_batch=self.compact_batch, prepare_g1_metadata=self.prepare_g1_metadata, prepare_g1_blocks=self.prepare_g1_blocks),
            shuffle=is_train if sampler is None and batch_sampler is None else False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            **({"prefetch_factor": self.prefetch_factor} if self.num_workers > 0 else {}),
        )

    def train_dataloader(self) -> DataLoader:
        return self._build_dataloader('train')

    def val_dataloader(self) -> DataLoader:
        return self._build_dataloader('val')

    def predict_dataloader(self) -> DataLoader:
        return self._build_dataloader('predict')