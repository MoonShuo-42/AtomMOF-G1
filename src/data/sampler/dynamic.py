import random
from typing import Callable, List

import torch
from torch import distributed as dist
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import BatchSampler


class DynamicBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset,
        sampler: Sampler,
        max_batch_units: int,
        max_batch_size: int = None,
        shuffle: bool = False,
        shuffle_batches: bool = True,
        drop_last: bool = False,
        skip_too_big: bool = False,
        sort_key: Callable = None,
        num_buckets: int = 100,
        sync_ddp: bool = False
    ):
        """
        A simplified batch sampler that dynamically groups samples into batches.
        This implementation uses a sort-and-bucket strategy for efficiency and simplicity.
        """
        super().__init__(sampler, batch_size=1, drop_last=drop_last)

        if sort_key is None:
            raise ValueError("A sort_key function must be provided to determine the number of units per sample.")

        self.max_batch_units = max_batch_units
        self.max_batch_size = max_batch_size
        self.sort_key = sort_key
        self.shuffle = shuffle
        self.shuffle_batches = shuffle_batches
        self.skip_too_big = skip_too_big
        self.sync_ddp = sync_ddp

        self.distributed = isinstance(self.sampler, DistributedSampler)
        self.bucket_size = len(dataset) // num_buckets if num_buckets > 0 else len(dataset)

        self._epoch = 0
        self._batches: List[List[int]] = []

    def __len__(self):
        if not self._batches:
            self._build_batches()
        return len(self._batches)

    def __iter__(self):
        if not self._batches:
            self._build_batches()
        for batch in self._batches:
            yield batch

    def _build_batches(self):
        # Get all indices and their sizes
        indices = list(self.sampler)
        sizes = [self.sort_key(idx) for idx in indices]

        # Sort samples by size (for efficiency)
        pairs = sorted(zip(indices, sizes), key=lambda x: x[1])

        # Bucket and shuffle (optional)
        # To reintroduce randomness, shuffle samples within each bucket
        if self.shuffle:
            shuffled_pairs = []
            for i in range(0, len(pairs), self.bucket_size):
                bucket = pairs[i : i + self.bucket_size]
                random.shuffle(bucket)
                shuffled_pairs.extend(bucket)
            pairs = shuffled_pairs
        
        # Pack greedily into batches
        batches = []
        current_batch = []
        current_batch_units = 0

        for index, size in pairs:
            # If the flag is set, skip samples that are too large to fit in a batch
            if self.skip_too_big and size > self.max_batch_units:
                continue

            # A batch is considered full if adding the next sample would exceed the
            # unit limit OR if the batch has already reached the max sample size.
            is_full_by_units = current_batch and (current_batch_units + size > self.max_batch_units)
            is_full_by_size = self.max_batch_size is not None and len(current_batch) >= self.max_batch_size
            
            if is_full_by_units or is_full_by_size:
                batches.append(current_batch)
                current_batch = []
                current_batch_units = 0
            
            current_batch.append(index)
            current_batch_units += size
        
        # Add the last batch if not empty
        if current_batch and not self.drop_last:
            batches.append(current_batch)
        
        # Shuffle the order of batches
        if self.shuffle and self.shuffle_batches:
            random.shuffle(batches)
        
        # DDP synchronization
        if self.distributed and self.sync_ddp:
            num_batches = torch.tensor(len(batches), device='cuda')
            dist.all_reduce(num_batches, op=dist.ReduceOp.MIN)
            # Truncate to the minimum number of batches across all processes
            batches = batches[:num_batches.item()]
        
        self._batches = batches
        assert all(len(b) > 0 for b in self._batches), "Sampler produced an empty batch."

    def set_epoch(self, epoch):
        self._epoch = epoch
        if self.distributed:
            self.sampler.set_epoch(epoch)
        self._batches = []  # Clear cached batches for the new epoch