import math
import random
from torch.utils.data.sampler import Sampler


class OverfitSampler(Sampler):
    def __init__(self, num_samples: int, dataset_len: int):
        self.num_samples = num_samples
        self.dataset_len = dataset_len

    def __iter__(self):
        idx_list = list(range(self.dataset_len))
        num_repeats = math.ceil(self.num_samples / self.dataset_len)
        full_list = (idx_list * num_repeats)[:self.num_samples]
        return iter(random.sample(full_list, len(full_list)))

    def __len__(self):
        return self.num_samples