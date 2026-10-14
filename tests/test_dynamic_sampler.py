"""Regression tests for the datamodule's oversized-sample policy."""

import pytest
from torch.utils.data import SequentialSampler

from src.data.bwdb_datamodule import BWDBDataModule


class AtomCounts:
    """A dataset index with no LMDB or structure featurization dependency."""

    num_units = [100, 200, 707, 708]

    def __len__(self):
        return len(self.num_units)


@pytest.mark.parametrize(
    "scaling,power,key",
    [
        ("linear", 1, "max_num_atoms"),
        ("square", 2, "max_num_atoms_square"),
        ("cubic", 3, "max_num_atoms_cubic"),
    ],
)
@pytest.mark.parametrize("skip_too_big", [True, False, None])
def test_datamodule_honors_oversized_sample_policy(scaling, power, key, skip_too_big):
    """Test actual batch membership, including an oversized singleton and defaults."""
    dataset = AtomCounts()
    budget = 707**power
    dynamic = {
        "scaling": scaling,
        key: budget,
        "max_batch_size": 16,
        "num_buckets": 1,
    }
    if skip_too_big is not None:
        dynamic["skip_too_big"] = skip_too_big
    datamodule = BWDBDataModule(
        lmdb_dir="unused",
        lmdb_suffix="bwdb",
        featurizer=None,
        prior_sampler=None,
        sampler="dynamic",
        dynamic=dynamic,
    )
    sampler = datamodule._build_dynamic_batch_sampler(dataset)
    # Lightning normally injects its sampler when preparing the DataLoader.
    sampler.sampler = SequentialSampler(dataset)
    batches = list(sampler)
    indices = [index for batch in batches for index in batch]

    if skip_too_big:
        assert indices == [0, 1, 2]
        assert all(sum(dataset.num_units[i] ** power for i in b) <= budget for b in batches)
    else:
        assert indices == [0, 1, 2, 3]
        assert [3] in batches
        assert dataset.num_units[3] ** power > budget
