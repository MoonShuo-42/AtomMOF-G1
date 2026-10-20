"""Data collation, augmentation RNG and loss/logging equivalence."""
from copy import deepcopy
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

# Retain the suite's isolation of the unused optional MLIP dependency.
from test_g1_integration import feats, model_args
from src.data.bwdb_datamodule import BWDBDataModule, collate_fn
from src.data.feature.featurizer import Featurizer
from src.models.simplemof_module import SimpleMOFModule


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _sample(n, *, dtype=torch.float32):
    return {
        "ref_element": torch.full((n,), 6, dtype=torch.long),
        "ref_charge": torch.arange(n, dtype=dtype),
        "ref_chirality": torch.zeros(n, dtype=torch.long),
        "block_index": torch.arange(n) // 2,
        "block_type": torch.ones(n, dtype=torch.long),
        # Non-contiguous input must work without changing values.
        "ref_pos": torch.arange(n * 3, dtype=dtype).reshape(3, n).T,
        "cart_coords": torch.arange(n * 3, dtype=dtype).reshape(n, 3),
        "frac_coords": torch.ones(n, 3, dtype=dtype),
        "cart_coords_0": torch.zeros(n, 3, dtype=dtype),
        "atom_to_token": torch.eye(n, dtype=dtype),
        "token_to_rep_atom": torch.eye(n, dtype=dtype),
        "lattice": torch.tensor([10, 11, 12, 90, 90, 90], dtype=dtype),
        "lattice_0": torch.tensor([11, 12, 13, 90, 90, 90], dtype=dtype),
        "cell": torch.eye(3, dtype=dtype),
        "bond_indices": (
            torch.tensor([[0, 1], [1, 0]]) if n > 1 else torch.empty(2, 0, dtype=torch.long)
        ),
        "bond_types": torch.tensor([1, 2]) if n > 1 else torch.empty(0, dtype=torch.long),
    }


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("sizes", [(1,), (0, 0), (0, 1, 5), (3, 5, 4)])
@pytest.mark.parametrize("mixed_dtype", [False, True])
def test_preallocated_collation_matches_all_fields_and_rng(compact, sizes, mixed_dtype):
    samples = [
        _sample(n, dtype=torch.float64 if mixed_dtype and i % 2 else torch.float32)
        for i, n in enumerate(sizes)
    ]
    # Legacy two-pass F.pad cannot expand a (0, 0) token map. Empty samples
    # may omit optional token maps; keep this comparison in its valid domain.
    if not compact and max(sizes) > 0:
        for sample in samples:
            if sample["ref_element"].numel() == 0:
                del sample["atom_to_token"]
                del sample["token_to_rep_atom"]
    rng = torch.get_rng_state().clone()
    reference = collate_fn(samples, compact_batch=compact)
    actual = collate_fn(samples, compact_batch=compact, preallocate_collate=True)
    assert list(actual) == list(reference)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert collate_fn([], preallocate_collate=True) == {}


@pytest.mark.parametrize("compact", [False, True])
def test_missing_optional_fields_keep_present_sample_order(compact):
    samples = [_sample(n) for n in (2, 5, 3)]
    del samples[0]["frac_coords"]
    del samples[1]["cart_coords_0"]
    del samples[2]["lattice_0"]
    del samples[1]["atom_to_token"]
    for sample in samples:
        del sample["ref_chirality"]
    reference = collate_fn(samples, compact_batch=compact)
    actual = collate_fn(samples, compact_batch=compact, preallocate_collate=True)
    assert list(actual) == list(reference)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert actual["frac_coords"].shape == (2, 5, 3)
    assert "ref_chirality" not in actual


@pytest.mark.parametrize("prepare_blocks", [False, True])
def test_preallocated_collation_preserves_g1_worker_metadata(prepare_blocks):
    samples = [_sample(n) for n in (2, 5, 3)]
    options = dict(compact_batch=True, prepare_g1_metadata=True, prepare_g1_blocks=prepare_blocks)
    reference = collate_fn(samples, **options)
    actual = collate_fn(samples, preallocate_collate=True, **options)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_preallocated_collation_keeps_input_autograd():
    samples = [_sample(n) for n in (3, 5)]
    for sample in samples:
        sample["ref_pos"].requires_grad_()
        sample["atom_to_token"].requires_grad_()
    other = deepcopy(samples)
    reference = collate_fn(samples)
    actual = collate_fn(other, preallocate_collate=True)
    for data in (reference, actual):
        (data["ref_pos"].square().sum() + data["atom_to_token"].sum()).backward()
    for old, new in zip(samples, other):
        for key in ("ref_pos", "atom_to_token"):
            torch.testing.assert_close(old[key].grad, new[key].grad, rtol=0, atol=0)


@pytest.mark.parametrize("workers", [0, 2])
def test_real_dataloader_preserves_batches_and_optional_fields(workers):
    data = [_sample(n) for n in (3, 5, 2, 4)]
    # Spawn is exercised on Windows and Linux; no GPU is needed for pinned input.
    options = dict(batch_size=2, num_workers=workers, pin_memory=False)
    if workers:
        options.update(multiprocessing_context="spawn", persistent_workers=True)
    old = DataLoader(data, collate_fn=partial(collate_fn, compact_batch=True), **options)
    new = DataLoader(
        data,
        collate_fn=partial(collate_fn, compact_batch=True, preallocate_collate=True),
        **options,
    )
    for first, second in zip(old, new):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    dm = BWDBDataModule(
        "unused",
        "bwdb",
        Featurizer(),
        None,
        num_workers=0,
        preallocate_collate=True,
        compact_batch=True,
        batch_size={"train": 2},
    )
    dm._datasets["train"] = data
    assert dm.train_dataloader().collate_fn.keywords["preallocate_collate"] is True


def _structure_with_empty_blocks():
    blocks = []
    for n in (0, 2, 0, 1, 4, 0):
        blocks.append(
            SimpleNamespace(
                atomic_numbers=np.full(n, 6),
                cart_coords=np.arange(n * 3, dtype=np.float32).reshape(n, 3) / 7,
                formal_charges=np.zeros(n),
                chirality_tags=np.zeros(n, dtype=int),
                block_type="LINKER",
                bond_indices=np.empty((2, 0), dtype=int),
                bond_types=np.empty(0, dtype=int),
            )
        )
    coords = np.concatenate([b.cart_coords for b in blocks])
    return SimpleNamespace(
        blocks=blocks,
        cart_coords=coords,
        frac_coords=coords / 10,
        lattice=np.array([10, 11, 12, 90, 90, 90]),
        cell=np.eye(3),
    )


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_contiguous_block_augmentation_preserves_features_and_random_stream(seed):
    data = _structure_with_empty_blocks()
    featurizer = Featurizer()
    method = featurizer._apply_augmentations
    # Force the original boolean-mask route on identical raw structures.
    with patch.object(
        featurizer, "_apply_augmentations", side_effect=lambda x, b, m, **kw: method(x, b, m)
    ):
        torch.manual_seed(seed)
        expected = featurizer.process(data)
        expected_next = torch.randn(31)
    torch.manual_seed(seed)
    actual = featurizer.process(data)
    actual_next = torch.randn(31)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_next, expected_next, rtol=0, atol=0)


class _LossFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.3, 0.8, 1.2, 0.4]))

    def forward(self, **kwargs):
        return {"times": torch.tensor([0.0, 0.25, 0.75, 1.0])}

    def compute_loss(self, *args, **kwargs):
        return {
            "coord_loss": self.weight.square(),
            "dist_loss": torch.zeros_like(self.weight),
            "length_loss": self.weight * 0.7,
            "angle_loss": self.weight * 1.3,
        }


@pytest.mark.parametrize("prefix", ["train", "val"])
def test_loss_means_are_reused_without_changing_gradients_or_logging(prefix):
    model = SimpleMOFModule(**model_args(False))
    model.structure_module = _LossFixture()
    reference = deepcopy(model.structure_module)
    values = reference.compute_loss()
    args = model.training_args
    expected = (
        args["coord_loss_weight"] * values["coord_loss"].mean()
        + args["lengths_loss_weight"] * values["length_loss"].mean()
        + args["angle_loss_weight"] * values["angle_loss"].mean()
        + args["dist_loss_weight"] * values["dist_loss"].mean()
    )
    calls = []
    with patch.object(
        model, "log", side_effect=lambda name, value, **kw: calls.append((name, value, kw))
    ):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            actual, losses, out = model._compute_and_log_loss(
                {"ref_element": torch.zeros(4, 2)}, prefix
            )
    assert sum(event.count for event in profile.key_averages() if event.key == "aten::mean") == 4
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(
        model.structure_module.weight.grad, reference.weight.grad, rtol=0, atol=0
    )
    assert [call[0] for call in calls] == [f"{prefix}/{key}" for key in values] + [
        f"{prefix}/total_loss"
    ]
    for (name, value, options), old in zip(calls, [*values.values(), expected]):
        torch.testing.assert_close(value, old.detach().mean(), rtol=0, atol=0)
        assert options == {"batch_size": 4, "sync_dist": prefix == "val"}


def test_training_step_retains_all_log_keys_empty_bins_and_step_options():
    model = SimpleMOFModule(**model_args(False))
    model.structure_module = _LossFixture()
    model.structure_module.flow_model = SimpleNamespace(
        execution_options={"joint_time_metrics": True}
    )
    calls = []
    with patch.object(
        model, "log", side_effect=lambda name, value, **kw: calls.append((name, value, kw))
    ):
        loss = model.training_step({"ref_element": torch.zeros(4, 2)}, 0)
    assert loss.requires_grad and len(calls) == 23
    keys = [key for key, _, _ in calls]
    assert len(set(keys)) == 23
    assert keys[5] == "train/batch_size" and keys[-1] == "train/samples_per_second"
    for key, value, options in calls[6:-1]:
        assert options == {"batch_size": 4}
        if "t=[0.25,0.50)" in key:
            assert torch.isnan(value)
        assert not value.requires_grad


def test_datamodule_preserves_old_positional_argument_contract():
    featurizer = Featurizer()
    limits = {"train": 11}
    sizes = {"train": 2, "predict": 3}
    dynamic = {"scaling": "square", "max_num_atoms_square": 500000}
    # All constructor arguments through predict_split predate preallocation.
    data = BWDBDataModule(
        "unused",
        "bwdb",
        featurizer,
        None,
        True,
        limits,
        300,
        sizes,
        2,
        3,
        False,
        True,
        True,
        True,
        False,
        "dynamic",
        dynamic,
        "test",
    )
    assert data.featurizer is featurizer
    assert data.optimal_transport is True and data.sample_limit_dict is limits
    assert data.max_num_atoms == 300 and data.batch_size is sizes
    assert data.num_workers == 2 and data.prefetch_factor == 3
    assert data.pin_memory is False and data.persistent_workers is True
    assert data.compact_batch is True and data.prepare_g1_metadata is True
    assert data.prepare_g1_blocks is False and data.sampler == "dynamic"
    assert data.dynamic is dynamic and data.predict_split == "test"
    assert data.preallocate_collate is False
