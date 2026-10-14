"""Validation-hook regressions using real CPU tensors and Gloo collectives.

The generative network is replaced by a small module: these tests exercise the
validation hooks without importing the optional CUDA/MLIP inference stack.
"""

import importlib.util
import math
import sys
from concurrent.futures import Future, TimeoutError
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


class ToyFlow(torch.nn.Module):
    """Provide a parameter/device without constructing the generative network."""

    def __init__(self, **kwargs):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))


def load_model_module():
    """Import the real validation hooks with only their network dependency replaced."""
    flow_module = ModuleType("src.models.simple.flow")
    flow_module.AtomFlowMatching = ToyFlow
    source = Path(__file__).resolve().parents[1] / "src/models/simplemof_module.py"
    spec = importlib.util.spec_from_file_location("_validation_module_under_test", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    module_name = "src.models.simple.flow"
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = flow_module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
    return module


model_module = load_model_module()


def make_model(num_samples=1):
    """Build a module with production validation configuration and a toy network."""
    return model_module.SimpleMOFModule(
        atom_s=4,
        atom_z=4,
        token_s=4,
        token_z=4,
        atom_feature_dim=10,
        flow_model_args={},
        training_args={},
        flow_process_args={},
        validation_args={
            "flow_samples": num_samples,
            "sample_every_n_epochs": 10,
            "sampling_steps": 10,
            "sampling_center_coords": True,
            "num_workers": 1,
            "timeout": 30,
            "match_mode": "framework",
            "structure_matcher_args": {"stol": 0.5, "ltol": 0.3, "angle_tol": 10},
        },
    )


def cached_batch(scores):
    """Encode one or more candidate scores per one-atom structure."""
    batch_size = len(scores)
    num_samples = len(scores[0])
    coords = torch.tensor(scores, dtype=torch.float32).reshape(-1, 1, 1).repeat(1, 1, 3)
    lattices = torch.tensor([[10.0, 10.0, 10.0, 90.0, 90.0, 90.0]])
    return {
        "sampled_coords": coords,
        "sampled_lattices": lattices.repeat(batch_size * num_samples, 1),
        "true_coords": torch.zeros(batch_size, 1, 3),
        "true_lattices": lattices.repeat(batch_size, 1),
        "atom_types": torch.full((batch_size, 1), 6, dtype=torch.long),
        "atom_mask": torch.ones(batch_size, 1, dtype=torch.bool),
        "block_type": torch.zeros(batch_size, 1, dtype=torch.long),
    }


def score_task(task):
    """Return known candidate results, including unmatched and failed tasks."""
    scores = task[0][:, 0, 0].tolist()
    if scores[0] == -2:
        raise TimeoutError("matching timeout")
    if scores[0] == -3:
        raise ValueError("invalid structure")
    return [score if score >= 0 else None for score in scores]


class InlinePool:
    """Run matching fixtures synchronously while preserving Future error behavior."""

    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def schedule(self, function, args, timeout=None):
        future = Future()
        try:
            future.set_result(function(*args))
        except Exception as error:
            future.set_exception(error)
        return future


def capture_logs(model):
    """Capture public metric values emitted by the hook."""
    logged = {}
    model.log = lambda name, value, **kwargs: logged.__setitem__(name, float(value))
    return logged


def test_validation_cache_is_detached_and_on_cpu():
    """The sampling step must not retain tensors or graphs on its input device."""
    model = make_model()
    model._trainer = SimpleNamespace(current_epoch=9)
    output = cached_batch([[0.0]])
    coords = output["sampled_coords"].requires_grad_()
    lattices = output["sampled_lattices"].requires_grad_()
    model._compute_and_log_loss = lambda *args, **kwargs: None
    model.forward = lambda *args, **kwargs: {
        "sampled_atom_coords": coords,
        "sampled_lattice": lattices,
    }
    batch = {
        "cart_coords": output["true_coords"],
        "lattice": output["true_lattices"],
        "ref_element": output["atom_types"],
        "atom_pad_mask": output["atom_mask"],
        "block_type": output["block_type"],
    }
    model.validation_step(batch, 0)
    assert len(model.validation_step_outputs) == 1
    for tensor in model.validation_step_outputs[0].values():
        assert tensor.device.type == "cpu"
        assert not tensor.requires_grad
        assert tensor.grad_fn is None


def test_best_candidate_metrics_and_cache_reset(monkeypatch):
    """MR counts structures and RMSD averages only the best matched candidates."""
    monkeypatch.setattr(model_module, "ProcessPool", InlinePool)
    monkeypatch.setattr(model_module, "find_best_match_rmsd", score_task)
    model = make_model(num_samples=2)
    logged = capture_logs(model)
    model.validation_step_outputs.append(cached_batch([[0.5, 0.1], [-1, -1], [0.4, 0.6]]))
    model.on_validation_epoch_end()
    assert logged["val/match_rate"] == pytest.approx(2 / 3)
    assert logged["val/avg_rmsd"] == pytest.approx(0.25)
    assert model.validation_step_outputs == []
    logged.clear()
    model.on_validation_epoch_end()
    assert logged == {}


def test_failed_matching_tasks_count_as_unmatched(monkeypatch):
    """Timeouts and invalid structures stay in the match-rate denominator."""
    monkeypatch.setattr(model_module, "ProcessPool", InlinePool)
    monkeypatch.setattr(model_module, "find_best_match_rmsd", score_task)
    model = make_model()
    logged = capture_logs(model)
    model.validation_step_outputs.append(cached_batch([[0.0], [-2], [-3]]))
    model.on_validation_epoch_end()
    assert logged["val/match_rate"] == pytest.approx(1 / 3)
    assert logged["val/avg_rmsd"] == 0.0
    assert model.validation_step_outputs == []


def test_cache_cleared_after_unexpected_error():
    """Finally-based cleanup also applies to malformed cached outputs."""
    model = make_model()
    model.validation_step_outputs.append({})
    with pytest.raises(KeyError):
        model.on_validation_epoch_end()
    assert model.validation_step_outputs == []


def test_real_structure_matching_process_pool():
    """A real spawned matcher recognizes an identical one-atom crystal."""
    model = make_model()
    logged = capture_logs(model)
    model.validation_step_outputs.append(cached_batch([[0.0]]))
    model.on_validation_epoch_end()
    assert logged["val/match_rate"] == 1.0
    assert logged["val/avg_rmsd"] == pytest.approx(0.0, abs=1e-7)
    assert model.validation_step_outputs == []


def distributed_worker(rank, world_size, init_method):
    """Exercise uneven and empty ranks through actual distributed reductions."""
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=init_method,
        timeout=timedelta(seconds=60),
    )
    try:
        cases = [
            ([[[0.1]], [[0.3], [-1], [-1]]], 0.5, 0.2),
            ([[], [[0.4], [-1]]], 0.5, 0.4),
            ([[[-1]], [[-1], [-1]]], 0.0, float("nan")),
            ([[], []], None, None),
        ]
        model = make_model()
        with patch.object(model_module, "ProcessPool", InlinePool), patch.object(
            model_module, "find_best_match_rmsd", score_task
        ):
            for per_rank, expected_mr, expected_rmsd in cases:
                logged = capture_logs(model)
                if per_rank[rank]:
                    model.validation_step_outputs.append(cached_batch(per_rank[rank]))
                model.on_validation_epoch_end()
                assert model.validation_step_outputs == []
                if expected_mr is None:
                    assert logged == {}
                else:
                    assert logged["val/match_rate"] == pytest.approx(expected_mr)
                    if math.isnan(expected_rmsd):
                        assert math.isnan(logged["val/avg_rmsd"])
                    else:
                        assert logged["val/avg_rmsd"] == pytest.approx(expected_rmsd)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo backend unavailable")
def test_distributed_validation_with_uneven_and_empty_ranks(tmp_path):
    """Use two real processes, including a rank with no sampling outputs."""
    rendezvous = (tmp_path / "gloo-init").as_uri()
    mp.spawn(distributed_worker, args=(2, rendezvous), nprocs=2, join=True)
