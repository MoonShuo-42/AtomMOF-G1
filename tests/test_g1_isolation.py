"""Optional G1 imports and data preparation follow the effective model."""
import inspect
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from test_g1_integration import feats, flow_args, inputs, loss, perturb
from test_execution import MATH
from test_execution_refinements import raw_structure

from src.data.bwdb_datamodule import BWDBDataModule
from src.data.feature.featurizer import Featurizer
from src.models.execution import configure_data_execution


def test_baseline_forward_backward_without_importable_g1():
    # A fresh interpreter matters: other integration tests already import G1.
    script = r"""
import importlib.abc
import importlib.util
import sys
from copy import deepcopy
from types import ModuleType
import torch

torch.set_num_threads(1)
if importlib.util.find_spec("fairchem") is None:
    mlip = ModuleType("src.models.energy.mlip")
    class UnavailableMLIP:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Optional MLIP is not used in this test")
    mlip.MLIPEnergy = UnavailableMLIP
    sys.modules[mlip.__name__] = mlip

class NoG1(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "src.models.g1" or fullname.startswith("src.models.g1."):
            raise ImportError("Baseline attempted to import optional G1: " + fullname)
sys.meta_path.insert(0, NoG1())
from src.models.simple.flow import FlowModule
from src.models.execution import configure_execution
from src.data.bwdb_datamodule import BWDBDataModule
"""
    script += "\n".join(inspect.getsource(f) for f in (flow_args, feats, inputs, perturb, loss))
    script += f"\noptions = {MATH!r}\n"
    script += r"""
torch.manual_seed(41)
reference = FlowModule(**flow_args())
rng = torch.get_rng_state().clone()
torch.manual_seed(41)
model = FlowModule(**flow_args(), g1={"enabled": False, "module": {"_target_": "bad"}})
torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
assert model.g1_bridges is None and not model.g1_enabled
assert reference.state_dict().keys() == model.state_dict().keys()
perturb(reference)
model.load_state_dict(reference.state_dict(), strict=True)
configure_execution(model, options)
a, b = reference(**inputs()), model(**inputs())
for key in a:
    torch.testing.assert_close(a[key], b[key], rtol=5e-5, atol=2e-6)
loss(a).backward()
loss(b).backward()
for (name, p), (_, q) in zip(reference.named_parameters(), model.named_parameters()):
    assert (p.grad is None) == (q.grad is None), name
    if p.grad is not None:
        torch.testing.assert_close(p.grad, q.grad, rtol=3e-4, atol=3e-6, msg=name)
assert not any(k == "src.models.g1" or k.startswith("src.models.g1.") for k in sys.modules)
assert "src.data.execution_features" not in sys.modules
"""
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root))
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=root, env=env,
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("requested", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_data_layouts_follow_model_and_ablation_without_changing_samples(requested, packed):
    dm = BWDBDataModule(
        lmdb_dir="unused", lmdb_suffix="bwdb", featurizer=Featurizer(),
        prior_sampler=None, batch_size={"train": 2}, num_workers=0,
        compact_batch=True, prepare_g1_metadata=requested,
        prepare_g1_blocks=requested, pin_memory=False,
    )
    dm._datasets["train"] = [
        dm.featurizer.process(raw_structure(n), include_token_maps=False) for n in (3, 5)
    ]
    model = SimpleNamespace(
        g1_enabled=False,
        execution_options={"prepared_layouts": True, "packed_block_attention": packed},
    )
    baseline = None
    # Exercise both directions so flags cannot remain stale after an ablation.
    for enabled in (False, True, False, True):
        model.g1_enabled = enabled
        configure_data_execution(dm, model)
        assert dm.prepare_g1_metadata is enabled
        assert dm.prepare_g1_blocks is (enabled and (requested or packed))
        torch.manual_seed(7)  # keep DataLoader's shuffling fixed
        batch = next(iter(dm.train_dataloader()))
        if not enabled:
            assert not any(k.startswith("g1_") for k in batch)
            baseline = batch
        else:
            assert "g1_pair_indices" in batch
            assert ("g1_block_layout" in batch) is (requested or packed)
            for key in baseline:
                torch.testing.assert_close(batch[key], baseline[key], rtol=0, atol=0)


def test_optional_data_interface_and_explicit_layout_requests():
    model = SimpleNamespace(g1_enabled=True, execution_options={})
    configure_data_execution(object(), model)  # custom datamodule has no layouts
    dm = BWDBDataModule(
        "unused", "bwdb", Featurizer(), None, prepare_g1_blocks=True,
    )
    configure_data_execution(dm, model)
    assert dm.prepare_g1_metadata and dm.prepare_g1_blocks
    model.g1_enabled = False
    configure_data_execution(dm, model)
    assert not dm.prepare_g1_metadata and not dm.prepare_g1_blocks


@pytest.mark.parametrize("enabled", [False, True])
def test_training_entry_binds_data_before_fit(tmp_path, monkeypatch, enabled):
    import src.train as entry
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from test_g1_integration import model_args
    from src.models.simplemof_module import SimpleMOFModule
    from src.models.execution import configure_execution

    model = SimpleMOFModule(**model_args(enabled))
    configure_execution(model.structure_module.flow_model, MATH)
    data = BWDBDataModule(
        "unused", "bwdb", Featurizer(), None, prepare_g1_metadata=not enabled,
        prepare_g1_blocks=not enabled,
    )
    seen = []

    class Trainer:
        callback_metrics = {}

        def fit(self, **kwargs):
            assert kwargs["model"] is model and kwargs["datamodule"] is data
            assert data.prepare_g1_metadata is enabled
            assert data.prepare_g1_blocks is enabled
            seen.append(True)

    objects = {"test.Model": model, "test.Data": data, "test.Trainer": Trainer()}

    def instantiate_entry(config, **kwargs):
        if config.get("_target_") in objects:
            return objects[config._target_]
        return instantiate(config, **kwargs)

    cfg = OmegaConf.create({
        "model": {"_target_": "test.Model"}, "data": {"_target_": "test.Data"},
        "trainer": {"_target_": "test.Trainer"}, "train": True, "test": False,
        "paths": {"output_dir": str(tmp_path)},
    })
    monkeypatch.setattr(entry.hydra.utils, "instantiate", instantiate_entry)
    entry.train(cfg)
    assert seen == [True]
