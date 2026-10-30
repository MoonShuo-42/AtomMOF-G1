"""Multiple G1-New sites: shared work/gradients, independent weights and loading."""

from copy import deepcopy

import pytest
import torch
from omegaconf import OmegaConf
from test_g1new import (
    FlowModule,
    SimpleMOFModule,
    branch_inputs,
    flow_args,
    inputs,
    model_args,
    new_config,
    perturb,
    save_checkpoint,
)

from src.models.g1new.bridge import G1NewBridge
from src.models.g1new.checkpoint import load_g1new_pretrained


@pytest.fixture(autouse=True, scope="module")
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def config(positions=(6, 12, 18, 24), checkpoint=True):
    cfg = new_config(checkpoint=checkpoint, chunk=7)
    cfg["insert_after"] = positions if type(positions) is int else list(positions)
    return cfg


def adapter(positions=(6, 12, 18, 24), checkpoint=True):
    cfg = config(positions, checkpoint)
    cfg.pop("enabled")
    return G1NewBridge(h_dim=16, depth=24, **cfg)


@pytest.mark.parametrize("positions", [[6, 12, 18, 24], [8, 16], [24], 12])
def test_positions_and_parameter_sharing(positions):
    bridge = adapter(positions)
    expected = (positions,) if type(positions) is int else tuple(positions)
    assert bridge.insert_after == expected
    first = bridge.first_bridge
    encoder = first.condition_encoder
    for site in bridge.bridges.values():
        assert site.condition_encoder is encoder
        assert site.geometry is first.geometry
        assert site.time_embedding is first.time_embedding
    rest = [
        p
        for site in bridge.bridges.values()
        for name, p in site.named_parameters()
        if not name.startswith("condition_encoder.")
    ]
    assert len({id(p) for p in rest}) == len(rest)
    assert len({id(p) for p in bridge.parameters()}) == len(list(bridge.parameters()))
    if len(expected) > 1:
        # Site parameters are independently initialized, not cloned/weight-tied.
        a, b = list(bridge.bridges.values())[:2]
        assert not torch.equal(a.readback.messages.first.weight, b.readback.messages.first.weight)


@pytest.mark.parametrize(
    "positions", [[], [6, 6], [12, 6], [0], [25], [6, True], True, "12", [6, 7.5]]
)
def test_invalid_positions(positions):
    with pytest.raises(ValueError, match="insert_after"):
        G1NewBridge(h_dim=16, depth=24, insert_after=positions)


@pytest.mark.parametrize("checkpoint", [False, True])
def test_shared_forward_and_all_gradients_match_recomputing_each_site(checkpoint):
    torch.manual_seed(37)
    shared = adapter(checkpoint=checkpoint).double().train()
    perturb(shared)
    reference = deepcopy(shared)
    H, t, feats, x, lattice = branch_inputs(torch.float64)
    left = [value.clone().requires_grad_() for value in (H, t, x, lattice)]
    right = [value.clone().requires_grad_() for value in (H, t, x, lattice)]
    context = shared.prepare(feats, left[2], left[3])
    actual, expected = left[0], right[0]
    for position in shared.insert_after:
        actual = shared(position, actual, left[1], context)
        site = reference.bridges[str(position)]
        expected = site(
            expected,
            right[1],
            feats,
            site.geometry(right[2], right[3], feats),
            site.encode_condition(feats),
        )
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)
    weight = torch.randn_like(actual)
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()
    for a, b in zip(left, right):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-8, atol=1e-9)
    for (name, a), (_, b) in zip(shared.named_parameters(), reference.named_parameters()):
        assert a.grad is not None and torch.isfinite(a.grad).all(), name
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-8, atol=1e-9, msg=name)


def test_real_optimized_flow_insertion_reuse_sampling_cache_and_baseline_ablation():
    options = OmegaConf.to_container(
        OmegaConf.load("configs/execution/optimized.yaml").model.flow_model_args.execution
    )
    model = FlowModule(**flow_args(), g1new=config(), execution=options).eval()
    perturb(model)
    bridge = model.g1new_bridge
    counters = {key: 0 for key in ("encoder", "geometry", "time")}
    events, handles = [], []

    def count(key):
        def hook(*args):
            counters[key] += 1

        return hook

    for key, module in (
        ("encoder", bridge.first_bridge.condition_encoder),
        ("geometry", bridge.first_bridge.geometry),
        ("time", bridge.first_bridge.time_embedding),
    ):
        handles.append(module.register_forward_hook(count(key)))
    for position, site in bridge.bridges.items():
        handles.append(
            site.register_forward_hook(lambda *a, p=int(position): events.append(("new", p)))
        )
    for position, block in enumerate(model.token_transformer.token_transformer.layers, 1):
        handles.append(block.register_forward_hook(lambda *a, p=position: events.append(p)))
    data = inputs(2)
    with torch.no_grad():
        uncached = model(**data)
        expected = []
        for position in range(1, 25):
            expected.append(position)
            if position in bridge.insert_after:
                expected.append(("new", position))
        assert events == expected
        assert counters == dict(encoder=1, geometry=1, time=1)
        static = model.prepare_sampling_conditions(data["feats"])
        cached = model(**data, static_condition=static)
        for key in uncached:
            torch.testing.assert_close(cached[key], uncached[key], rtol=0, atol=0)
        data["r_noisy"] = data["r_noisy"] + 0.02
        model(**data, static_condition=static)
        assert counters == dict(encoder=2, geometry=3, time=3)
        model.set_g1new_enabled(False)
        baseline = FlowModule(**flow_args(), execution=options).eval()
        baseline.load_state_dict(
            {k: v for k, v in model.state_dict().items() if not k.startswith("g1new_bridge.")},
            strict=True,
        )
        for key, value in baseline(**data).items():
            torch.testing.assert_close(model(**data)[key], value, rtol=0, atol=0)
    for handle in handles:
        handle.remove()


@pytest.mark.parametrize("legacy_g1", [False, True])
def test_transfer_to_four_sites_and_strict_roundtrip(tmp_path, legacy_g1):
    source = SimpleMOFModule(**model_args(legacy_g1))
    perturb(source)
    path = tmp_path / "source.ckpt"
    save_checkpoint(source, path)
    args = model_args(False)
    args["flow_model_args"]["g1new"] = config()
    target = SimpleMOFModule(**args)
    initialized = {
        k: v.clone()
        for k, v in target.state_dict().items()
        if "g1new_bridge" in k and "condition_encoder" not in k
    }
    load_g1new_pretrained(target, path)
    for key, value in initialized.items():
        assert torch.equal(value, target.state_dict()[key])
    if legacy_g1:
        encoder = source.structure_module.flow_model.g1_bridges.bridges["6"].condition_encoder
        for site in target.structure_module.flow_model.g1new_bridge.bridges.values():
            for key, value in encoder.state_dict().items():
                assert torch.equal(value, site.condition_encoder.state_dict()[key])
    path = tmp_path / "multi.ckpt"
    save_checkpoint(target, path)
    loaded = SimpleMOFModule.load_from_checkpoint(path, strict=True)
    for key, value in target.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])
    bad = deepcopy(target.state_dict())
    alias = next(key for key in bad if "bridges.12.condition_encoder." in key)
    bad[alias] = bad[alias] + 1
    with pytest.raises(RuntimeError, match="aliases disagree"):
        loaded.load_state_dict(bad, strict=True)


def test_original_single_checkpoint_names_load_strictly_without_expanding(tmp_path):
    args = model_args(False)
    args["flow_model_args"]["g1new"] = config(12)
    source = SimpleMOFModule(**args).eval()
    perturb(source)
    path = tmp_path / "old_single.ckpt"
    save_checkpoint(source, path)
    ckpt = torch.load(path, weights_only=False)
    ckpt["state_dict"] = {
        key.replace("g1new_bridge.bridges.12.", "g1new_bridge.module."): value
        for key, value in ckpt["state_dict"].items()
    }
    torch.save(ckpt, path)
    loaded = SimpleMOFModule.load_from_checkpoint(path, strict=True).eval()
    for key, value in source.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])
    with torch.no_grad():
        for key, value in source.structure_module.flow_model(**inputs()).items():
            torch.testing.assert_close(
                loaded.structure_module.flow_model(**inputs())[key], value, rtol=0, atol=0
            )
    args["flow_model_args"]["g1new"] = config()
    with pytest.raises(RuntimeError, match="original single-site config"):
        SimpleMOFModule(**args).load_state_dict(ckpt["state_dict"], strict=True)
