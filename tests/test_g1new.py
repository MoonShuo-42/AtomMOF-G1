"""Exact geometry, v0 equations, chunk gradients and real Flow integration."""

from copy import deepcopy
from itertools import product
from pathlib import Path

import lightning as L
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

# Reuse actual Flow test fixtures; only the optional, unused MLIP backend is stubbed.
from test_g1_integration import (
    FlowModule,
    SimpleMOFModule,
    flow_args,
    inputs,
    loss,
    model_args,
    perturb,
)

from src.models.g1new import G1NewModule
from src.models.g1new.checkpoint import load_g1new_pretrained
from src.models.g1new.conditioning import ConditionEncoder
from src.models.g1new.geometry import LocalGeometry, SearchBudgetExceeded
from src.models.g1new.geometry.lattice import lattice_parameters_to_cell


@pytest.fixture(autouse=True, scope="module")
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def new_config(checkpoint=True, chunk=3):
    return dict(
        enabled=True,
        insert_after=12,
        module={
            "_target_": "src.models.g1new.G1NewModule",
            "h_dim": 16,
            "width": 8,
            "hidden_dim": 16,
            "edge_chunk_size": chunk,
            "activation_checkpointing": checkpoint,
            "condition_encoder": {
                "_target_": "src.models.g1new.conditioning.ConditionEncoder",
                "atom_s": 16,
                "atom_z": 8,
                "hidden_dim": 8,
                "pair_dim": 4,
                "output_dim": 8,
                "depth": 1,
                "heads": 4,
            },
            "geometry": {
                "_target_": "src.models.g1new.geometry.LocalGeometry",
                "pair_chunk_size": 5,
                "image_chunk_size": 7,
            },
        },
    )


def small_module(checkpoint=False, chunk=3):
    return instantiate(new_config(checkpoint, chunk)["module"])


def branch_inputs(dtype=torch.float32):
    data = inputs()
    feats = {k: v.to(dtype) if v.is_floating_point() else v for k, v in data["feats"].items()}
    H = torch.randn(2, 4, 16, dtype=dtype)
    return H, data["times"].to(dtype), feats, data["r_noisy"].to(dtype), data["l_noisy"].to(dtype)


def run_branch(model, H, t, feats, x, lattice):
    return model(H, t, feats, model.geometry(x, lattice, feats), model.encode_condition(feats))


def test_independent_condition_encoder_matches_g1_initialization_outputs_and_gradients():
    from src.models.execution import configure_execution
    from src.models.g1.conditioning import ConditionEncoder as LegacyEncoder

    kwargs = {
        k: v for k, v in new_config()["module"]["condition_encoder"].items() if k != "_target_"
    }
    torch.manual_seed(73)
    old = LegacyEncoder(**kwargs)
    torch.manual_seed(73)
    new = ConditionEncoder(**kwargs)
    assert type(new) is not type(old)
    assert old.state_dict().keys() == new.state_dict().keys()
    for key, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[key]), key
    perturb(old)
    new.load_state_dict(old.state_dict(), strict=True)
    options = dict(prepared_layouts=True, fused_pair_features=True, fused_qkvg=True)
    configure_execution(old, options)
    configure_execution(new, options)
    a, b = deepcopy(inputs()["feats"]), deepcopy(inputs()["feats"])
    a["ref_pos"].requires_grad_()
    b["ref_pos"].requires_grad_()
    expected, actual = old(a), new(b)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(a["ref_pos"].grad, b["ref_pos"].grad, rtol=0, atol=0)
    for (name, p), (_, q) in zip(old.named_parameters(), new.named_parameters()):
        assert p.grad is not None and q.grad is not None, name
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0, msg=name)


def test_g1new_real_forward_backward_without_importable_legacy_g1():
    import inspect
    import os
    import subprocess
    import sys

    from test_g1_integration import feats

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
class NoLegacyG1(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "src.models.g1" or fullname.startswith("src.models.g1."):
            raise ImportError("G1-New attempted to import legacy G1: " + fullname)
sys.meta_path.insert(0, NoLegacyG1())
from src.models.simple.flow import FlowModule
from src.models.execution import configure_execution
from src.data.execution_features import prepare_execution_layouts
"""
    script += "\n".join(
        inspect.getsource(f) for f in (flow_args, feats, inputs, perturb, loss, new_config)
    )
    script += r"""
config = new_config()
config['insert_after'] = [6, 12, 18, 24]
model = FlowModule(**flow_args(), g1new=config)
perturb(model)
configure_execution(model, dict(prepared_layouts=True, fused_pair_features=True))
data = inputs()
prepare_execution_layouts(data["feats"])
loss(model(**data)).backward()
assert model.g1_bridges is None and model.g1new_enabled
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
assert not any(k == "src.models.g1" or k.startswith("src.models.g1.") for k in sys.modules)
"""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=root,
        env=dict(os.environ, PYTHONPATH=str(root)),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def edge_keys(edges):
    return {
        (int(a), int(k), *map(int, n)): int(bond)
        for a, k, n, bond in zip(edges.atom, edges.env, edges.shift, edges.bond)
    }


def test_periodic_search_matches_bruteforce_and_retains_images():
    H, t, feats, x, lattice = branch_inputs(torch.float64)
    lattice[:, :3] = torch.tensor([0.55, 0.75, 1.1])
    lattice[:, 3:] = torch.tensor([1.25, 1.45, 1.1])
    feats["ref_element"][0, 1] = 1
    feats["bond_types"][0, 0, 1] = feats["bond_types"][0, 1, 0] = 1
    geometry = LocalGeometry(pair_chunk_size=3, image_chunk_size=5)
    edges = geometry(x, lattice, feats)
    cell, _ = lattice_parameters_to_cell(lattice)
    shifts = torch.tensor(list(product(range(-3, 4), repeat=3)), dtype=torch.float64)
    expected = {}
    for k, anchor in enumerate(edges.anchors.tolist()):
        b, a = divmod(anchor, x.shape[1])
        for i in range(x.shape[1]):
            if not feats["atom_pad_mask"][b, i]:
                continue
            distances = torch.linalg.vector_norm(x[b, i] + shifts @ cell[b] - x[b, a], dim=-1)
            nearest = int(distances.argmin())
            bond = int(feats["bond_types"][b, i, a])
            for j, n in enumerate(shifts.long().tolist()):
                if i == a and n == [0, 0, 0]:
                    continue
                if distances[j] < 0.8 or (bond > 0 and j == nearest):
                    expected[(b * x.shape[1] + i, k, *n)] = bond if j == nearest else 0
    assert edge_keys(edges) == expected
    assert len(expected) == edges.atom.numel()
    assert len(edges.anchors) == 4  # hydrogen and padding never create environments
    assert any(n != (0, 0, 0) for _, _, *n_list in expected for n in [tuple(n_list)])


def test_known_bond_outside_radius_and_hard_image_switch():
    feats = dict(
        atom_pad_mask=torch.ones(1, 2, dtype=torch.bool),
        ref_element=torch.tensor([[6, 1]]),
        block_index=torch.zeros(1, 2, dtype=torch.long),
        bond_types=torch.tensor([[[0, 1], [1, 0]]]),
    )
    lattice = torch.tensor(
        [[2.0, 2.0, 2.0, torch.pi / 2, torch.pi / 2, torch.pi / 2]], dtype=torch.float64
    )
    geo = LocalGeometry()
    directions = []
    for dx in (1 - 1e-6, 1 + 1e-6):
        x = torch.tensor([[[0.0, 0, 0], [dx, 0, 0]]], dtype=torch.float64)
        result = geo(x, lattice, feats)
        selected = result.bond > 0
        assert selected.sum() == 1
        assert result.omega[selected].item() == 1
        assert result.distance[selected].item() > geo.cutoff
        directions.append(result.direction[selected, 0].item())
    assert directions[0] > 0.99 and directions[1] < -0.99


def test_budget_failure_is_explicit_and_never_truncates():
    _, _, feats, x, lattice = branch_inputs(torch.float64)
    lattice[:, :3] = 0.01
    with pytest.raises(SearchBudgetExceeded, match="budget"):
        LocalGeometry(max_search_candidates=100)(x, lattice, feats)
    _, _, feats, x, lattice = branch_inputs()
    with pytest.raises(SearchBudgetExceeded, match="edges budget"):
        LocalGeometry(max_edges=1)(x, lattice, feats)


def test_selected_periodic_geometry_passes_finite_difference_gradcheck():
    feats = dict(
        atom_pad_mask=torch.ones(1, 2, dtype=torch.bool),
        ref_element=torch.tensor([[6, 8]]),
        block_index=torch.zeros(1, 2, dtype=torch.long),
        bond_types=torch.zeros(1, 2, 2, dtype=torch.long),
    )
    x = torch.tensor(
        [[[0.1, 0.2, 0.05], [0.35, 0.4, 0.12]]], dtype=torch.float64, requires_grad=True
    )
    lattice = torch.tensor(
        [[1.0, 1.2, 1.3, 1.4, 1.5, 1.3]], dtype=torch.float64, requires_grad=True
    )
    geo = LocalGeometry()

    def value(coords, cell):
        edge = geo(coords, cell, feats)
        return torch.cat((edge.distance, edge.direction.flatten(), edge.omega))

    assert torch.autograd.gradcheck(value, (x, lattice), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_permutation_equivariance_and_no_target_leakage():
    model = small_module().double().eval()
    with torch.no_grad():
        model.output_projection.weight.normal_(std=0.1)
    H, t, feats, x, lattice = branch_inputs(torch.float64)
    expected = run_branch(model, H, t, feats, x, lattice)
    changed = deepcopy(feats)
    for key in ("cart_coords", "lattice", "cart_coords_0", "lattice_0"):
        changed[key] = torch.full_like(changed[key], 12345)
    torch.testing.assert_close(
        run_branch(model, H, t, changed, x, lattice), expected, rtol=0, atol=0
    )
    order = torch.tensor([2, 0, 3, 1])
    for key in (
        "atom_pad_mask",
        "ref_element",
        "ref_charge",
        "ref_pos",
        "block_index",
        "block_type",
    ):
        changed[key] = feats[key][:, order]
    changed["bond_types"] = feats["bond_types"][:, order][:, :, order]
    actual = run_branch(model, H[:, order], t, changed, x[:, order], lattice)
    torch.testing.assert_close(actual, expected[:, order], rtol=1e-10, atol=1e-11)


def dense_reference(model, H, t, feats, edges, condition):
    """Direct v0 concatenation + global reductions, without split projections."""
    mask = feats["atom_pad_mask"].bool()
    active = mask & edges.geometry_ok[:, None]
    a = model.atom_projection(model.atom_norm(H.masked_fill(~active[..., None], 0)))
    c = model.condition_projection(condition.masked_fill(~active[..., None], 0))
    time = model.time_embedding(t)[:, None].expand(-1, H.shape[1], -1)
    features = (
        model.atom_features(torch.cat((a, c, time), -1))
        .masked_fill(~active[..., None], 0)
        .flatten(0, 1)
    )
    chi = model.relations._chunk(edges.distance, edges.direction, edges.same_block, edges.bond)
    env = model.environment
    state = env.init(features[edges.anchors])
    for _ in range(model.rounds):
        q = torch.cat(
            (features[edges.atom], features[edges.anchors[edges.env]], state[edges.env], chi), -1
        )
        z = env.messages.finish(env.messages.first(q))
        heads = env.heads(z)
        p = edges.omega * heads[:, 0].sigmoid()
        mass = state.new_zeros(len(state)).index_add(0, edges.env, p)
        numerator = torch.zeros_like(state).index_add(0, edges.env, p[:, None] * heads[:, 1:])
        u = numerator / (1 + mass[:, None])
        q = torch.cat((state, u, mass.log1p()[:, None]), -1)
        state = state + (mass / (1 + mass))[:, None] * env.update(env.update_norm(q))
    read = model.readback
    y = read.messages.finish(
        read.messages.first(torch.cat((features[edges.atom], state[edges.env], chi), -1))
    )
    heads = read.heads(y)
    logits = heads[:, 0] + edges.omega.log()
    maximum = features.new_zeros(len(features)).scatter_reduce(
        0, edges.atom, logits.detach(), reduce="amax"
    )
    w = (logits - maximum[edges.atom]).exp()
    denominator = (-maximum).exp().index_add(0, edges.atom, w)
    numerator = torch.zeros_like(features).index_add(0, edges.atom, w[:, None] * heads[:, 1:])
    return (
        H + model.output_projection(numerator / denominator[:, None]).reshape_as(H)
    ).masked_fill(~mask[..., None], 0)


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("chunk", [1, 7, 8192])
def test_chunked_equations_and_all_gradients_match_dense(checkpointing, chunk):
    torch.manual_seed(19)
    model = small_module(checkpointing, chunk).double().train()
    with torch.no_grad():
        model.output_projection.weight.normal_(std=0.1)
    other = deepcopy(model)
    H, t, feats, x, lattice = branch_inputs(torch.float64)
    left = [v.clone().requires_grad_() for v in (H, t, x, lattice)]
    right = [v.clone().requires_grad_() for v in (H, t, x, lattice)]
    a = run_branch(model, left[0], left[1], feats, left[2], left[3])
    edges = other.geometry(right[2], right[3], feats)
    b = dense_reference(other, right[0], right[1], feats, edges, other.encode_condition(feats))
    torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-10)
    weight = torch.randn_like(a)
    (a * weight).sum().backward()
    (b * weight).sum().backward()
    for p, q in zip(left, right):
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-8, atol=1e-9)
    for (name, p), (_, q) in zip(model.named_parameters(), other.named_parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-8, atol=1e-9, msg=name)


@pytest.mark.parametrize(
    "empty", ["hydrogen", "no_neighbours", "invalid_cell", "invalid_coordinates"]
)
def test_empty_paths_preserve_output_and_connect_all_parameters(empty):
    model = small_module(True).train()
    with torch.no_grad():
        model.output_projection.weight.normal_()
    H, t, feats, x, lattice = branch_inputs()
    if empty == "hydrogen":
        feats["ref_element"].fill_(1)
    elif empty == "no_neighbours":
        feats["atom_pad_mask"][:, 1:] = False
        lattice[:, :3] = 4
    elif empty == "invalid_cell":
        lattice[:, 0] = -1
    else:
        x[:, 0, 0] = float("nan")
    edges = model.geometry(x, lattice, feats)
    assert edges.atom.numel() == 0
    out = model(H, t, feats, edges, model.encode_condition(feats))
    torch.testing.assert_close(
        out, H.masked_fill(~feats["atom_pad_mask"].bool()[..., None], 0), rtol=0, atol=0
    )
    out.square().sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_initial_identity_and_output_projection_receives_first_gradient():
    model = small_module(True)
    H, t, feats, x, lattice = branch_inputs()
    out = run_branch(model, H, t, feats, x, lattice)
    torch.testing.assert_close(
        out, H.masked_fill(~feats["atom_pad_mask"].bool()[..., None], 0), rtol=0, atol=0
    )
    (out * torch.randn_like(out)).sum().backward()
    assert model.output_projection.weight.grad.abs().sum() > 0
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        if name != "output_projection.weight":
            assert torch.count_nonzero(p.grad) == 0, name


def test_disabled_branch_preserves_backbone_rng_and_state():
    torch.manual_seed(42)
    base = FlowModule(**flow_args())
    state = torch.get_rng_state().clone()
    torch.manual_seed(42)
    disabled = FlowModule(
        **flow_args(), g1new={"enabled": False, "module": {"_target_": "missing"}}
    )
    assert torch.equal(state, torch.get_rng_state())
    assert base.state_dict().keys() == disabled.state_dict().keys()
    for name, value in base.state_dict().items():
        assert torch.equal(value, disabled.state_dict()[name])
    with pytest.raises(ValueError, match="independent alternatives"):
        FlowModule(**flow_args(), g1={"enabled": True}, g1new={"enabled": True})


def test_real_flow_layer_multiplicity_cache_and_ablation():
    torch.manual_seed(11)
    model = FlowModule(**flow_args(), g1new=new_config()).eval()
    perturb(model)
    data = inputs(2)
    events = []
    handle = model.g1new_bridge.module.register_forward_hook(lambda *args: events.append("new"))
    for i, layer in enumerate(model.token_transformer.token_transformer.layers, 1):
        layer.register_forward_hook(lambda *args, i=i: events.append(i))
    with torch.no_grad():
        uncached = model(**data)
        expected = list(range(1, 13)) + ["new"] + list(range(13, 25))
        assert events == expected
        cache = model.prepare_sampling_conditions(data["feats"])
        cached = model(**data, static_condition=cache)
        for key in cached:
            torch.testing.assert_close(cached[key], uncached[key], rtol=0, atol=0)
        model.set_g1new_enabled(False)
        with pytest.raises(ValueError, match="different model"):
            model(**data, static_condition=cache)
        baseline = FlowModule(**flow_args()).eval()
        baseline.load_state_dict(
            {k: v for k, v in model.state_dict().items() if not k.startswith("g1new_bridge.")},
            strict=True,
        )
        for key, value in baseline(**data).items():
            torch.testing.assert_close(model(**data)[key], value, rtol=0, atol=0)
    handle.remove()


def test_hydra_config_and_parameter_registration():
    import os

    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("PROJECT_ROOT", str(root))
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        cfg = compose(config_name="train", overrides=["model=g1new", "execution=optimized"])
    model = instantiate(cfg.model)
    flow = model.structure_module.flow_model
    assert flow.g1new_enabled and not flow.g1_enabled and flow.g1_bridges is None
    group = flow.g1new_bridge
    module = group.first_bridge
    assert group.insert_after == (6, 12, 18, 24)
    assert module.width == 64 and module.rounds == 2
    assert module.condition_encoder.hidden_dim == 128
    assert len(module.condition_encoder.blocks) == 1
    assert sum(p.numel() for p in flow.parameters()) == 39_479_632
    assert sum(p.numel() for p in group.parameters()) == 1_103_312
    assert sum(p.numel() for p in module.parameters()) == 556340


@pytest.mark.parametrize(
    "stage,atoms,batch,accum,square",
    [
        (1, 200, 64, 1, 3000000),
        (2, 300, 32, 2, 3000000),
        (3, None, 32, 2, 500000),
    ],
)
def test_stage_configs(stage, atoms, batch, accum, square):
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base=None
    ):
        cfg = compose(
            config_name="train",
            overrides=[f"experiment=g1new_small_stage{stage}", "execution=optimized"],
        )
    assert cfg.model.flow_model_args.g1new.enabled and not cfg.model.flow_model_args.g1.enabled
    assert cfg.data.max_num_atoms == atoms
    assert cfg.data.dynamic.max_batch_size == batch
    assert cfg.data.dynamic.max_num_atoms_square == square
    assert cfg.trainer.accumulate_grad_batches == accum
    assert cfg.trainer.precision == "32-true"


def save_checkpoint(model, path):
    torch.save(
        dict(
            state_dict=model.state_dict(),
            hyper_parameters=dict(model.hparams),
            **{"pytorch-lightning_version": L.__version__},
        ),
        path,
    )


def test_checkpoint_roundtrip_and_explicit_transfer(tmp_path):
    args = model_args(False)
    args["flow_model_args"]["g1new"] = new_config()
    new = SimpleMOFModule(**args).eval()
    baseline = SimpleMOFModule(**model_args(False)).eval()
    perturb(baseline)
    path = tmp_path / "base.ckpt"
    save_checkpoint(baseline, path)
    report = load_g1new_pretrained(new, path)
    assert report["condition_tensors"] == 0
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, new.state_dict()[name])
    for key, value in baseline.structure_module.flow_model(**inputs()).items():
        result = new.structure_module.flow_model(**inputs())[key]
        # v0 explicitly zeros padded hidden states at the insertion. Only valid
        # atoms and the masked global lattice output are part of the model task.
        if key == "r_update":
            mask = inputs()["feats"]["atom_pad_mask"].bool()
            value, result = value[mask], result[mask]
        torch.testing.assert_close(value, result, rtol=0, atol=0)
    path = tmp_path / "new.ckpt"
    perturb(new)
    save_checkpoint(new, path)
    loaded = SimpleMOFModule.load_from_checkpoint(path, strict=True).eval()
    assert loaded.structure_module.flow_model.g1new_enabled
    for name, value in new.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name])
    with pytest.raises(ValueError, match="ordinary strict"):
        load_g1new_pretrained(loaded, path)


def test_legacy_condition_transfer_and_rejection(tmp_path):
    source = SimpleMOFModule(**model_args(True))
    args = model_args(False)
    args["flow_model_args"]["g1new"] = new_config()
    target = SimpleMOFModule(**args)
    initialized = {
        k: v.clone()
        for k, v in target.state_dict().items()
        if "g1new_bridge" in k and "condition_encoder" not in k
    }
    path = tmp_path / "old.ckpt"
    save_checkpoint(source, path)
    report = load_g1new_pretrained(target, path)
    assert report["condition_tensors"] > 0 and report["discarded_legacy_tensors"] > 0
    for key, value in initialized.items():
        assert torch.equal(target.state_dict()[key], value)
    old = source.structure_module.flow_model.g1_bridges.bridges["6"].condition_encoder
    new = target.structure_module.flow_model.g1new_bridge.module.condition_encoder
    for key, value in old.state_dict().items():
        assert torch.equal(new.state_dict()[key], value)
    ckpt = torch.load(path, weights_only=False)
    key = next(k for k in ckpt["state_dict"] if ".bridges.12.condition_encoder." in k)
    ckpt["state_dict"][key] = ckpt["state_dict"][key] + 1
    torch.save(ckpt, path)
    with pytest.raises(ValueError, match="aliases disagree"):
        load_g1new_pretrained(target, path)


@pytest.mark.parametrize("positions", [12, [6, 12, 18, 24]])
def test_real_lightning_train_save_resume_and_sampling(tmp_path, positions):
    from test_g1_integration import TinyDataset
    from torch.utils.data import DataLoader

    from src.data.prior.prior_sampler import CartCoordsSampler, LatticeSampler, PriorSampler

    args = model_args(False)
    args["flow_model_args"]["g1new"] = new_config()
    args["flow_model_args"]["g1new"]["insert_after"] = positions
    args["flow_process_args"]["prior_sampler"] = PriorSampler(
        [CartCoordsSampler(), LatticeSampler()]
    )
    model = SimpleMOFModule(**args)
    perturb(model)

    def trainer(steps):
        return L.Trainer(
            accelerator="cpu",
            devices=1,
            max_steps=steps,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            limit_val_batches=0,
            default_root_dir=str(tmp_path),
        )

    first = trainer(2)
    first.fit(model, train_dataloaders=DataLoader(TinyDataset(), batch_size=2))
    path = tmp_path / "trained.ckpt"
    first.save_checkpoint(path)
    loaded = SimpleMOFModule.load_from_checkpoint(path, strict=True)
    second = trainer(3)
    second.fit(
        loaded, train_dataloaders=DataLoader(TinyDataset(), batch_size=2), ckpt_path=str(path)
    )
    assert second.global_step == 3
    from src.models.execution import configure_execution

    configure_execution(loaded.structure_module.flow_model, {"cache_sampling_conditions": True})
    loaded.eval()
    with torch.inference_mode():
        sampled = loaded(
            inputs()["feats"], num_sampling_steps=2, multiplicity_flow_sample=2, refine_final=False
        )
    assert torch.isfinite(sampled["sampled_atom_coords"]).all()
    assert torch.isfinite(sampled["sampled_lattice"]).all()


def _ddp_worker(rank, path):
    from datetime import timedelta

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(path).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        config = new_config()
        config["insert_after"] = [6, 12, 18, 24]
        model = FlowModule(**flow_args(), g1new=config).train()
        perturb(model)
        wrapped = DistributedDataParallel(model, find_unused_parameters=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for _ in range(2):
            data = inputs()
            if rank == 1:
                data["feats"]["ref_element"].fill_(1)  # this rank has NO environments
            optimizer.zero_grad(set_to_none=True)
            loss(wrapped(**data)).backward()
            assert all(
                p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
            )
            optimizer.step()
        value = model.g1new_bridge.first_bridge.output_projection.weight.detach().clone()
        other = value.clone()
        dist.broadcast(other, 0)
        torch.testing.assert_close(value, other, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_two_rank_ddp_with_entirely_empty_rank(tmp_path):
    import torch.distributed as dist
    import torch.multiprocessing as mp

    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Gloo unavailable")
    mp.spawn(_ddp_worker, args=(str(tmp_path / "gloo"),), nprocs=2, join=True)
