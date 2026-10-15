"""Real AtomMOF/G1 integration; the unused optional MLIP backend is isolated on CPU."""

import importlib.util
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import lightning as L
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

# CPU validation does not install the optional ASE/fairchem MLIP stack. Never replace
# Flow, DiT, G1, the sampler, Lightning, or the actual atomic-overlap implementation.
if importlib.util.find_spec("fairchem") is None:
    mlip = ModuleType("src.models.energy.mlip")

    class UnavailableMLIP:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("MLIP is not exercised by these CPU integration tests")

    mlip.MLIPEnergy = UnavailableMLIP
    sys.modules.setdefault("src.models.energy.mlip", mlip)

from src.models.g1.bridges import G1Bridges
from src.models.simple.flow import AtomFlowMatching, FlowModule
from src.models.simplemof_module import SimpleMOFModule


@pytest.fixture(scope="module", autouse=True)
def small_cpu_workloads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def flow_args():
    return dict(
        atom_s=16,
        atom_z=8,
        token_s=16,
        token_z=8,
        atom_feature_dim=10,
        atom_encoder_depth=1,
        atom_encoder_heads=4,
        atom_encoder_positional_encoding=False,
        token_transformer_depth=24,
        token_transformer_heads=4,
        atom_decoder_depth=1,
        atom_decoder_heads=4,
        attention_impl="pytorch",
        activation_checkpointing=False,
    )


def bridge_config(checkpoint=False, share=True, share_condition=True):
    return dict(
        enabled=True,
        insert_after=[6, 12, 18, 24],
        activation_checkpointing=checkpoint,
        share_geometry=share,
        share_condition=share_condition,
        module={
            "_target_": "src.models.g1.G1Module",
            "condition_encoder": {
                "_target_": "src.models.g1.conditioning.ConditionEncoder",
                "atom_s": 16,
                "atom_z": 8,
                "hidden_dim": 8,
                "pair_dim": 4,
                "activation_checkpointing": checkpoint,
                "output_dim": 8,
                "depth": 1,
                "heads": 4,
            },
            "router": {
                "_target_": "src.models.g1.routing.SoftRouter",
                "h_dim": 16,
                "c_dim": 8,
                "time_dim": 8,
                "hidden_dim": 16,
                "key_dim": 8,
                "max_groups": 3,
            },
            "group_interaction": {
                "_target_": "src.models.g1.group_attention.GroupInteraction",
                "h_dim": 16,
                "group_dim": 8,
                "heads": 2,
            },
            "readback": {
                "_target_": "src.models.g1.readback.AtomReadback",
                "h_dim": 16,
                "group_dim": 8,
                "heads": 2,
                "head_dim": 4,
                "value_hidden_dim": 12,
            },
        },
    )


def feats():
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    generator = torch.Generator().manual_seed(79)
    coords = torch.rand(2, 4, 3, generator=generator) * mask[..., None]
    lattice = torch.tensor([[10.0, 10.0, 10.0, 90.0, 90.0, 90.0]]).repeat(2, 1)
    return dict(
        ref_pos=coords,
        ref_charge=torch.zeros(2, 4),
        ref_element=torch.tensor([[6, 8, 30, 0], [6, 8, 0, 0]]),
        block_type=torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]]),
        block_index=torch.tensor([[1, 1, 2, 0], [3, 3, 0, 0]]),
        bond_types=torch.zeros(2, 4, 4, dtype=torch.long),
        atom_pad_mask=mask,
        cart_coords=coords * 3,
        lattice=lattice,
        cart_coords_0=coords * 6,
        lattice_0=lattice + torch.tensor([1.0, 1.0, 1.0, 0, 0, 0]),
    )


def inputs(multiplicity=1):
    data = feats()
    lattice = data["lattice"].clone()
    lattice[:, :3] *= 0.1
    lattice[:, 3:] *= torch.pi / 180
    return dict(
        r_noisy=(data["cart_coords"] * 0.1).repeat_interleave(multiplicity, 0),
        l_noisy=lattice.repeat_interleave(multiplicity, 0),
        times=torch.tensor([0.2, 0.7]).repeat_interleave(multiplicity, 0),
        feats=data,
        multiplicity=multiplicity,
    )


def perturb(model):
    """Open originally zero output heads and time gates to exercise every gradient."""
    with torch.no_grad():
        for p in model.parameters():
            p.uniform_(-0.15, 0.15)


def loss(out):
    return out["r_update"].square().sum() + out["l_update"].square().sum()


def test_disabled_creates_no_bridges_and_preserves_initialization_rng_and_state():
    torch.manual_seed(41)
    old_interface = FlowModule(**flow_args())
    rng = torch.get_rng_state().clone()
    torch.manual_seed(41)
    # An unimportable G1 config must never be evaluated while disabled.
    disabled = FlowModule(**flow_args(), g1={"enabled": False, "module": {"_target_": "bad"}})
    assert disabled.g1_bridges is None and not disabled.g1_enabled
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert old_interface.state_dict().keys() == disabled.state_dict().keys()
    for key, value in old_interface.state_dict().items():
        torch.testing.assert_close(disabled.state_dict()[key], value, rtol=0, atol=0)
    perturb(old_interface)
    disabled.load_state_dict(old_interface.state_dict(), strict=True)
    a, b = old_interface(**inputs(2)), disabled(**inputs(2))
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    loss(a).backward()
    loss(b).backward()
    for p, q in zip(old_interface.parameters(), disabled.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


def test_bridges_run_after_6_12_18_24_with_shared_condition_and_independent_updates():
    model = FlowModule(**flow_args(), g1=bridge_config())
    perturb(model)
    events, handles = [], []
    for i, block in enumerate(model.token_transformer.token_transformer.layers, 1):
        handles.append(block.register_forward_hook(lambda m, a, o, i=i: events.append(("dit", i))))
    for key, bridge in model.g1_bridges.bridges.items():
        handles.append(
            bridge.readback.register_forward_hook(
                lambda m, a, o, key=key: events.append(("g1", int(key)))
            )
        )
    out = model(**inputs(2))
    expected = []
    for i in range(1, 25):
        expected.append(("dit", i))
        if i % 6 == 0:
            expected.append(("g1", i))
    assert events == expected
    assert out["r_update"].shape == (4, 4, 3) and out["l_update"].shape == (4, 6)
    loss(out).backward()
    parameters = []
    for bridge in model.g1_bridges.bridges.values():
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all() for p in bridge.parameters()
        )
        assert sum(p.grad.abs().sum() for p in bridge.parameters()) > 0
        parameters.extend(
            p.data_ptr()
            for name, p in bridge.named_parameters()
            if not name.startswith("condition_encoder.")
        )
    assert len(parameters) == len(set(parameters))
    assert len({id(b.condition_encoder) for b in model.g1_bridges.bridges.values()}) == 1
    for handle in handles:
        handle.remove()


def test_current_geometry_is_shared_and_recomputed_and_conditions_follow_multiplicity():
    schedule = G1Bridges(
        h_dim=16, depth=24, **{k: v for k, v in bridge_config().items() if k != "enabled"}
    )
    data, calls = inputs(2), []
    handles = [
        b.geometry.register_forward_hook(lambda m, a, o: calls.append(o))
        for b in schedule.bridges.values()
    ]
    contexts = schedule.prepare(data["feats"], data["r_noisy"], data["l_noisy"], 2)
    assert len(calls) == 1
    assert len({id(c.geometry) for c in contexts.values()}) == 1
    for key, context in contexts.items():
        encoded = schedule.bridges[key].encode_condition(data["feats"])
        torch.testing.assert_close(context.c_cond, encoded.repeat_interleave(2, 0))
        assert context.c_cond.grad_fn is not None
    other = schedule.prepare(data["feats"], data["r_noisy"] * 1.3, data["l_noisy"], 2)
    assert len(calls) == 2 and other["6"].geometry is not contexts["6"].geometry
    for handle in handles:
        handle.remove()


def test_bridge_and_dit_checkpointing_preserve_outputs_and_all_gradients():
    plain = FlowModule(**flow_args(), g1=bridge_config())
    perturb(plain)
    args = flow_args()
    args["activation_checkpointing"] = True
    checked = FlowModule(**args, g1=bridge_config(checkpoint=True))
    checked.load_state_dict(plain.state_dict(), strict=True)
    a, b = plain(**inputs()), checked(**inputs())
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    loss(a).backward()
    loss(b).backward()
    for (name, p), (_, q) in zip(plain.named_parameters(), checked.named_parameters()):
        assert p.grad is not None and q.grad is not None, name
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-5, atol=1e-7, msg=name)


def test_eval_ablation_matches_baseline_and_cannot_enable_missing_weights():
    enabled = FlowModule(**flow_args(), g1=bridge_config()).eval()
    perturb(enabled)
    baseline = FlowModule(**flow_args()).eval()
    baseline.load_state_dict(
        {k: v for k, v in enabled.state_dict().items() if not k.startswith("g1_bridges.")},
        strict=True,
    )
    enabled.set_g1_enabled(False)
    for key, value in baseline(**inputs()).items():
        torch.testing.assert_close(enabled(**inputs())[key], value, rtol=0, atol=0)
    enabled.set_g1_enabled(True)
    assert not torch.allclose(enabled(**inputs())["r_update"], baseline(**inputs())["r_update"])
    with pytest.raises(ValueError, match="no G1 weights"):
        baseline.set_g1_enabled(True)
    with pytest.raises(TypeError):
        enabled.set_g1_enabled("false")
    enabled.train()
    with pytest.raises(RuntimeError, match="training/DDP"):
        enabled.set_g1_enabled(False)


@pytest.mark.parametrize("positions", [[], [0], [25], [12, 6], [6, 6], [True]])
def test_invalid_insertion_sites_fail_at_construction(positions):
    with pytest.raises(ValueError, match="insert_after"):
        G1Bridges(h_dim=16, depth=24, insert_after=positions)


def model_args(enabled):
    args = flow_args()
    sizes = {
        k: args.pop(k) for k in ["atom_s", "atom_z", "token_s", "token_z", "atom_feature_dim"]
    }
    if enabled:
        args["g1"] = bridge_config()
    return dict(
        **sizes,
        flow_model_args=args,
        flow_process_args={"prior_sampler": None},
        training_args=OmegaConf.create(
            dict(
                train_multiplicity=2,
                loss_type="l2",
                use_dist_loss=False,
                dist_cutoff=15.0,
                use_dist_schedule=False,
                coord_loss_weight=1.0,
                lengths_loss_weight=1.0,
                angle_loss_weight=0.01,
                dist_loss_weight=0.0,
                lr=1e-4,
                weight_decay=0.0,
                lr_scheduler=None,
            )
        ),
        validation_args={"structure_matcher_args": {}, "sample_every_n_epochs": 0},
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_checkpoint_architecture_roundtrip_and_legacy_strict_loading(tmp_path, enabled):
    model = SimpleMOFModule(**model_args(enabled)).eval()
    perturb(model)
    path = tmp_path / "model.ckpt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": L.__version__,
        },
        path,
    )
    loaded = SimpleMOFModule.load_from_checkpoint(path, strict=True).eval()
    assert loaded.structure_module.flow_model.g1_enabled is enabled
    for key, value in model.structure_module.flow_model(**inputs()).items():
        torch.testing.assert_close(
            loaded.structure_module.flow_model(**inputs())[key], value, rtol=0, atol=0
        )


def test_real_flow_training_and_sampling_use_all_bridges_and_units():
    args = flow_args()
    args["g1"] = bridge_config()
    flow = AtomFlowMatching(flow_model_args=args, prior_sampler=None)
    perturb(flow)
    calls = []
    handle = flow.flow_model.g1_bridges.bridges["6"].geometry.register_forward_pre_hook(
        lambda m, a: calls.append(a)
    )
    result = flow(feats(), multiplicity=2)
    losses = flow.compute_loss(feats(), result, multiplicity=2)
    sum(v.mean() for v in losses.values()).backward()
    assert all(p.grad is not None for p in flow.flow_model.g1_bridges.parameters())
    torch.testing.assert_close(calls[0][1][:, 3:], torch.full((4, 3), torch.pi / 2))
    flow.eval()
    calls.clear()
    batch = feats()
    sampled = flow.sample(
        atom_mask=batch["atom_pad_mask"],
        feats=batch,
        multiplicity=2,
        num_sampling_steps=2,
        center_coords=False,
        refine_final=True,
        cached_priors={
            k: batch[k].repeat_interleave(2, 0) for k in ["cart_coords_0", "lattice_0"]
        },
    )
    assert len(calls) == 3  # two ODE evaluations and the existing final refinement
    assert sampled["sampled_atom_coords"].shape == (4, 4, 3)
    assert torch.isfinite(sampled["sampled_atom_coords"]).all()
    handle.remove()


@pytest.mark.parametrize("entry", ["train", "predict"])
def test_real_hydra_entries_compose_deferred_config_and_follow_token_width(entry):
    os.environ.setdefault("PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base=None
    ):
        cfg = compose(
            config_name=entry,
            overrides=["model.token_s=16", "model.token_z=8", "model.atom_s=16", "model.atom_z=8"],
        )
    model = instantiate(cfg.model)
    assert model.structure_module.flow_model.g1_bridges is None
    assert cfg.model.flow_model_args.g1.module.router.h_dim == 16
    assert not isinstance(model.hparams.flow_model_args.g1.module, torch.nn.Module)
    if entry == "predict":
        assert cfg.predict_args.g1_enabled is None
    cfg.model.flow_model_args.g1.enabled = True
    model = instantiate(cfg.model)
    assert list(model.structure_module.flow_model.g1_bridges.bridges) == ["6", "12", "18", "24"]
    assert all(
        b.h_dim == 16 for b in model.structure_module.flow_model.g1_bridges.bridges.values()
    )


class TinyDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {k: v[index] for k, v in feats().items()}


def test_lightning_optimizer_runs_two_real_training_steps():
    model = SimpleMOFModule(**model_args(True))
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
    )
    parameter = model.structure_module.flow_model.atom_decoder.feats_to_coords[1].weight
    before = parameter.detach().clone()
    trainer.fit(model, train_dataloaders=DataLoader(TinyDataset(), batch_size=2))
    assert trainer.global_step == 2 and not torch.equal(before, parameter)


def _ddp_worker(rank, init_file, checkpoint):
    from datetime import timedelta

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(init_file).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        model = FlowModule(**flow_args(), g1=bridge_config(checkpoint=checkpoint))
        wrapped = DistributedDataParallel(model, find_unused_parameters=False)
        optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-3)
        for step in range(2):
            data = inputs()
            if rank == 1:
                data["l_noisy"][0, 0] = -1  # one rank takes geometry fallback
            optimizer.zero_grad(set_to_none=True)
            loss(wrapped(**data)).backward()
            assert all(
                p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
            )
            optimizer.step()
        value = next(model.g1_bridges.parameters()).detach().clone()
        other = value.clone()
        dist.broadcast(other, src=0)
        torch.testing.assert_close(value, other, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("checkpoint", [False, True])
def test_two_rank_ddp_uses_all_parameters_with_shared_condition(tmp_path, checkpoint):
    import torch.distributed as dist
    import torch.multiprocessing as mp

    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Gloo is unavailable in this PyTorch build")
    mp.spawn(_ddp_worker, args=(str(tmp_path / "gloo_init"), checkpoint), nprocs=2, join=True)


@pytest.mark.parametrize("requested,expected", [(None, True), (False, False), (True, True)])
def test_prediction_entry_honors_checkpoint_and_override(
    tmp_path, monkeypatch, requested, expected
):
    import src.predict as entry

    model = SimpleMOFModule(**model_args(True))
    ckpt = tmp_path / "run" / "checkpoints" / "model.ckpt"
    ckpt.parent.mkdir(parents=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": L.__version__,
        },
        ckpt,
    )
    cfg = OmegaConf.create(
        {
            "ckpt_path": str(ckpt),
            "seed": 42,
            "data": {"_target_": "test.Data", "predict_split": "test"},
            "trainer": {"_target_": "test.Trainer", "devices": 1},
            "predict_args": {
                "g1_enabled": requested,
                "num_samples": 1,
                "sampling_steps": 2,
                "refine_final": False,
                "use_sde_coords": False,
                "measure_time": False,
                "steering_args": {"fk_steering": False, "energy_fn": "atomic_overlap"},
            },
        }
    )
    seen = []

    class Trainer:
        def predict(self, loaded, **kwargs):
            flow = loaded.structure_module.flow_model
            assert flow.g1_enabled is expected
            assert torch.isfinite(flow(**inputs())["r_update"]).all()
            seen.append(True)

    # Loading still constructs all real G1 weights; instantiate is also used there.
    original_instantiate = instantiate

    def instantiate_entry(config, **kwargs):
        if config.get("_target_") == "test.Trainer":
            return Trainer()
        if config.get("_target_") == "test.Data":
            return object()
        return original_instantiate(config, **kwargs)

    monkeypatch.setattr(entry.hydra.utils, "instantiate", instantiate_entry)
    entry.predict(cfg)
    assert seen == [True]
    dirname = entry.set_dir_name(cfg)
    assert ("_g1-" in dirname) is (requested is not None)


def test_baseline_checkpoint_can_initialize_g1_without_changing_backbone(tmp_path):
    baseline = SimpleMOFModule(**model_args(False))
    perturb(baseline)
    path = tmp_path / "baseline.ckpt"
    torch.save(
        {
            "state_dict": baseline.state_dict(),
            "hyper_parameters": dict(baseline.hparams),
            "pytorch-lightning_version": L.__version__,
        },
        path,
    )
    fresh = SimpleMOFModule(**model_args(True))
    with pytest.warns(UserWarning, match="checkpoint but not in the model|not in the checkpoint"):
        loaded = SimpleMOFModule.load_from_checkpoint(path, strict=False, **fresh.hparams)
    assert loaded.structure_module.flow_model.g1_enabled
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)


def test_hydra_checkpoint_restores_g1_configuration_and_can_resume(tmp_path):
    os.environ.setdefault("PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base=None
    ):
        cfg = compose(
            config_name="train",
            overrides=[
                "model.token_s=16",
                "model.token_z=8",
                "model.atom_s=16",
                "model.atom_z=8",
                "model.flow_model_args.g1.enabled=true",
                "model.flow_model_args.g1.module.condition_encoder.atom_s=16",
                "model.flow_model_args.g1.module.condition_encoder.atom_z=8",
                "model.flow_model_args.g1.module.condition_encoder.depth=1",
                "model.flow_model_args.g1.module.condition_encoder.hidden_dim=8",
                "model.flow_model_args.g1.module.condition_encoder.pair_dim=4",
            ],
        )
    model = instantiate(cfg.model)
    path = tmp_path / "hydra_g1.ckpt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": L.__version__,
        },
        path,
    )
    loaded = SimpleMOFModule.load_from_checkpoint(path, strict=True)
    assert loaded.structure_module.flow_model.g1_enabled
    assert loaded.hparams.flow_model_args.g1.share_condition is True
    assert loaded.hparams.flow_model_args.g1.module.router.h_dim == 16
    assert loaded.hparams.flow_model_args.g1.module.condition_encoder.hidden_dim == 8
    assert loaded.hparams.flow_model_args.g1.module.condition_encoder.pair_dim == 4
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)


def test_small_parameter_budget_keeps_baseline_and_other_g1_components_unchanged():
    os.environ.setdefault("PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base=None
    ):
        cfg = compose(config_name="train", overrides=["model.token_s=256", "model.token_z=128"])
    baseline = instantiate(cfg.model)
    baseline_count = sum(p.numel() for p in baseline.parameters())
    assert baseline_count == 38_376_320
    del baseline
    cfg.model.flow_model_args.g1.enabled = True
    model = instantiate(cfg.model)
    assert sum(p.numel() for p in model.parameters()) == 40_525_216
    bridges = model.structure_module.flow_model.g1_bridges.bridges
    assert list(bridges) == ["6", "12", "18", "24"]
    expected = {
        "condition_encoder": 374_016,
        "geometry": 0,
        "router": 60_128,
        "aggregation": 0,
        "group_interaction": 231_876,
        "readback": 151_716,
    }
    for bridge in bridges.values():
        assert {
            name: sum(p.numel() for p in getattr(bridge, name).parameters()) for name in expected
        } == expected
        assert sum(p.numel() for p in bridge.parameters()) == 817_736


def test_shared_condition_runs_once_and_receives_the_sum_of_all_bridge_gradients():
    shared = FlowModule(**flow_args(), g1=bridge_config())
    independent = FlowModule(**flow_args(), g1=bridge_config(share_condition=False))
    perturb(shared)
    independent.load_state_dict(shared.state_dict(), strict=True)
    encoder = shared.g1_bridges.bridges["6"].condition_encoder
    calls = []
    handle = encoder.register_forward_hook(lambda m, a, o: calls.append(o))
    try:
        out = shared(**inputs(2))
        assert len(calls) == 1
        # A later forward must recompute, with a fresh graph.
        shared(**inputs())
        assert len(calls) == 2
    finally:
        handle.remove()
    reference = independent(**inputs(2))
    for key in out:
        torch.testing.assert_close(out[key], reference[key], rtol=0, atol=0)
    loss(out).backward()
    loss(reference).backward()
    individual = [
        dict(b.condition_encoder.named_parameters())
        for b in independent.g1_bridges.bridges.values()
    ]
    for name, param in encoder.named_parameters():
        expected = sum(parameters[name].grad for parameters in individual)
        torch.testing.assert_close(param.grad, expected, rtol=3e-5, atol=2e-6, msg=name)
    data = inputs(2)
    contexts = shared.g1_bridges.prepare(data["feats"], data["r_noisy"], data["l_noisy"], 2)
    assert len({id(c.c_cond) for c in contexts.values()}) == 1
    assert contexts["6"].c_cond.grad_fn is not None
    parameter_ids = [id(p) for p in shared.parameters()]
    assert len(parameter_ids) == len(set(parameter_ids))


def test_old_compact_checkpoint_without_sharing_flag_keeps_independent_encoders(tmp_path):
    args = model_args(True)
    del args["flow_model_args"]["g1"]["share_condition"]
    old = SimpleMOFModule(**args).eval()
    assert not old.structure_module.flow_model.g1_bridges.share_condition
    assert (
        len(
            {
                id(b.condition_encoder)
                for b in old.structure_module.flow_model.g1_bridges.bridges.values()
            }
        )
        == 4
    )
    path = tmp_path / "independent_compact.ckpt"
    torch.save(
        {
            "state_dict": old.state_dict(),
            "hyper_parameters": dict(old.hparams),
            "pytorch-lightning_version": L.__version__,
        },
        path,
    )
    restored = SimpleMOFModule.load_from_checkpoint(path, strict=True).eval()
    assert not restored.structure_module.flow_model.g1_bridges.share_condition
    for key, value in old.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    shared = SimpleMOFModule(**model_args(True))
    with pytest.raises(RuntimeError, match="independent condition encoder weights"):
        shared.load_state_dict(old.state_dict(), strict=True)


def test_stage1_experiment_aligns_small_batch_and_disables_all_recomputation():
    os.environ.setdefault("PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base=None
    ):
        cfg = compose(config_name="train", overrides=["experiment=g1_small_stage1"])
    assert cfg.trainer.devices == 8 and cfg.trainer.accumulate_grad_batches == 2
    assert cfg.trainer.precision == "32-true" and cfg.trainer.max_epochs == 500
    assert cfg.data.max_num_atoms == 200 and cfg.data.dynamic.max_batch_size == 32
    assert cfg.model.token_s == 256 and cfg.model.token_z == 128
    flow = cfg.model.flow_model_args
    assert not flow.activation_checkpointing
    assert flow.g1.enabled and flow.g1.share_condition and flow.g1.share_geometry
    assert not flow.g1.activation_checkpointing
    assert not flow.g1.module.condition_encoder.activation_checkpointing
    assert not flow.g1.module.geometry.activation_checkpointing
    # All shipped defaults also disable recomputation without an experiment override.
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base=None
    ):
        default = compose(config_name="train")
    assert not default.model.flow_model_args.activation_checkpointing
    assert not default.model.flow_model_args.g1.activation_checkpointing
    assert not default.model.flow_model_args.g1.module.condition_encoder.activation_checkpointing
    assert not default.model.flow_model_args.g1.module.geometry.activation_checkpointing
