"""Depth transfer preserves the established K-expansion start and grows zero branches."""

import os
from copy import deepcopy
from pathlib import Path

import lightning as L
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from test_g1_checkpoint_expansion import save_checkpoint
from test_g1_integration import (
    SimpleMOFModule,
    TinyDataset,
    inputs,
    loss,
    model_args,
    perturb,
)
from torch.utils.data import DataLoader

from src.models.g1.bridge_transfer import build_pretrained_with_bridge_expansion
from src.models.g1.checkpoint import load_pretrained_with_group_expansion

OLD = [6, 12, 18, 24]
DENSE = [3, 6, 9, 12, 15, 18, 21, 24]
PREFIX = "structure_module.flow_model.g1_bridges.bridges."


@pytest.fixture(scope="module", autouse=True)
def small_cpu_workloads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(groups=3, positions=DENSE, optimized=False):
    cfg = model_args(True)
    cfg["_target_"] = "src.models.simplemof_module.SimpleMOFModule"
    g1 = cfg["flow_model_args"]["g1"]
    g1.update(_target_="builtins.dict", _recursive_=False)
    g1["insert_after"] = list(positions)
    g1["module"]["router"]["max_groups"] = groups
    if optimized:
        profile = OmegaConf.load(
            Path(__file__).resolve().parents[1] / "configs/execution/optimized.yaml"
        )
        cfg["flow_model_args"]["execution"] = OmegaConf.to_container(
            profile.model.flow_model_args.execution,
            resolve=True,
        )
    return cfg


def source_checkpoint(tmp_path, groups=2):
    torch.manual_seed(11)
    source = instantiate(config(groups, OLD))
    perturb(source)
    return source, save_checkpoint(tmp_path / "source.ckpt", source)


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("expand", [False, True])
def test_preserves_reference_state_rng_outputs_and_shared_condition(tmp_path, optimized, expand):
    source, path = source_checkpoint(tmp_path, groups=2 if expand else 3)
    before_bytes = path.read_bytes()
    torch.manual_seed(42)
    reference = instantiate(config(3, OLD, optimized))
    if expand:
        load_pretrained_with_group_expansion(reference, path)
    else:
        reference.load_state_dict(source.state_dict(), strict=True)
    rng = torch.get_rng_state().clone()

    torch.manual_seed(42)
    cfg = config(3, DENSE, optimized)
    original_config = deepcopy(cfg)
    model, report = build_pretrained_with_bridge_expansion(cfg, path, expand_groups=expand)
    assert cfg == original_config
    assert torch.equal(torch.get_rng_state(), rng)
    assert path.read_bytes() == before_bytes
    assert report.old_positions == tuple(OLD)
    assert report.added_positions == (3, 9, 15, 21)
    assert len(report.group_expansions) == (4 if expand else 0)
    assert report.total_parameters == sum(p.numel() for p in model.parameters())
    assert report.added_bridge_parameters == report.total_parameters - sum(
        p.numel() for p in reference.parameters()
    )
    assert list(model.hparams.flow_model_args["g1"]["insert_after"]) == DENSE
    assert all(p.requires_grad for p in model.parameters())
    for key, old in reference.state_dict().items():
        assert torch.equal(model.state_dict()[key], old), key

    bridges = model.structure_module.flow_model.g1_bridges.bridges
    assert len({id(b.condition_encoder) for b in bridges.values()}) == 1
    for bridge in bridges.values():
        for key, value in (
            reference.structure_module.flow_model.g1_bridges.bridges["6"]
            .condition_encoder.state_dict()
            .items()
        ):
            assert torch.equal(bridge.condition_encoder.state_dict()[key], value)

    observed = []
    handles = []

    def verify_identity(module, args, output):
        valid = args[5].bool()
        assert torch.equal(output.H_updated[valid], args[0][valid])
        assert torch.count_nonzero(output.delta_H) == 0
        observed.append(True)

    for position in report.added_positions:
        handles.append(bridges[str(position)].readback.register_forward_hook(verify_identity))
    data = inputs()
    expected = reference.structure_module.flow_model(**data)
    actual = model.structure_module.flow_model(**data)
    assert len(observed) == 4
    for handle in handles:
        handle.remove()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-6, atol=1e-7)

    loss(expected).backward()
    loss(actual).backward()
    target_params = dict(model.named_parameters())
    for key, param in reference.named_parameters():
        # The shared encoder's canonical name moves from site 6 to site 3.
        target_key = key.replace(f"{PREFIX}6.condition_encoder.", f"{PREFIX}3.condition_encoder.")
        if param.grad is not None:
            torch.testing.assert_close(
                target_params[target_key].grad, param.grad, rtol=1e-5, atol=1e-7
            )
    for position in report.added_positions:
        bridge = bridges[str(position)]
        for name, param in bridge.named_parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
            if name.startswith("condition_encoder."):
                continue
            if name == "readback.residual.output_projection.weight":
                assert param.grad.abs().sum() > 0
            else:
                assert torch.count_nonzero(param.grad) == 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    loss(model.structure_module.flow_model(**data)).backward()
    for position in report.added_positions:
        grad = bridges[str(position)].router.template_bank.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_trains_saves_strictly_reloads_and_resumes_without_rezero(tmp_path):
    _, path = source_checkpoint(tmp_path)
    model, report = build_pretrained_with_bridge_expansion(
        config(optimized=True), path, expand_groups=True
    )

    def trainer(steps):
        return L.Trainer(
            accelerator="cpu",
            devices=1,
            precision="32-true",
            max_steps=steps,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            limit_val_batches=0,
            default_root_dir=str(tmp_path),
        )

    run = trainer(2)
    run.fit(model, train_dataloaders=DataLoader(TinyDataset(), batch_size=2))
    assert run.global_step == 2
    for position in report.added_positions:
        key = f"{PREFIX}{position}.readback.residual.output_projection.weight"
        assert torch.count_nonzero(model.state_dict()[key]) > 0
    checkpoint_path = tmp_path / "expanded.ckpt"
    run.save_checkpoint(checkpoint_path)
    loaded = SimpleMOFModule.load_from_checkpoint(checkpoint_path, strict=True)
    for key, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value), key
    assert list(loaded.hparams.flow_model_args["g1"]["insert_after"]) == DENSE
    run = trainer(3)
    run.fit(
        loaded,
        train_dataloaders=DataLoader(TinyDataset(), batch_size=2),
        ckpt_path=str(checkpoint_path),
    )
    assert run.global_step == 3
    for position in report.added_positions:
        parameter = loaded.structure_module.flow_model.g1_bridges.bridges[
            str(position)
        ].readback.residual.output_projection.weight
        assert int(run.optimizers[0].state[parameter]["step"]) == 3


@pytest.mark.parametrize(
    "positions", [OLD, [3, 6, 9, 12, 15, 18, 21], [3, 6, 6, 12, 18, 24], DENSE + [25]]
)
def test_rejects_removing_moving_duplicate_or_invalid_sites(tmp_path, positions):
    _, path = source_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="insertion sites|insert_after"):
        build_pretrained_with_bridge_expansion(config(3, positions), path, expand_groups=True)


@pytest.mark.parametrize(
    "problem",
    ["missing", "extra", "shape", "dtype", "alias", "source_sites", "shrink", "implicit_k"],
)
def test_rejects_incompatible_weights_and_implicit_k_change(tmp_path, problem):
    source, path = source_checkpoint(tmp_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    backbone_key = next(k for k, v in state.items() if v.ndim == 2 and "g1_bridges" not in k)
    target = config()
    if problem == "missing":
        del state[backbone_key]
    elif problem == "extra":
        state["extra.weight"] = torch.ones(1)
    elif problem == "shape":
        state[backbone_key] = state[backbone_key][:-1]
    elif problem == "dtype":
        state[backbone_key] = state[backbone_key].double()
    elif problem == "alias":
        # Shared tensors must be cloned before corrupting just one saved alias.
        key = next(k for k in state if "bridges.12.condition_encoder." in k)
        state[key] = state[key].clone() + 1
    elif problem == "source_sites":
        checkpoint["hyper_parameters"]["flow_model_args"]["g1"]["insert_after"] = [6, 12, 24]
    elif problem == "shrink":
        target = config(1)
    torch.save(checkpoint, path)
    with pytest.raises((ValueError, RuntimeError)):
        build_pretrained_with_bridge_expansion(target, path, expand_groups=problem != "implicit_k")


@pytest.mark.parametrize(
    "pretrained,resume,flag,new_transfer",
    [
        (None, None, True, False),
        ("x", "y", True, False),
        ("x", None, "true", False),
        ("x", None, True, True),
    ],
)
def test_entry_rejects_invalid_transfer_before_instantiation(
    tmp_path, monkeypatch, pretrained, resume, flag, new_transfer
):
    import src.train as entry

    monkeypatch.setattr(
        entry.hydra.utils, "instantiate", lambda *a, **kw: pytest.fail("constructed too early")
    )
    cfg = OmegaConf.create(
        dict(
            pretrained_ckpt_path=pretrained,
            resume_ckpt_path=resume,
            pretrained_g1_expand_bridges=flag,
            pretrained_g1new_transfer=new_transfer,
            paths=dict(output_dir=str(tmp_path)),
        )
    )
    with pytest.raises(ValueError, match="pretrained_g1"):
        entry.train(cfg)


def test_training_entry_uses_expanded_architecture_without_resuming(tmp_path, monkeypatch):
    import src.train as entry

    _, path = source_checkpoint(tmp_path)
    seen = []

    class Trainer:
        callback_metrics = {}

        def fit(self, **kwargs):
            assert kwargs["ckpt_path"] is None
            model = kwargs["model"]
            assert model.structure_module.flow_model.g1_bridges.insert_after == tuple(DENSE)
            assert len(model.structure_module.flow_model.g1_bridges.bridges) == 8
            seen.append(model)

    original = entry.hydra.utils.instantiate

    def construct(cfg, **kwargs):
        if cfg._target_ == "test.Data":
            return object()
        if cfg._target_ == "test.Trainer":
            return Trainer()
        return original(cfg, **kwargs)

    monkeypatch.setattr(entry.hydra.utils, "instantiate", construct)
    cfg = OmegaConf.create(
        dict(
            model=config(),
            data={"_target_": "test.Data"},
            trainer={"_target_": "test.Trainer"},
            train=True,
            test=False,
            seed=42,
            pretrained_ckpt_path=str(path),
            resume_ckpt_path=None,
            pretrained_g1_expand_groups=True,
            pretrained_g1_expand_bridges=True,
            paths=dict(output_dir=str(tmp_path)),
        )
    )
    entry.train(cfg)
    assert len(seen) == 1


def _dense_ddp_worker(rank, init_file, checkpoint_path):
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
        torch.manual_seed(42)
        model, _ = build_pretrained_with_bridge_expansion(
            config(optimized=True), checkpoint_path, expand_groups=True
        )
        wrapped = DistributedDataParallel(
            model.structure_module.flow_model,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-3, weight_decay=0)
        for step in range(2):
            data = inputs()
            if rank == 1:
                data["l_noisy"][0, 0] = -1  # Different geometry fallback on one rank.
            optimizer.zero_grad(set_to_none=True)
            loss(wrapped(**data)).backward()
            assert all(
                p.grad is not None and torch.isfinite(p.grad).all() for p in wrapped.parameters()
            )
            optimizer.step()
        for position in (3, 9, 15, 21):
            value = wrapped.module.g1_bridges.bridges[
                str(position)
            ].readback.residual.output_projection.weight.detach()
            assert value.abs().sum() > 0
            other = value.clone()
            dist.broadcast(other, src=0)
            torch.testing.assert_close(value, other, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_zero_branches_work_with_two_rank_ddp(tmp_path):
    import torch.distributed as dist
    import torch.multiprocessing as mp

    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Gloo is unavailable in this PyTorch build")
    _, path = source_checkpoint(tmp_path)
    mp.spawn(_dense_ddp_worker, args=(str(tmp_path / "init"), str(path)), nprocs=2, join=True)


def compose_dense():
    os.environ.setdefault("PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base=None
    ):
        return compose(
            config_name="train",
            overrides=["experiment=g1_small_stage2_dense", "execution=optimized"],
        )


def test_experiment_config_and_periodic_checkpoints(tmp_path):
    cfg = compose_dense()
    assert cfg.trainer.max_epochs == 200 and cfg.trainer.accumulate_grad_batches == 2
    assert cfg.trainer.devices == 8 and cfg.trainer.precision == "32-true"
    assert cfg.data.max_num_atoms == 300 and cfg.data.dynamic.max_num_atoms_square == 3000000
    assert cfg.data.dynamic.max_batch_size == 32 and cfg.data.dynamic.skip_too_big is False
    assert cfg.model.flow_model_args.g1.module.router.max_groups == 64
    assert list(cfg.model.flow_model_args.g1.insert_after) == DENSE
    assert cfg.model.flow_model_args.execution.fused_qkvg is True
    assert cfg.model.training_args.lr_warmup_no_steps == 10000
    assert cfg.model.training_args.max_lr == 0.0001
    assert cfg.pretrained_g1_expand_bridges and cfg.pretrained_g1_expand_groups
    model = instantiate(cfg.model)
    assert sum(p.numel() for p in model.parameters()) == 42_308_288
    del model
    callback = instantiate(cfg.callbacks.model_checkpoint, dirpath=str(tmp_path))

    class Counter(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.ones(1))

        def training_step(self, batch, batch_idx):
            return self.value.square().sum()

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.001)

    run = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=200,
        logger=False,
        callbacks=[callback],
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        default_root_dir=str(tmp_path),
    )
    run.fit(Counter(), train_dataloaders=DataLoader(torch.zeros(1, 1), batch_size=1))
    assert {p.name for p in tmp_path.glob("*.ckpt")} == {
        "epoch_99-step_100.ckpt",
        "epoch_199-step_200.ckpt",
        "last.ckpt",
    }
    for name, epoch in [
        ("epoch_99-step_100.ckpt", 99),
        ("epoch_199-step_200.ckpt", 199),
        ("last.ckpt", 199),
    ]:
        saved = torch.load(tmp_path / name, map_location="cpu", weights_only=False)
        assert saved["epoch"] == epoch and saved["optimizer_states"]
