"""Weights-only stage transfer expands K without dropping or reinitializing weights."""

from pathlib import Path
import os

import lightning as L
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from test_g1_integration import SimpleMOFModule, TinyDataset, model_args, perturb
from src.models.g1.checkpoint import load_pretrained_with_group_expansion


@pytest.fixture(scope="module", autouse=True)
def small_cpu_workloads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_model(groups, optimized=False):
    args = model_args(True)
    args["flow_model_args"]["g1"]["module"]["router"]["max_groups"] = groups
    if optimized:
        profile = OmegaConf.load(Path(__file__).resolve().parents[1] / "configs/execution/optimized.yaml")
        args["flow_model_args"]["execution"] = OmegaConf.to_container(
            profile.model.flow_model_args.execution, resolve=True,
        )
    return SimpleMOFModule(**args)


def save_checkpoint(path, model, state=None):
    torch.save({
        "state_dict": model.state_dict() if state is None else state,
        "hyper_parameters": dict(model.hparams),
        "pytorch-lightning_version": L.__version__,
        "epoch": 499,
        "global_step": 163000,
        # Transfer must ignore optimizer history from the old architecture.
        "optimizer_states": [{"old_optimizer": True}],
    }, path)
    return path


def clone_state(model):
    return {key: value.clone() for key, value in model.state_dict().items()}


def test_32_to_64_preserves_every_old_weight_and_new_initialization(tmp_path):
    torch.manual_seed(123)
    source = make_model(32)
    perturb(source)
    path = save_checkpoint(tmp_path / "stage1.ckpt", source)
    checkpoint_bytes = path.read_bytes()
    torch.manual_seed(42)
    target = make_model(64, optimized=True)
    initialized = clone_state(target)
    parameters = dict(target.named_parameters())
    rng = torch.get_rng_state().clone()

    changes = load_pretrained_with_group_expansion(target, path)

    assert len(changes) == 4
    assert all((c.old_groups, c.new_groups) == (32, 64) for c in changes)
    assert sum(p.numel() for p in target.parameters()) - sum(
        p.numel() for p in source.parameters()
    ) == sum(c.added_parameters for c in changes)
    changed = {c.key for c in changes}
    for key, old in source.state_dict().items():
        actual = target.state_dict()[key]
        if key in changed:
            assert torch.equal(actual[:32], old), key
            assert torch.equal(actual[32:], initialized[key][32:]), key
        else:
            assert torch.equal(actual, old), key
    assert all(p is dict(target.named_parameters())[key] for key, p in parameters.items())
    assert all(p.requires_grad for p in target.parameters())
    assert torch.equal(torch.get_rng_state(), rng)
    assert path.read_bytes() == checkpoint_bytes

    # New rows participate in real routing and receive nonzero task gradients.
    for bridge in target.structure_module.flow_model.g1_bridges.bridges.values():
        router = bridge.router
        routing = router(
            torch.randn(1, 80, 16), torch.randn(1, 80, 8), torch.randn(1, 80, 11),
            torch.tensor([0.5]), torch.ones(1, dtype=torch.bool),
            torch.ones(1, 80, dtype=torch.bool),
        )
        assert routing.group_counts.tolist() == [64]
        (routing.P * torch.randn_like(routing.P)).sum().backward()
        grad = router.template_bank.weight.grad
        assert torch.isfinite(grad).all()
        assert grad[:32].abs().sum() > 0 and grad[32:].abs().sum() > 0

    # The new architecture is saved normally; no expansion hook is needed later.
    expanded_path = save_checkpoint(tmp_path / "stage2_k64.ckpt", target)
    restored = SimpleMOFModule.load_from_checkpoint(expanded_path, strict=True)
    assert restored.hparams.flow_model_args["g1"]["module"]["router"]["max_groups"] == 64
    for key, value in target.state_dict().items():
        assert torch.equal(restored.state_dict()[key], value), key


def test_expanded_optimized_model_trains_saves_and_resumes_at_same_k(tmp_path):
    torch.manual_seed(42)
    source = make_model(2)
    perturb(source)
    target = make_model(4, optimized=True)
    path = save_checkpoint(tmp_path / "source.ckpt", source)
    changes = load_pretrained_with_group_expansion(target, path)
    before = {c.key: target.state_dict()[c.key].clone() for c in changes}

    def make_trainer(steps):
        return L.Trainer(
            accelerator="cpu", devices=1, precision="32-true", max_steps=steps,
            logger=False, enable_checkpointing=False, enable_progress_bar=False,
            enable_model_summary=False, num_sanity_val_steps=0, limit_val_batches=0,
            default_root_dir=str(tmp_path),
        )

    trainer = make_trainer(2)
    trainer.fit(target, train_dataloaders=DataLoader(TinyDataset(), batch_size=2))
    assert trainer.global_step == 2  # not the source checkpoint's 163000
    for change in changes:
        value = target.state_dict()[change.key]
        assert not torch.equal(value[:2], before[change.key][:2])
        # TinyDataset has up to 3 atoms, so the third row is newly active.
        assert not torch.equal(value[2], before[change.key][2])
    trained_path = tmp_path / "expanded.ckpt"
    trainer.save_checkpoint(trained_path)
    restored = SimpleMOFModule.load_from_checkpoint(trained_path, strict=True)
    resumed_trainer = make_trainer(3)
    resumed_trainer.fit(
        restored, train_dataloaders=DataLoader(TinyDataset(), batch_size=2),
        ckpt_path=str(trained_path),
    )
    assert resumed_trainer.global_step == 3
    for bridge in restored.structure_module.flow_model.g1_bridges.bridges.values():
        weight = bridge.router.template_bank.weight
        optimizer_state = resumed_trainer.optimizers[0].state[weight]
        assert optimizer_state["exp_avg"].shape == (4, 8)
        assert optimizer_state["exp_avg_sq"].shape == (4, 8)


@pytest.mark.parametrize("problem", [
    "shrink", "key_dim", "other_shape", "missing", "unexpected", "dtype", "same_k",
])
def test_invalid_transfer_fails_before_copying_any_weights(tmp_path, problem):
    source = make_model(32)
    target = make_model(64)
    state = clone_state(source)
    template = next(k for k in state if k.endswith("template_bank.weight"))
    other = next(k for k, v in state.items() if v.ndim == 2 and "g1_bridges" not in k)
    if problem == "shrink":
        source, target = make_model(64), make_model(32)
        state = clone_state(source)
        message = "Only increasing"
    elif problem == "key_dim":
        state[template] = state[template][:, :-1]
        message = "key_dim"
    elif problem == "other_shape":
        state[other] = state[other][:-1]
        message = "outside G1 templates"
    elif problem == "missing":
        del state[other]
        message = "identical state_dict keys"
    elif problem == "unexpected":
        state["unexpected.weight"] = torch.ones(1)
        message = "identical state_dict keys"
    elif problem == "dtype":
        state[template] = state[template].half()
        message = "dtype"
    else:
        target = make_model(32)
        message = "No G1 group capacity increase"
    before = clone_state(target)
    path = save_checkpoint(tmp_path / "invalid.ckpt", source, state)
    with pytest.raises(ValueError, match=message):
        load_pretrained_with_group_expansion(target, path)
    for key, value in target.state_dict().items():
        assert torch.equal(value, before[key]), key


def test_expansion_is_explicit_and_shared_alias_checks_remain_strict(tmp_path):
    source, target = make_model(32), make_model(64)
    with pytest.raises(RuntimeError, match="size mismatch"):
        target.load_state_dict(source.state_dict(), strict=False)
    state = clone_state(source)
    alias = next(k for k in state if "bridges.12.condition_encoder" in k)
    state[alias].add_(1)
    path = save_checkpoint(tmp_path / "bad_alias.ckpt", source, state)
    with pytest.raises(RuntimeError, match="Inconsistent weights"):
        load_pretrained_with_group_expansion(target, path)


def test_expansion_rejects_baseline_and_invalid_checkpoint(tmp_path):
    baseline = SimpleMOFModule(**model_args(False))
    path = save_checkpoint(tmp_path / "baseline.ckpt", baseline)
    with pytest.raises(ValueError, match="identical state_dict keys"):
        load_pretrained_with_group_expansion(make_model(64), path)
    with pytest.raises(ValueError, match="requires a G1 model"):
        load_pretrained_with_group_expansion(baseline, path)
    torch.save({}, path)
    with pytest.raises(ValueError, match="containing state_dict"):
        load_pretrained_with_group_expansion(make_model(64), path)


def test_training_entry_expands_pretrained_without_resuming(tmp_path, monkeypatch):
    import src.train as entry

    source, target = make_model(32), make_model(64)
    path = save_checkpoint(tmp_path / "stage1.ckpt", source)
    seen = []

    class Trainer:
        callback_metrics = {}

        def fit(self, **kwargs):
            assert kwargs["model"] is target
            assert kwargs["ckpt_path"] is None
            for key, old in source.state_dict().items():
                actual = target.state_dict()[key]
                assert torch.equal(actual[:32] if key.endswith("template_bank.weight") else actual, old)
            assert target.hparams.flow_model_args["g1"]["module"]["router"]["max_groups"] == 64
            seen.append(True)

    objects = {"test.Model": target, "test.Data": object(), "test.Trainer": Trainer()}
    monkeypatch.setattr(entry.hydra.utils, "instantiate", lambda cfg, **kw: objects[cfg._target_])
    cfg = OmegaConf.create({
        "model": {"_target_": "test.Model"}, "data": {"_target_": "test.Data"},
        "trainer": {"_target_": "test.Trainer"}, "train": True, "test": False,
        "pretrained_ckpt_path": str(path), "pretrained_g1_expand_groups": True,
        "resume_ckpt_path": None, "paths": {"output_dir": str(tmp_path)},
    })
    entry.train(cfg)
    assert seen == [True]


@pytest.mark.parametrize("pretrained,resume,expand", [
    (None, None, True), ("source.ckpt", "resume.ckpt", True),
    (None, "resume.ckpt", True), ("source.ckpt", None, "true"),
])
def test_training_rejects_invalid_expansion_mode_before_model_creation(
    tmp_path, monkeypatch, pretrained, resume, expand,
):
    import src.train as entry

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid mode should fail before model or datamodule creation")

    monkeypatch.setattr(entry.hydra.utils, "instantiate", unexpected)
    cfg = OmegaConf.create({
        "pretrained_ckpt_path": pretrained, "resume_ckpt_path": resume,
        "pretrained_g1_expand_groups": expand, "paths": {"output_dir": str(tmp_path)},
    })
    with pytest.raises(ValueError, match="pretrained_g1_expand_groups"):
        entry.train(cfg)


def test_small_64_group_parameter_count_and_hydra_overrides():
    from hydra.utils import instantiate

    os.environ.setdefault("PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base=None,
    ):
        cfg = compose(config_name="train", overrides=[
            "execution=optimized", "model.token_s=256", "model.token_z=128",
            "model.flow_model_args.g1.enabled=true",
            "model.flow_model_args.g1.module.router.max_groups=64",
            "pretrained_g1_expand_groups=true",
        ])
    assert cfg.pretrained_g1_expand_groups is True
    model = instantiate(cfg.model)
    assert sum(p.numel() for p in model.parameters()) == 40_529_312
