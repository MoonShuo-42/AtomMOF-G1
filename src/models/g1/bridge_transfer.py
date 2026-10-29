"""Weights-only G1 depth expansion; existing execution and forward remain unchanged."""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn

from .bridges import G1Bridges
from .checkpoint import GroupExpansion, _load_state_with_group_expansion
from .readback.projection import AtomResidualUpdate


@dataclass(frozen=True)
class BridgeExpansion:
    old_positions: tuple[int, ...]
    new_positions: tuple[int, ...]
    added_positions: tuple[int, ...]
    group_expansions: tuple[GroupExpansion, ...]
    added_bridge_parameters: int
    total_parameters: int


def _positions(value, name):
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
        or any(type(i) is not int or i < 1 for i in value)
        or tuple(value) != tuple(sorted(set(value)))
    ):
        raise ValueError(f"{name}.insert_after must be sorted, unique, positive integer layers")
    return tuple(value)


def _g1_config(config, name):
    flow = config.get("flow_model_args", {})
    g1 = flow.get("g1", {})
    if g1.get("enabled") is not True or (flow.get("g1new") or {}).get("enabled", False):
        raise ValueError(f"{name} must enable original G1 only")
    return g1


def _schedule(model):
    schedules = [
        (name, module) for name, module in model.named_modules() if isinstance(module, G1Bridges)
    ]
    if len(schedules) != 1:
        raise ValueError("Bridge expansion requires exactly one G1Bridges schedule")
    return schedules[0]


@torch.no_grad()
def build_pretrained_with_bridge_expansion(
    model_config: Mapping,
    checkpoint_path: str | Path,
    *,
    expand_groups: bool = False,
) -> tuple[nn.Module, BridgeExpansion]:
    """Build a denser G1 model without disturbing the old-sites initialization.

    First instantiate the requested model with ONLY the checkpoint's insertion
    sites, then load old weights (optionally expanding K exactly as before).
    Construct the denser model in a CPU RNG fork and copy this reference state.
    New branches retain normal initialization except for their zero final
    residual projection; ALL condition-encoder aliases receive the old encoder.

    Thus the old sites' new template rows and the caller's CPU RNG state match
    the existing group-only transfer. Target hyperparameters describe the full
    architecture, so later strict loading/resume never repeats zero-init.
    Does not restore source optimizer/scheduler/epoch state. Intended for the
    normal CPU construction path, before Trainer moves the model to CUDA/DDP.
    """
    if type(expand_groups) is not bool:
        raise ValueError("expand_groups must be a boolean")
    target_config = deepcopy(
        OmegaConf.to_container(model_config, resolve=True)
        if OmegaConf.is_config(model_config)
        else dict(model_config)
    )
    target_g1 = _g1_config(target_config, "target")
    new_positions = _positions(target_g1.get("insert_after"), "target")
    depth = target_config["flow_model_args"]["token_transformer_depth"]
    if new_positions[-1] > depth:
        raise ValueError("target.insert_after exceeds token_transformer_depth")
    # EMA copies need a separate migration contract.
    if target_config.get("ema", False):
        raise ValueError("Bridge expansion currently requires ema=false")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping) or not isinstance(
        checkpoint.get("state_dict"), Mapping
    ):
        raise ValueError("Bridge expansion requires a checkpoint containing state_dict")
    hyperparameters = checkpoint.get("hyper_parameters")
    if not isinstance(hyperparameters, Mapping):
        raise ValueError("Bridge expansion requires checkpoint hyper_parameters")
    source_g1 = _g1_config(hyperparameters, "source")
    old_positions = _positions(source_g1.get("insert_after"), "source")
    if not set(old_positions) < set(new_positions):
        raise ValueError("Only adding G1 insertion sites is supported; retain every old site")

    reference_config = deepcopy(target_config)
    reference_config["flow_model_args"]["g1"]["insert_after"] = list(old_positions)
    reference = instantiate(reference_config)
    if any(p.device.type != "cpu" for p in reference.parameters()):
        raise ValueError("Bridge expansion must run on CPU before Trainer setup")
    source = checkpoint["state_dict"]
    if expand_groups:
        changes = _load_state_with_group_expansion(reference, source)
    else:
        # Prevent load_state_dict's implicit dtype conversion, also at fixed K.
        for key, value in reference.state_dict().items():
            if key in source and (
                not isinstance(source[key], torch.Tensor) or source[key].dtype != value.dtype
            ):
                raise ValueError(f"Bridge expansion cannot change tensor dtype: {key}")
        reference.load_state_dict(source, strict=True)
        changes = []

    # Do not let construction of new sites alter the established training RNG.
    with torch.random.fork_rng(devices=[]):
        model = instantiate(target_config)
    prefix, schedule = _schedule(model)
    ref_prefix, ref_schedule = _schedule(reference)
    if prefix != ref_prefix or ref_schedule.insert_after != old_positions:
        raise ValueError("Unexpected G1 schedule while expanding insertion sites")
    prefix = f"{prefix}.bridges." if prefix else "bridges."
    added = tuple(i for i in new_positions if i not in old_positions)
    source_state, target_state = reference.state_dict(), model.state_dict()
    new_prefixes = tuple(f"{prefix}{i}." for i in added)
    if source_state.keys() - target_state.keys() or any(
        not key.startswith(new_prefixes) for key in target_state.keys() - source_state.keys()
    ):
        raise ValueError("State keys changed outside the added G1 bridges")
    for key, value in source_state.items():
        if target_state[key].shape != value.shape or target_state[key].dtype != value.dtype:
            raise ValueError(f"Existing tensor changed while adding G1 bridges: {key}")
        target_state[key] = value

    condition_prefix = f"{prefix}{old_positions[0]}.condition_encoder."
    for position in added:
        bridge = schedule.bridges[str(position)]
        if type(bridge.readback.residual) is not AtomResidualUpdate:
            raise ValueError("Zero-init requires the standard G1 AtomResidualUpdate")
        # Added first site (e.g. 3) must share the TRAINED encoder, not its fresh one.
        for name in bridge.condition_encoder.state_dict():
            target_state[f"{prefix}{position}.condition_encoder.{name}"] = source_state[
                condition_prefix + name
            ]
        output_key = f"{prefix}{position}.readback.residual.output_projection.weight"
        target_state[output_key] = torch.zeros_like(target_state[output_key])
    model.load_state_dict(target_state, strict=True)
    total = sum(p.numel() for p in model.parameters())
    return model, BridgeExpansion(
        old_positions,
        new_positions,
        added,
        tuple(changes),
        total - sum(p.numel() for p in reference.parameters()),
        total,
    )
