"""Explicit, weights-only transfer from a smaller G1 group capacity."""

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .routing import SoftRouter


@dataclass(frozen=True)
class GroupExpansion:
    key: str
    old_groups: int
    new_groups: int
    key_dim: int

    @property
    def added_parameters(self) -> int:
        return (self.new_groups - self.old_groups) * self.key_dim


@torch.no_grad()
def load_pretrained_with_group_expansion(
    model: nn.Module, checkpoint_path: str | Path,
) -> list[GroupExpansion]:
    """Strictly load existing weights, extending only SoftRouter template rows.

    Construct the destination model with the desired max_groups and seed first.
    Old rows are copied exactly; additional rows keep TemplateBank's normal
    initialization. Every other key, shape and dtype must match. Neither the
    checkpoint nor the model's hyperparameters/Parameter objects are replaced.

    This is a stage-transfer operation, not an optimizer/scheduler/epoch resume.
    Ordinary load_state_dict/load_from_checkpoint remain strict about shapes.
    """
    # Lightning checkpoints include hyperparameter objects as well as tensors.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping) or not isinstance(
        checkpoint.get("state_dict"), Mapping
    ):
        raise ValueError("G1 expansion requires a checkpoint containing state_dict")
    return _load_state_with_group_expansion(model, checkpoint["state_dict"])


@torch.no_grad()
def _load_state_with_group_expansion(model: nn.Module, source: Mapping) -> list[GroupExpansion]:
    """Shared validation for group-only and bridge-plus-group stage transfer."""
    target = model.state_dict()
    missing = sorted(target.keys() - source.keys())
    unexpected = sorted(source.keys() - target.keys())
    if missing or unexpected:
        raise ValueError(
            "G1 expansion requires identical state_dict keys; "
            f"missing={missing}, unexpected={unexpected}"
        )

    template_keys = {
        f"{name}.template_bank.weight" if name else "template_bank.weight"
        for name, module in model.named_modules()
        if isinstance(module, SoftRouter)
    }
    if not template_keys:
        raise ValueError("G1 expansion requires a G1 model with SoftRouter templates")

    # Validate all shapes before copying any weights into the destination model.
    expansions = []
    for key, destination in target.items():
        saved = source[key]
        if not isinstance(saved, torch.Tensor) or not isinstance(destination, torch.Tensor):
            raise ValueError(f"G1 expansion requires tensor state entries: {key}")
        if saved.dtype != destination.dtype:
            raise ValueError(
                f"G1 expansion cannot change dtype: {key}: "
                f"{saved.dtype} -> {destination.dtype}"
            )
        if saved.shape == destination.shape:
            continue
        if key not in template_keys:
            raise ValueError(
                f"Shape mismatch outside G1 templates: {key}: "
                f"{tuple(saved.shape)} -> {tuple(destination.shape)}"
            )
        if saved.ndim != 2 or destination.ndim != 2 or saved.shape[1:] != destination.shape[1:]:
            raise ValueError(f"G1 expansion cannot change template key_dim: {key}")
        old_groups, new_groups = saved.shape[0], destination.shape[0]
        if not 0 < old_groups < new_groups:
            raise ValueError(
                "Only increasing G1 max_groups is supported: "
                f"{key}: {old_groups} -> {new_groups}"
            )
        expansions.append(GroupExpansion(key, old_groups, new_groups, destination.shape[1]))
    if not expansions:
        raise ValueError(
            "No G1 group capacity increase found; use ordinary pretrained loading "
            "when max_groups is unchanged"
        )

    expanded = OrderedDict(source)
    if hasattr(source, "_metadata"):
        expanded._metadata = source._metadata
    for change in expansions:
        weights = target[change.key].detach().clone()
        weights[:change.old_groups].copy_(source[change.key])
        expanded[change.key] = weights
    # Keep shared-condition alias validation and all normal strict checks active.
    model.load_state_dict(expanded, strict=True)
    return expansions
