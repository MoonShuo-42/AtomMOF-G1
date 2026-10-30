"""Explicit weights-only transfer into a NEW architecture, never K expansion."""

from collections import OrderedDict
from collections.abc import Mapping

import torch


@torch.no_grad()
def load_g1new_pretrained(model, checkpoint_path):
    """Copy every backbone tensor; optionally copy the exact old G1 encoder.

    Discarded legacy bridge tensors are deliberately identified by namespace.
    New local-environment parameters retain their initialized values. All loaded
    names, shapes, dtypes and shared-encoder aliases are checked before mutation.
    Optimizer/scheduler/epoch state is not transferred."""
    prefix = "structure_module.flow_model."
    new_prefix = prefix + "g1new_bridge."
    old_prefix = prefix + "g1_bridges."
    target = model.state_dict()
    if not any(key.startswith(new_prefix) for key in target):
        raise ValueError("pretrained_g1new_transfer requires an enabled G1-New target")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping) or not isinstance(
        checkpoint.get("state_dict"), Mapping
    ):
        raise ValueError("Expected a Lightning checkpoint with state_dict")
    source = checkpoint["state_dict"]
    if any(key.startswith(new_prefix) for key in source):
        raise ValueError("Use ordinary strict pretrained/resume loading for a G1-New checkpoint")
    backbone = {key for key in target if not key.startswith(new_prefix)}
    source_backbone = {key for key in source if not key.startswith(old_prefix)}
    if source_backbone != backbone:
        raise ValueError(
            f"Backbone keys differ: missing={sorted(backbone-source_backbone)}, "
            f"unexpected={sorted(source_backbone-backbone)}"
        )
    replacement = OrderedDict(target)

    def checked_copy(key, value):
        destination = target[key]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != destination.shape
            or value.dtype != destination.dtype
        ):
            raise ValueError(f"G1-New transfer shape/dtype mismatch: {key}")
        replacement[key] = value

    for key in backbone:
        checked_copy(key, source[key])
    condition = {}
    for key, value in source.items():
        if key.startswith(old_prefix) and ".condition_encoder." in key:
            suffix = key.split(".condition_encoder.", 1)[1]
            if suffix in condition and not torch.equal(condition[suffix], value):
                raise ValueError("Legacy G1 shared condition aliases disagree")
            condition[suffix] = value
    if any(key.startswith(old_prefix) for key in source):
        condition_targets = {
            key: key.split(".condition_encoder.", 1)[1]
            for key in target
            if key.startswith(new_prefix) and ".condition_encoder." in key
        }
        expected = {suffix for suffix in condition_targets.values()}
        if condition.keys() != expected:
            raise ValueError("Legacy G1 condition encoder does not match the G1-New encoder")
        for key, suffix in condition_targets.items():
            checked_copy(key, condition[suffix])
    model.load_state_dict(replacement, strict=True)
    return dict(
        backbone_tensors=len(backbone),
        condition_tensors=len(condition),
        discarded_legacy_tensors=sum(
            key.startswith(old_prefix) and ".condition_encoder." not in key for key in source
        ),
    )
