"""Layer-specific G1-New updates with one static encoder and one geometry search."""

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass

import torch
from hydra.utils import instantiate
from torch import Tensor, nn

from src.models.execution.ops import repeat_batch

from .geometry import LocalPairs
from .module import G1NewModule


@dataclass
class G1NewContext:
    """Owned by ONE network forward; never stored on the model/checkpoint."""

    feats: dict
    edges: LocalPairs
    condition: Tensor
    times: Tensor | None = None
    time_features: Tensor | None = None
    relation_features: Tensor | None = None


class G1NewBridge(nn.Module):
    """Existing backbone adapter interface, now scheduling multiple insertions.

    The encoder is shared by reference (parameters() deduplicates it). Each site
    owns its other learned weights. Geometry and fixed time features are shared
    only within a forward; layer-dependent H, F, chi and states are never reused.
    """

    def __init__(self, h_dim, depth, insert_after=(6, 12, 18, 24), module=None):
        super().__init__()
        if type(insert_after) is int:
            positions = (insert_after,)
        elif isinstance(insert_after, Sequence) and not isinstance(insert_after, (str, bytes)):
            positions = tuple(insert_after)
        else:
            raise ValueError("g1new.insert_after must be an integer or a sequence of layers")
        if (
            not positions
            or any(type(i) is not int or not 1 <= i <= depth for i in positions)
            or positions != tuple(sorted(set(positions)))
        ):
            raise ValueError(
                "g1new.insert_after must be sorted, unique, one-based layers <= depth"
            )
        self.insert_after = positions
        self.bridges = nn.ModuleDict()
        for position in positions:
            shared = (
                {}
                if not self.bridges
                else dict(
                    condition_encoder=self.first_bridge.condition_encoder,
                    geometry=self.first_bridge.geometry,
                )
            )
            # Do not initialize and discard an extra encoder at every insertion.
            bridge = (
                G1NewModule(h_dim=h_dim, **shared)
                if module is None
                else instantiate(deepcopy(module), **shared)
            )
            if not isinstance(bridge, G1NewModule) or bridge.h_dim != h_dim:
                raise ValueError("g1new module.h_dim must equal token_s")
            if self.bridges:
                if any(getattr(bridge, name) is not value for name, value in shared.items()):
                    raise RuntimeError("G1-New factory must preserve shared module identities")
                bridge.time_embedding = self.first_bridge.time_embedding
            self.bridges[str(position)] = bridge

    @property
    def first_bridge(self):
        return self.bridges[str(self.insert_after[0])]

    @property
    def module(self):
        """Single-site accessor retained for code using the original adapter."""
        if len(self.bridges) != 1:
            raise AttributeError("Multiple G1-New sites: select bridges[str(layer)] explicitly")
        return self.first_bridge

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Original single-site checkpoints used module.*. Reconstructing their
        # saved scalar insert_after keeps parameter order and optimizer IDs.
        legacy = prefix + "module."
        legacy_keys = [key for key in state_dict if key.startswith(legacy)]
        if legacy_keys:
            if len(self.bridges) != 1 or any(
                key.startswith(prefix + "bridges.") for key in state_dict
            ):
                error_msgs.append(
                    "Single-site G1-New checkpoint requires its original single-site config"
                )
            else:
                destination = prefix + f"bridges.{self.insert_after[0]}."
                for key in legacy_keys:
                    state_dict[destination + key[len(legacy) :]] = state_dict.pop(key)
        # Shared aliases must agree before they reach the shared encoder.
        first_prefix = prefix + f"bridges.{self.insert_after[0]}.condition_encoder."
        for suffix in self.first_bridge.condition_encoder.state_dict():
            reference = state_dict.get(first_prefix + suffix)
            if reference is None:
                continue
            for layer in self.insert_after[1:]:
                value = state_dict.get(prefix + f"bridges.{layer}.condition_encoder." + suffix)
                if value is not None and (
                    value.dtype != reference.dtype or not torch.equal(value, reference)
                ):
                    error_msgs.append("G1-New shared condition encoder aliases disagree")
                    break
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def encode_condition(self, feats):
        return self.first_bridge.encode_condition(feats)

    def prepare(self, feats, x_t, l_t, multiplicity=1, condition=None):
        if type(multiplicity) is not int or multiplicity < 1:
            raise ValueError("multiplicity must be a positive integer")
        enabled = getattr(self, "avoid_repeat_copies", False)
        expanded = {
            name: repeat_batch(feats[name], multiplicity, enabled=enabled)
            for name in ("atom_pad_mask", "ref_element", "bond_types", "block_index")
        }
        if condition is None:
            condition = self.encode_condition(feats)
        condition = repeat_batch(condition, multiplicity, enabled=enabled)
        # x_t/l_t remain fixed across insertions within one network evaluation.
        # Keep the live graph: geometry gradients from ALL sites accumulate.
        edges = self.first_bridge.geometry(x_t, l_t, expanded)
        relation_features = (
            self.first_bridge.relations.geometry_features(edges)
            if self.first_bridge.optimized else None
        )
        return G1NewContext(expanded, edges, condition, relation_features=relation_features)

    def forward(self, layer, hidden, times, context):
        key = str(layer)
        if key not in self.bridges:
            return hidden
        if context.time_features is None:
            context.times = times
            context.time_features = self.first_bridge.time_embedding(times)
        elif context.times is not times:
            raise ValueError("G1-New context cannot be reused with different times")
        return self.bridges[key](
            hidden,
            times,
            context.feats,
            context.edges,
            context.condition,
            time_features=context.time_features,
            relation_features=context.relation_features,
        )
