"""Layer-specific G1 updates sharing one static condition encoder."""

from collections.abc import Mapping, Sequence
from copy import deepcopy

import torch
from hydra.utils import instantiate
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from src.models.execution.time import time_feature_scope

from .geometry import PeriodicGeometry
from .module import G1Module
from .outputs import G1Context
from .routing import SoftRouter
from .routing.groups import AtomCountGroups
from .validation import validate_mask
from .aggregation.geometry import prepare_geometry_layout
from src.models.execution.ops import repeat_batch


class G1Bridges(nn.Module):
    """Separate updates per site, always sharing one condition encoder.

    Shared weights are registered by reference in each bridge; PyTorch deduplicates
    them in parameters()/optimizers. Contexts live only in the current forward.
    Shared geometry requires the same parameter-free PeriodicGeometry at each site.
    """

    def __init__(
        self,
        h_dim: int,
        depth: int,
        insert_after: Sequence[int] = (6, 12, 18, 24),
        module: Mapping | None = None,
        activation_checkpointing: bool = False,
        share_geometry: bool = True,
    ):
        super().__init__()
        positions = tuple(insert_after)
        if (
            not positions
            or any(type(i) is not int or i < 1 or i > depth for i in positions)
            or positions != tuple(sorted(set(positions)))
        ):
            raise ValueError("G1 insert_after must be sorted, unique, one-based layers <= depth")
        if any(type(value) is not bool for value in (activation_checkpointing, share_geometry)):
            raise TypeError("G1 checkpointing and sharing flags must be bool")
        self.insert_after = positions
        self.activation_checkpointing = activation_checkpointing
        self.share_geometry = share_geometry
        self.bridges = nn.ModuleDict()
        condition_encoder = None
        for position in positions:
            bridge = (
                G1Module(router=SoftRouter(h_dim=h_dim))
                if module is None
                else instantiate(deepcopy(module))
            )
            if not isinstance(bridge, G1Module) or bridge.h_dim != h_dim:
                raise ValueError("Every G1 bridge must be a G1Module with h_dim == token_s")
            if share_geometry and (
                type(bridge.geometry) is not PeriodicGeometry
                or any(True for _ in bridge.geometry.parameters())
            ):
                raise ValueError("share_geometry requires the parameter-free PeriodicGeometry")
            if bridge.geometry.length_unit != "nm" or bridge.geometry.angle_unit != "radian":
                raise ValueError("Flow supplies G1 current coordinates/lattice in nm and radian")
            if condition_encoder is None:
                condition_encoder = bridge.condition_encoder
            elif bridge.c_dim != condition_encoder.output_dim:
                raise ValueError("Shared condition encoder must match every bridge's c_dim")
            else:
                bridge.condition_encoder = condition_encoder
            self.bridges[str(position)] = bridge

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
        # Shared aliases must agree; reject inconsistent state instead of overwriting.
        first = str(self.insert_after[0])
        encoder = self.bridges[first].condition_encoder
        for name in encoder.state_dict():
            reference = state_dict.get(f"{prefix}bridges.{first}.condition_encoder.{name}")
            if reference is None:
                continue
            for layer in self.insert_after[1:]:
                candidate = state_dict.get(f"{prefix}bridges.{layer}.condition_encoder.{name}")
                if candidate is not None and not torch.equal(reference, candidate):
                    error_msgs.append("Inconsistent weights for the shared G1 condition encoder.")
                    break
            if error_msgs:
                break
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def encode_condition(self, feats):
        first = self.bridges[str(self.insert_after[0])]
        return first.encode_condition(feats)

    def prepare(
        self,
        feats: Mapping[str, Tensor],
        x_t: Tensor,
        l_t: Tensor,
        multiplicity: int = 1,
        *, c_cond: Tensor | None = None,
    ) -> dict[str, G1Context]:
        """Encode conditions once, plus current expanded noisy geometry.

        Static conditions are encoded at the original batch size, then repeated
        in exactly the same order as AtomMOF's train/sampling multiplicity.
        Nothing is detached or retained across calls / sampling steps.
        """
        if type(multiplicity) is not int or multiplicity < 1:
            raise ValueError("multiplicity must be a positive integer")
        mask = repeat_batch(validate_mask(feats["atom_pad_mask"]), multiplicity,
                            enabled=getattr(self, "avoid_repeat_copies", False))
        if x_t.shape != (*mask.shape, 3) or l_t.shape != (mask.shape[0], 6):
            raise ValueError("G1 noisy state must match the multiplicity-expanded atom mask")
        contexts = {}
        shared = None
        if c_cond is None:
            c_cond = self.encode_condition(feats)
        c_cond = repeat_batch(c_cond, multiplicity,
                              enabled=getattr(self, "avoid_repeat_copies", False))
        layouts = {}
        geometry_layouts = {}
        for key, bridge in self.bridges.items():
            if not self.share_geometry or shared is None:
                geo = bridge.geometry(x_t, l_t, mask)
                if self.share_geometry:
                    shared = geo
            else:
                geo = shared
            selector = bridge.router.group_selector
            # Reuse only the known atom-count policy; custom selectors prepare independently.
            capacity = selector.max_groups
            if type(selector) is AtomCountGroups:
                if capacity not in layouts:
                    kwargs = {}
                    if getattr(self, "prepared_layouts", False) and "g1_atom_counts" in feats:
                        kwargs = dict(
                            atom_counts=repeat_batch(feats["g1_atom_counts"], multiplicity,
                                                    enabled=getattr(self, "avoid_repeat_copies", False)),
                            max_atoms=feats["g1_batch_max_atoms"],
                        )
                    layouts[capacity] = selector.prepare(mask, **kwargs)
                layout = layouts[capacity]
            else:
                layout = selector.prepare(mask)
            prepared = None
            if getattr(self, "geometry_layout_reuse", False):
                geometry_key = id(geo)
                if geometry_key not in geometry_layouts:
                    geometry_layouts[geometry_key] = prepare_geometry_layout(geo.E, mask, geo.geometry_ok)
                prepared = geometry_layouts[geometry_key]
            contexts[key] = G1Context(c_cond, geo, mask, layout, prepared)
        return contexts

    def forward(
        self, layer: int, H: Tensor, t: Tensor, contexts: Mapping[str, G1Context]
    ) -> Tensor:
        key = str(layer)
        if key not in self.bridges:
            return H
        bridge, context = self.bridges[key], contexts[key]

        # Bind this bridge in a new call frame, including during backward recomputation.
        def update(hidden, times, prepared):
            return bridge.forward_with_context(hidden, times, prepared).H_updated

        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            def recomputable_update(hidden, times, prepared):
                # Recompute must execute the same operations as its own forward;
                # it cannot depend on another bridge's forward-local autograd cache.
                with time_feature_scope(False):
                    return update(hidden, times, prepared)
            return checkpoint(recomputable_update, H, t, context, use_reentrant=False)
        return update(H, t, context)
