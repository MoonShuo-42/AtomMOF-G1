"""Layer-specific G1 updates with optional shared static condition encoding."""

from collections.abc import Mapping, Sequence
from copy import deepcopy

import torch
from hydra.utils import instantiate
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .geometry import PeriodicGeometry
from .module import G1Module
from .outputs import G1Context
from .routing import SoftRouter
from .routing.groups import AtomCountGroups
from .validation import validate_mask


class G1Bridges(nn.Module):
    """Separate updates per site, optionally sharing one condition encoder.

    Hydra enables share_condition for new runs. Its constructor default is False
    so older checkpoint configs without the flag retain independent encoders.
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
        share_condition: bool = False,
    ):
        super().__init__()
        positions = tuple(insert_after)
        if (
            not positions
            or any(type(i) is not int or i < 1 or i > depth for i in positions)
            or positions != tuple(sorted(set(positions)))
        ):
            raise ValueError("G1 insert_after must be sorted, unique, one-based layers <= depth")
        if any(
            type(value) is not bool
            for value in (activation_checkpointing, share_geometry, share_condition)
        ):
            raise TypeError("G1 checkpointing and sharing flags must be bool")
        self.insert_after = positions
        self.activation_checkpointing = activation_checkpointing
        self.share_geometry = share_geometry
        self.share_condition = share_condition
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
            if share_condition:
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
        # Never silently overwrite a shared weight with four different trained copies.
        # Old checkpoint configs omit share_condition and therefore bypass this guard.
        if self.share_condition:
            first = str(self.insert_after[0])
            encoder = self.bridges[first].condition_encoder
            for name in encoder.state_dict():
                reference = state_dict.get(f"{prefix}bridges.{first}.condition_encoder.{name}")
                if reference is None:
                    continue
                for layer in self.insert_after[1:]:
                    candidate = state_dict.get(f"{prefix}bridges.{layer}.condition_encoder.{name}")
                    if candidate is not None and not torch.equal(reference, candidate):
                        error_msgs.append(
                            "Cannot load independent condition encoder weights into shared G1; "
                            "use share_condition=False for the old architecture, or start "
                            "a new shared-condition training run."
                        )
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

    def prepare(
        self, feats: Mapping[str, Tensor], x_t: Tensor, l_t: Tensor, multiplicity: int = 1
    ) -> dict[str, G1Context]:
        """Encode conditions once when shared, plus current expanded noisy geometry.

        Static conditions are encoded at the original batch size, then repeated
        in exactly the same order as AtomMOF's train/sampling multiplicity.
        Nothing is detached or retained across calls / sampling steps.
        """
        if type(multiplicity) is not int or multiplicity < 1:
            raise ValueError("multiplicity must be a positive integer")
        mask = validate_mask(feats["atom_pad_mask"]).repeat_interleave(multiplicity, 0)
        if x_t.shape != (*mask.shape, 3) or l_t.shape != (mask.shape[0], 6):
            raise ValueError("G1 noisy state must match the multiplicity-expanded atom mask")
        contexts = {}
        shared = None
        shared_condition = None
        layouts = {}
        for key, bridge in self.bridges.items():
            if not self.share_condition or shared_condition is None:
                c_cond = bridge.encode_condition(feats).repeat_interleave(multiplicity, 0)
                if self.share_condition:
                    shared_condition = c_cond
            else:
                c_cond = shared_condition
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
                    layouts[capacity] = selector.prepare(mask)
                layout = layouts[capacity]
            else:
                layout = selector.prepare(mask)
            contexts[key] = G1Context(c_cond, geo, mask, layout)
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
            return checkpoint(update, H, t, context, use_reentrant=False)
        return update(H, t, context)
