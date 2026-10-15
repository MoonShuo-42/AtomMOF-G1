"""Independent G1 bridges scheduled after selected one-based trunk layers."""

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
from .validation import validate_mask


class G1Bridges(nn.Module):
    """One configuration, separately instantiated learned weights at every site.

    Contexts live only in the caller's current Flow forward, never on this module.
    Shared geometry is valid because every bridge has the same geometry config;
    only the existing parameter-free PeriodicGeometry supports this optimization.
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
        if type(activation_checkpointing) is not bool or type(share_geometry) is not bool:
            raise TypeError("G1 activation_checkpointing and share_geometry must be bool")
        self.insert_after = positions
        self.activation_checkpointing = activation_checkpointing
        self.share_geometry = share_geometry
        self.bridges = nn.ModuleDict()
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
            self.bridges[str(position)] = bridge

    def prepare(
        self, feats: Mapping[str, Tensor], x_t: Tensor, l_t: Tensor, multiplicity: int = 1
    ) -> dict[str, G1Context]:
        """Encode each bridge's conditions and the current, expanded noisy geometry.

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
        for key, bridge in self.bridges.items():
            c_cond = bridge.encode_condition(feats).repeat_interleave(multiplicity, 0)
            if not self.share_geometry or shared is None:
                geo = bridge.geometry(x_t, l_t, mask)
                if self.share_geometry:
                    shared = geo
            else:
                geo = shared
            contexts[key] = G1Context(c_cond, geo, mask)
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
