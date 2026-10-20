"""Standalone G1 soft router with atom-count-dependent templates and groups."""

import math

import torch
from torch import Tensor, nn
from src.models.execution.time import shared_time_features

from .features import RouterQueryEncoder
from .groups import AtomCountGroups, GroupLayout
from .normalization import RoutingOutput, normalize_assignments
from .templates import TemplateBank
from .time import FourierTimeEmbedding


class SoftRouter(nn.Module):
    """Use K=min(max_groups,N_valid) active templates AND soft groups per sample.

    Parameters retain a stable maximum capacity; only the active prefix varies.
    Instantiate one router per bridge for independent parameters. This module
    does not create bridges or load baseline checkpoints.
    """

    def __init__(
        self,
        h_dim: int = 256,
        c_dim: int = 128,
        rho_dim: int = 11,
        time_dim: int = 32,
        hidden_dim: int = 128,
        key_dim: int = 32,
        max_groups: int = 32,
        time_frequency_divisor: float = 4.0,
        pool_eps: float = 1e-8,
    ):
        super().__init__()
        if not math.isfinite(pool_eps) or pool_eps <= 0:
            raise ValueError("pool_eps must be finite and positive")
        self.pool_eps = pool_eps
        self.max_groups = max_groups
        self.group_selector = AtomCountGroups(max_groups)
        self.time_embedding = FourierTimeEmbedding(time_dim, time_frequency_divisor)
        self.query_encoder = RouterQueryEncoder(
            h_dim, c_dim, rho_dim, time_dim, hidden_dim, key_dim
        )
        self.template_bank = TemplateBank(max_groups, key_dim)

    def forward(
        self,
        h: Tensor,
        c: Tensor,
        rho: Tensor,
        t: Tensor,
        geometry_ok: Tensor,
        atom_mask: Tensor,
        *,
        group_layout: GroupLayout | None = None,
    ) -> RoutingOutput:
        """Return P/Pbar (B,K_batch,N), mass/group_mask, and per-sample group_counts."""
        if h.ndim != 3 or atom_mask.shape != h.shape[:2]:
            raise ValueError("h must be (B,N,D) and atom_mask must be (B,N)")
        time_input = t
        if t.ndim == 2 and t.shape[-1] == 1:
            t = t[:, 0]
        if t.shape != (h.shape[0],) or t.device != h.device:
            raise ValueError("t must be (B,) or (B,1) on h's device")

        # All-padding samples must remain safe even if their unused t is nonfinite.
        def encode_time():
            safe_t = torch.where(atom_mask.bool().any(-1), t, 0)
            return self.time_embedding(safe_t)

        if (
            getattr(self, "cache_time_features", False)
            and type(self.time_embedding) is FourierTimeEmbedding
        ):
            encoded_time = shared_time_features(
                ("g1", self.time_embedding.dim, self.time_embedding.frequency_divisor),
                (time_input, atom_mask),
                encode_time,
            )
        else:
            encoded_time = encode_time()
        return self.route_with_time_embedding(
            h, c, rho, encoded_time, geometry_ok, atom_mask, group_layout=group_layout
        )

    def route_with_time_embedding(
        self,
        h: Tensor,
        c: Tensor,
        rho: Tensor,
        time_embedding: Tensor,
        geometry_ok: Tensor,
        atom_mask: Tensor,
        *,
        group_layout: GroupLayout | None = None,
    ) -> RoutingOutput:
        """Use precomputed time and optionally a trusted layout from AtomCountGroups.

        A supplied layout is valid only for this atom mask and max_groups policy.
        P/Pbar and all H-dependent values are always recomputed.
        """
        keys = self.query_encoder(h, c, rho, time_embedding, geometry_ok, atom_mask)
        if group_layout is None:
            group_layout = self.group_selector.prepare(atom_mask)
        group_mask = group_layout.mask
        if (
            group_mask.ndim != 2
            or group_mask.shape[0] != atom_mask.shape[0]
            or not 1 <= group_mask.shape[1] <= self.max_groups
            or group_mask.device != atom_mask.device
            or group_mask.dtype != torch.bool
            or group_layout.counts.shape != (atom_mask.shape[0],)
            or group_layout.counts.device != atom_mask.device
        ):
            raise ValueError(
                "group_layout must match atom_mask and this router's capacity"
            )
        logits = self.template_bank(keys, num_active=group_mask.shape[1])
        return normalize_assignments(
            logits,
            atom_mask,
            self.pool_eps,
            group_mask=group_mask,
            trusted_layout=True,
        )
