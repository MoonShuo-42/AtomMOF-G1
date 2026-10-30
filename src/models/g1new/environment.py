"""Independent edge participation and two shared-weight environment updates."""

from typing import NamedTuple

import torch
from torch import nn

from .layers import JointMLP, edge_slices, mlp, recompute, split_apply


class PreparedEnvironment(NamedTuple):
    atom: torch.Tensor
    anchor: torch.Tensor
    relation: torch.Tensor


class EnvironmentUpdate(nn.Module):
    def __init__(self, width=64, hidden_dim=128):
        super().__init__()
        self.width = width
        self.optimized = True
        self.init = mlp(width, hidden_dim, width)
        self.messages = JointMLP((width, width, width, 32), hidden_dim, width)
        # Independent score/value rows in one GEMM; no parameter sharing between heads.
        self.heads = nn.Linear(width, 1 + width)
        self.update_norm = nn.LayerNorm(2 * width + 1, eps=1e-6)
        self.update = mlp(2 * width + 1, hidden_dim, width)

    def prepare(
        self, features, anchors, edges=None, chi=None, *, checkpointing=False, anchor_features=None
    ):
        atom = self.messages.project(features, 0)
        if anchor_features is None:
            anchor_features = features.index_select(0, anchors)
        anchor = self.messages.project(anchor_features, 1)
        if self.optimized and edges is not None and not checkpointing:
            # Cache the invariant relation projection, but keep the original
            # atom + (anchor + state) + relation addition order in every round.
            return PreparedEnvironment(atom, anchor, self.messages.project(chi, 3, bias=True))
        return atom, anchor

    def _chunk(self, atom_projection, env_projection, chi, atom, env, omega):
        hidden = atom_projection[atom] + env_projection[env]
        hidden = hidden + self.messages.project(chi, 3, bias=True)
        heads = self.heads(self.messages.finish(hidden))
        p = omega * torch.sigmoid(heads[:, 0])
        return p, p[:, None] * heads[:, 1:]

    def _pool(self, heads, omega, env, count):
        p = omega * torch.sigmoid(heads[:, 0])
        # Keep the reference's scalar/vector reduction shapes. Packing both
        # into T*65 changes CUDA reduction layout and allocates another large
        # edge tensor. Two global reductions still avoid per-chunk scattering.
        mass = heads.new_zeros(count).index_add(0, env, p)
        numerator = heads.new_zeros((count, self.width)).index_add(
            0, env, p[:, None] * heads[:, 1:]
        )
        return mass, numerator

    def forward(self, state, projected, edges, chi, chunk_size, checkpointing, *, head_map=None):
        if self.optimized:
            if head_map is None:
                head_map = self.messages.head_map(self.heads)
            if isinstance(projected, PreparedEnvironment):
                env_projection = projected.anchor + self.messages.project(state, 2)
                hidden = projected.atom.index_select(0, edges.atom)
                hidden = hidden + env_projection.index_select(0, edges.env)
                hidden = hidden + projected.relation
                heads = split_apply(head_map, chunk_size, checkpointing, hidden)
                pooled = self._pool(heads, edges.omega, edges.env, state.shape[0])
            else:
                atom_projection, anchor_projection = projected
                env_projection = anchor_projection + self.messages.project(state, 2)

                def pool(relation, atom_projection, env_projection, omega):
                    def edge_heads(part, atom, env):
                        hidden = atom_projection.index_select(0, atom)
                        hidden = hidden + env_projection.index_select(0, env)
                        hidden = hidden + self.messages.project(part, 3, bias=True)
                        return head_map(hidden)

                    heads = split_apply(
                        edge_heads, chunk_size, False, relation, edges.atom, edges.env
                    )
                    return self._pool(heads, omega, edges.env, state.shape[0])

                # Checkpoint through the weighted aggregation, returning M + M*64,
                # not T*65 heads/values or T*128 projected hidden activations.
                pooled = recompute(
                    pool, checkpointing, chi, atom_projection, env_projection, edges.omega
                )
            # Global scalar/vector reductions, one null item per environment.
            mass, numerator = pooled
            u = numerator / (1 + mass[:, None])
            q = torch.cat((state, u, torch.log1p(mass)[:, None]), -1)
            delta = self.update(self.update_norm(q))
            return state + (mass / (1 + mass))[:, None] * delta

        atom_projection, anchor_projection = projected
        env_projection = anchor_projection + self.messages.project(state, 2)
        mass = state.new_zeros(state.shape[0])
        numerator = torch.zeros_like(state)
        for part in edge_slices(edges.atom.numel(), chunk_size):
            p, weighted = recompute(
                self._chunk,
                checkpointing,
                atom_projection,
                env_projection,
                chi[part],
                edges.atom[part],
                edges.env[part],
                edges.omega[part],
            )
            mass = mass.index_add(0, edges.env[part], p)
            numerator = numerator.index_add(0, edges.env[part], weighted)
        # One null message per ENVIRONMENT, not per chunk.
        u = numerator / (1 + mass[:, None])
        q = torch.cat((state, u, torch.log1p(mass)[:, None]), -1)
        delta = self.update(self.update_norm(q))
        state = state + (mass / (1 + mass))[:, None] * delta
        return state
