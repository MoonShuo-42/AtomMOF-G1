"""Independent edge participation and two shared-weight environment updates."""

import torch
from torch import nn

from .layers import JointMLP, edge_slices, mlp, recompute


class EnvironmentUpdate(nn.Module):
    def __init__(self, width=64, hidden_dim=128):
        super().__init__()
        self.width = width
        self.init = mlp(width, hidden_dim, width)
        self.messages = JointMLP((width, width, width, 32), hidden_dim, width)
        # Independent score/value rows in one GEMM; no parameter sharing between heads.
        self.heads = nn.Linear(width, 1 + width)
        self.update_norm = nn.LayerNorm(2 * width + 1, eps=1e-6)
        self.update = mlp(2 * width + 1, hidden_dim, width)

    def prepare(self, features, anchors):
        atom = self.messages.project(features, 0)
        anchor = self.messages.project(features[anchors], 1)
        return atom, anchor

    def _chunk(self, atom_projection, env_projection, chi, atom, env, omega):
        hidden = atom_projection[atom] + env_projection[env]
        hidden = hidden + self.messages.project(chi, 3, bias=True)
        heads = self.heads(self.messages.finish(hidden))
        p = omega * torch.sigmoid(heads[:, 0])
        return p, p[:, None] * heads[:, 1:]

    def forward(self, state, projected, edges, chi, chunk_size, checkpointing):
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
