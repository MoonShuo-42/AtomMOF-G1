"""Receiver-dependent image messages, with a globally normalized null message."""

import torch
from torch import nn

from .layers import JointMLP, edge_slices, recompute, split_apply


class AtomReadback(nn.Module):
    def __init__(self, width=64, hidden_dim=128):
        super().__init__()
        self.width = width
        self.optimized = True
        self.messages = JointMLP((width, width, 32), hidden_dim, width)
        self.heads = nn.Linear(width, 1 + width)

    def _chunk(self, atom_projection, env_projection, chi, atom, env, omega, previous_max):
        hidden = atom_projection[atom] + env_projection[env]
        hidden = hidden + self.messages.project(chi, 2, bias=True)
        heads = self.heads(self.messages.finish(hidden))
        # Never evaluate log(0); zero-support edges receive exactly zero weight.
        safe_omega = torch.where(omega > 0, omega, 1)
        logits = (heads[:, 0] + safe_omega.log()).masked_fill(omega <= 0, float("-inf"))
        with torch.no_grad():
            current_max = previous_max.scatter_reduce(
                0, atom, logits.detach(), reduce="amax", include_self=True
            )
        weights = (logits - current_max[atom]).exp()
        return weights, weights[:, None] * heads[:, 1:], current_max

    def _pool(self, heads, omega, atom, count):
        safe_omega = torch.where(omega > 0, omega, 1)
        logits = (heads[:, 0] + safe_omega.log()).masked_fill(omega <= 0, float("-inf"))
        with torch.no_grad():
            maximum = heads.new_zeros(count).scatter_reduce(
                0, atom, logits.detach(), reduce="amax", include_self=True
            )
        weights = (logits - maximum.index_select(0, atom)).exp()
        messages = torch.cat((weights[:, None], weights[:, None] * heads[:, 1:]), -1)
        pooled = heads.new_zeros((count, self.width + 1)).index_add(0, atom, messages)
        # A single global log-sum-exp: the null logit is zero, value is zero.
        denominator = (-maximum).exp() + pooled[:, 0]
        return pooled[:, 1:] / denominator[:, None]

    def forward(self, features, state, edges, chi, chunk_size, checkpointing):
        atom_projection = self.messages.project(features, 0)
        env_projection = self.messages.project(state, 1)
        if self.optimized:
            head_map = self.messages.fused_heads(self.heads)
            if checkpointing:
                def readback(relation, atom_projection, env_projection, omega):
                    def edge_heads(part, atom, env):
                        hidden = atom_projection.index_select(0, atom)
                        hidden = hidden + env_projection.index_select(0, env)
                        hidden = hidden + self.messages.project(part, 2, bias=True)
                        return head_map(hidden)

                    heads = split_apply(
                        edge_heads, chunk_size, False, relation, edges.atom, edges.env
                    )
                    return self._pool(heads, omega, edges.atom, features.shape[0])

                # Recompute projection, scores, values AND global normalization;
                # only small node tensors and shared relations remain saved.
                return recompute(
                    readback, True, chi, atom_projection, env_projection, edges.omega
                )
            else:
                hidden = (
                    atom_projection.index_select(0, edges.atom)
                    + env_projection.index_select(0, edges.env)
                    + self.messages.project(chi, 2, bias=True)
                )
                heads = split_apply(head_map, chunk_size, False, hidden)
            return self._pool(heads, edges.omega, edges.atom, features.shape[0])

        # Null logit=0/value=0 exists ONCE per receiver. Online log-sum-exp
        # rescales all previous chunks whenever a larger logit arrives.
        maximum = features.new_zeros(features.shape[0])
        denominator = features.new_ones(features.shape[0])
        numerator = torch.zeros_like(features)
        for part in edge_slices(edges.atom.numel(), chunk_size):
            weights, weighted, current = recompute(
                self._chunk,
                checkpointing,
                atom_projection,
                env_projection,
                chi[part],
                edges.atom[part],
                edges.env[part],
                edges.omega[part],
                maximum,
            )
            rescale = (maximum - current).exp()
            denominator = (denominator * rescale).index_add(0, edges.atom[part], weights)
            numerator = (numerator * rescale[:, None]).index_add(0, edges.atom[part], weighted)
            maximum = current
        return numerator / denominator[:, None]
