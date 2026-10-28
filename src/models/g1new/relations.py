"""Per-image distance/direction/chemical encoding, before any aggregation."""

import torch
from torch import nn

from .layers import edge_slices, mlp, recompute


class RelationEncoder(nn.Module):
    def __init__(self, cutoff=0.8):
        super().__init__()
        self.cutoff = cutoff
        self.bond_embedding = nn.Embedding(6, 8)
        self.network = mlp(28, 64, 32)
        self.register_buffer("rbf_indices", torch.arange(16), persistent=False)

    def _chunk(self, distance, direction, same_block, bond):
        centers = self.rbf_indices.to(distance.dtype) * (self.cutoff / 15)
        rbf = torch.exp(-((distance[:, None] - centers) / (self.cutoff / 15)).square())
        return self.network(
            torch.cat((rbf, direction, same_block[:, None], self.bond_embedding(bond)), -1)
        )

    def forward(self, edges, chunk_size, checkpointing):
        return torch.cat(
            [
                recompute(
                    self._chunk,
                    checkpointing,
                    edges.distance[part],
                    edges.direction[part],
                    edges.same_block[part],
                    edges.bond[part],
                )
                for part in edge_slices(edges.atom.numel(), chunk_size)
            ]
        )
