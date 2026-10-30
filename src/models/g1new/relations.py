"""Per-image distance/direction/chemical encoding, before any aggregation."""

import torch
from torch import nn
from torch.nn import functional as F

from .layers import edge_slices, mlp, recompute, split_apply


class RelationEncoder(nn.Module):
    def __init__(self, cutoff=0.8):
        super().__init__()
        self.cutoff = cutoff
        self.optimized = True
        self.bond_embedding = nn.Embedding(6, 8)
        self.network = mlp(28, 64, 32)
        self.register_buffer("rbf_indices", torch.arange(16), persistent=False)

    def geometry_features(self, edges):
        """Unlearned features shared by all sites, with their geometry graph intact."""
        centers = self.rbf_indices.to(edges.distance.dtype) * (self.cutoff / 15)
        rbf = torch.exp(-((edges.distance[:, None] - centers) / (self.cutoff / 15)).square())
        return torch.cat((rbf, edges.direction, edges.same_block[:, None]), -1)

    def _chunk(self, distance, direction, same_block, bond):
        centers = self.rbf_indices.to(distance.dtype) * (self.cutoff / 15)
        rbf = torch.exp(-((distance[:, None] - centers) / (self.cutoff / 15)).square())
        return self.network(
            torch.cat((rbf, direction, same_block[:, None], self.bond_embedding(bond)), -1)
        )

    def forward(self, edges, chunk_size, checkpointing, geometry_features=None):
        if self.optimized:
            if geometry_features is None:
                geometry_features = self.geometry_features(edges)
            # Project the SIX bond embeddings once, instead of applying that
            # part of the first linear to every edge's 8-channel embedding.
            # The original embedding/MLP parameters and their gradients remain.
            first = self.network[0]
            bond_projection = F.linear(self.bond_embedding.weight, first.weight[:, 20:])

            def encode(geometry, bond):
                hidden = F.linear(geometry, first.weight[:, :20], first.bias)
                hidden = hidden + F.embedding(bond, bond_projection)
                return self.network[2](self.network[1](hidden))

            return split_apply(encode, chunk_size, checkpointing, geometry_features, edges.bond)
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
