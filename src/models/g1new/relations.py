"""Per-image distance/direction/chemical encoding, before any aggregation."""

import torch
from torch import nn

from .layers import edge_slices, mlp, recompute, split_apply


class RelationEncoder(nn.Module):
    def __init__(self, cutoff=0.8):
        super().__init__()
        self.cutoff = cutoff
        self.optimized = True
        self.checkpoint_relation_inputs = False
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
            if checkpointing and self.checkpoint_relation_inputs:
                # Save shared geometry and integer bond indices, not a separate
                # full T*28 concatenation at every insertion site. Recreate the
                # same joint Linear input within each checkpointed chunk.
                def encode(geometry, bond):
                    inputs = torch.cat((geometry, self.bond_embedding(bond)), -1)
                    return self.network(inputs)

                return split_apply(
                    encode, chunk_size, True, geometry_features, edges.bond
                )
            # Keep the reference's joint 28-channel linear reduction. Splitting
            # it into geometry/bond GEMMs changes FP32/TF32 rounding points.
            inputs = torch.cat((geometry_features, self.bond_embedding(edges.bond)), -1)
            return split_apply(self.network, chunk_size, checkpointing, inputs)
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
