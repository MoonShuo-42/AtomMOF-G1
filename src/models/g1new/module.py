"""G1-New v0: one local-environment residual, width 64, two shared rounds."""

from collections.abc import Mapping

import torch
from torch import nn

from .conditioning import ConditionEncoder
from .environment import EnvironmentUpdate
from .geometry import LocalGeometry
from .layers import mlp
from .readback import AtomReadback
from .relations import RelationEncoder
from .time import FourierTimeEmbedding


class G1NewModule(nn.Module):
    def __init__(
        self,
        h_dim=256,
        width=64,
        hidden_dim=128,
        rounds=2,
        edge_chunk_size=65536,
        activation_checkpointing=True,
        optimized=True,
        condition_encoder=None,
        geometry=None,
        inplace_edge_sum=False,
        checkpoint_relation_inputs=False,
        stream_edge_pooling=False,
    ):
        super().__init__()
        for name, value in dict(
            h_dim=h_dim,
            width=width,
            hidden_dim=hidden_dim,
            rounds=rounds,
            edge_chunk_size=edge_chunk_size,
        ).items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(activation_checkpointing) is not bool:
            raise TypeError("g1new.activation_checkpointing must be bool")
        if type(optimized) is not bool:
            raise TypeError("g1new.optimized must be bool")
        for name, value in dict(
            inplace_edge_sum=inplace_edge_sum,
            checkpoint_relation_inputs=checkpoint_relation_inputs,
            stream_edge_pooling=stream_edge_pooling,
        ).items():
            if type(value) is not bool:
                raise TypeError(f"g1new.{name} must be bool")
            setattr(self, name, value)
        self.optimized = optimized
        self.h_dim, self.width, self.rounds = h_dim, width, rounds
        self.edge_chunk_size = edge_chunk_size
        self.activation_checkpointing = activation_checkpointing
        self.condition_encoder = (
            condition_encoder if condition_encoder is not None else ConditionEncoder()
        )
        self.geometry = geometry if geometry is not None else LocalGeometry()
        if not isinstance(self.condition_encoder, ConditionEncoder) or not isinstance(
            self.geometry, LocalGeometry
        ):
            raise TypeError("g1new requires its static ConditionEncoder and LocalGeometry")
        self.atom_norm = nn.LayerNorm(h_dim, eps=1e-6)
        self.atom_projection = nn.Linear(h_dim, width)
        self.condition_projection = nn.Linear(self.condition_encoder.output_dim, 32)
        self.time_embedding = FourierTimeEmbedding(32)
        self.atom_features = mlp(width + 64, hidden_dim, width)
        self.environment = EnvironmentUpdate(width, hidden_dim)
        self.relations = RelationEncoder(self.geometry.cutoff)
        self.readback = AtomReadback(width, hidden_dim)
        for component in (self.geometry, self.relations, self.environment, self.readback):
            component.optimized = optimized
        self.relations.checkpoint_relation_inputs = checkpoint_relation_inputs
        for component in (self.environment, self.readback):
            component.inplace_edge_sum = inplace_edge_sum
            component.stream_edge_pooling = stream_edge_pooling
        self.output_projection = nn.Linear(width, h_dim, bias=False)
        nn.init.zeros_(self.output_projection.weight)

    def encode_condition(self, feats: Mapping):
        return self.condition_encoder(feats)

    def forward(self, H, t, feats, edges, condition, *, time_features=None, relation_features=None):
        mask = feats["atom_pad_mask"].bool()
        if H.shape != (*mask.shape, self.h_dim) or t.shape != (H.shape[0],):
            raise ValueError("g1new expects H=(B,N,h_dim) and t=(B,)")
        if H.dtype not in (torch.float32, torch.float64):
            raise ValueError("g1new keeps FP32 precision; FP64 is supported for numerical tests")
        with torch.autocast(device_type=H.device.type, enabled=False):
            # Geometry-invalid rows cannot affect supported rows. Sanitize branch
            # inputs before nonlinear operations; the backbone H is not modified.
            active = mask & edges.geometry_ok[:, None]
            safe_H = H.masked_fill(~active[..., None], 0)
            c = condition.masked_fill(~active[..., None], 0)
            a = self.atom_projection(self.atom_norm(safe_H))
            c = self.condition_projection(c)
            if time_features is None:
                time_features = self.time_embedding(t)
            if time_features.shape != (H.shape[0], self.time_embedding.dim):
                raise ValueError("g1new time_features must match the expanded batch")
            time = time_features[:, None].expand(-1, H.shape[1], -1)
            features = self.atom_features(torch.cat((a, c, time), -1))
            features = features.masked_fill(~active[..., None], 0).flatten(0, 1)
            checkpointing = (
                self.activation_checkpointing and self.training and torch.is_grad_enabled()
            )
            chi = self.relations(
                edges, self.edge_chunk_size, checkpointing, geometry_features=relation_features
            )
            anchor_features = features.index_select(0, edges.anchors)
            state = self.environment.init(anchor_features)
            projected = self.environment.prepare(
                features, edges.anchors, edges, chi,
                checkpointing=checkpointing, anchor_features=anchor_features,
            )
            # Reuse a callable with the reference's affine operation order.
            # Do not pre-multiply learned weights under the TF32 policy.
            head_map = (
                self.environment.messages.head_map(self.environment.heads)
                if self.optimized else None
            )
            for _ in range(self.rounds):
                state = self.environment(
                    state, projected, edges, chi, self.edge_chunk_size, checkpointing,
                    head_map=head_map,
                )
            # The large static edge projection is not saved by addition's
            # backward. Drop the Python reference before readback's workspace.
            del projected, head_map
            result = self.readback(
                features, state, edges, chi, self.edge_chunk_size, checkpointing
            )
            delta = self.output_projection(result).reshape_as(H)
            return (H + delta).masked_fill(~mask[..., None], 0)
