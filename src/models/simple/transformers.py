import math
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.simple.attention import AttentionPairBias
from src.models.execution.ops import normalize_pairs, project_pair_bias_group


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#                         Embedding Layers for Timesteps                        #
#################################################################################

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_dim, frequency_embedding_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.frequency_embedding_dim = frequency_embedding_dim

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        # t: Float['b] -> t_emb: Float['b d]
        t_freq = self.timestep_embedding(t, self.frequency_embedding_dim)
        t_emb = self.mlp(t_freq)
        return t_emb
    

class PositionalEmbedder(nn.Module):
    """Embeds integer indices into vector representations using sine/cosine positional encoding."""

    def __init__(self, hidden_dim, frequency_embedding_dim=256, max_len=2048):
        super().__init__()

        if frequency_embedding_dim % 2 != 0:
            raise ValueError(f"frequency_embedding_dim must be even, but got {frequency_embedding_dim}")

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.frequency_embedding_dim = frequency_embedding_dim
        self.max_len = max_len

    @staticmethod
    def _create_positional_embedding(indices, dim, max_len=2048):
        """Creates the core sine/cosine positional embedding."""
        K = torch.arange(dim // 2, device=indices.device)
        div_term = max_len ** (2 * K / dim)
        angles = indices[..., None] * math.pi / div_term
        pos_embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return pos_embedding

    def forward(self, indices):
        # indices: Long['b n'] -> pos_emb: Float['b n d']
        pos_freq = self._create_positional_embedding(
            indices, self.frequency_embedding_dim, self.max_len
        )
        pos_emb = self.mlp(pos_freq)
        return pos_emb
    

class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

AttentionImpl = Literal["manual", "pytorch", "xformers"]


class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(
        self, 
        heads, 
        dim,
        dim_pair=None,
        mlp_ratio=4.0, 
        dropout=0.0,
        attention_impl: AttentionImpl = "pytorch"
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = AttentionPairBias(c_s=dim, c_z=dim_pair, num_heads=heads, attention_impl=attention_impl)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=dropout
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Zero-out adaLN modulation layers in DiT encoder blocks:
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c, mask, z=None, pair_normalized=False, pair_projected=False):
        # Generate modulation parameters (shift, scale gate) from condition c
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c
        ).chunk(6, dim=1)

        # Attention block
        _x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            s=_x, mask=mask, z=z, pair_normalized=pair_normalized, pair_projected=pair_projected
        )

        # MLP block
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
    

class DiT(nn.Module):
    """Transformer with DiT blocks"""

    def __init__(
        self, 
        depth, 
        heads, 
        dim,
        dim_pair=None,
        mlp_ratio=4.0, 
        dropout=0.0,
        attention_impl: AttentionImpl = "pytorch",
        activation_checkpointing=False
    ):
        super().__init__()

        self.activation_checkpointing = activation_checkpointing
        self.pair_norm_reuse = False
        self.pair_bias_chunk_size = 1
        self.t_embedder = TimestepEmbedder(dim)
        self.layers = nn.ModuleList([
            DiTBlock(
                dim=dim,
                heads=heads,
                dim_pair=dim_pair,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_impl=attention_impl
            )
            for _ in range(depth)
        ])

    def _project_pair_chunk(self, z, normalized, start):
        """Batch independent projections sharing the same standardization."""
        groups, projected = {}, {}
        for index in range(start, min(start + self.pair_bias_chunk_size, len(self.layers))):
            attn = self.layers[index].attn
            if attn.use_pair_bias:
                norm = attn.proj_z[0]
                groups.setdefault((norm.normalized_shape, norm.eps), []).append(index)
        for key, indices in groups.items():
            if key not in normalized:
                normalized[key] = normalize_pairs(z, self.layers[indices[0]].attn.proj_z[0])
            projections = [
                (self.layers[index].attn.proj_z[0], self.layers[index].attn.proj_z[1])
                for index in indices
            ]
            biases = project_pair_bias_group(normalized[key], projections)
            projected.update(zip((index + 1 for index in indices), biases))
        return projected

    def forward(self, x, t, mask, z=None, *, after_layer=None):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, D)
            t (torch.Tensor): Time step for each sample (B,)
            mask (torch.Tensor): True if valid token, False if padding (B, N)
        """
        # Embed t
        time_embed = self.t_embedder(t) # (B, D)

        normalized = {}
        projected = {}
        chunk_size = self.pair_bias_chunk_size
        for layer_index, layer in enumerate(self.layers, start=1):
            # z stays fixed across the stack, even when G1 updates x. Only a
            # bounded group of head-sized biases is materialized at a time.
            if (
                self.pair_norm_reuse and z is not None and chunk_size > 1
                and (layer_index - 1) % chunk_size == 0
            ):
                projected = self._project_pair_chunk(z, normalized, layer_index - 1)
            reuse = self.pair_norm_reuse and z is not None and layer.attn.use_pair_bias
            pair = z
            if reuse:
                norm = layer.attn.proj_z[0]
                key = (norm.normalized_shape, norm.eps)
                if key not in normalized:
                    normalized[key] = normalize_pairs(z, norm)
                pair = normalized[key]
            is_projected = layer_index in projected
            if is_projected:
                pair = projected.pop(layer_index)
            if self.activation_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, time_embed, mask, pair, reuse, is_projected, use_reentrant=False
                )
            else:
                x = layer(x, time_embed, mask, pair, reuse, is_projected)
            if after_layer is not None:
                x = after_layer(layer_index, x)
        return x
