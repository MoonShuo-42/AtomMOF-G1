"""Static condition Transformer: original c/p, block mask, no time modulation."""

from typing import Literal, Mapping, Optional

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from src.models.simple.transformers import Mlp

from .attention import BlockPairAttention, block_attention_mask
from .features import AtomConditionFeatures

AttentionImpl = Literal["manual", "pytorch", "xformers"]


class StaticConditionBlock(nn.Module):
    """DiT's attention/MLP residual branches with time modulation removed.

    Retains original LayerNorm settings and the normalization/feature gate
    inside AttentionPairBias. Neither residual branch has a time gate.
    Pair features are read-only across the stack.
    """

    def __init__(
        self,
        dim: int,
        dim_pair: int,
        heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_impl: AttentionImpl = "pytorch",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = BlockPairAttention(
            c_s=dim, c_z=dim_pair, num_heads=heads, attention_impl=attention_impl
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=dropout,
        )
        # Match DiT's ordinary linear initialization, without adaLN-Zero.
        self.apply(self._initialize_linear)

    @staticmethod
    def _initialize_linear(module: nn.Module) -> None:
        """Initialize active residual branches rather than zeroing their gates."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor, p: Tensor, atom_mask: Tensor, allowed_pairs: Tensor) -> Tensor:
        """Update atom features using static pair bias and ordinary residuals."""
        x = x + self.attn(self.norm1(x), atom_mask, p, allowed_pairs)
        x = x + self.mlp(self.norm2(x))
        return x.masked_fill(~atom_mask.bool().unsqueeze(-1), 0)


class ConditionEncoder(nn.Module):
    """Encode each input block independently into per-atom conditions.

    Defaults preserve AtomMOF-small's 256/128 single/pair widths, three encoder
    layers and four heads, followed by a 256 -> 128 projection for G1. Set
    output_dim=None to keep atom_s channels without an output projection.

    forward(feats) builds c/p; encode(c, p, mask, block_index) also accepts
    externally computed original c/p, enabling later feature/parameter reuse.
    No noise, lattice, time, positional index embedding, or target is read.
    Parameters are newly initialized, not automatically loaded from small.ckpt.
    """

    def __init__(
        self,
        atom_s: int = 256,
        atom_z: int = 128,
        output_dim: Optional[int] = 128,
        depth: int = 3,
        heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_impl: AttentionImpl = "pytorch",
        activation_checkpointing: bool = False,
    ):
        super().__init__()
        if atom_s <= 0 or atom_z <= 0 or heads <= 0 or atom_s % heads != 0:
            raise ValueError("positive atom_s must be divisible by positive heads; atom_z > 0")
        if depth < 1 or int(atom_s * mlp_ratio) < 1:
            raise ValueError("depth and MLP hidden width must be positive")
        if output_dim is not None and output_dim <= 0:
            raise ValueError("output_dim must be positive or None")
        self.atom_s = atom_s
        self.atom_z = atom_z
        self.output_dim = atom_s if output_dim is None else output_dim
        self.activation_checkpointing = activation_checkpointing
        self.features = AtomConditionFeatures(atom_s, atom_z)
        self.blocks = nn.ModuleList(
            [
                StaticConditionBlock(
                    dim=atom_s,
                    dim_pair=atom_z,
                    heads=heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_impl=attention_impl,
                )
                for _ in range(depth)
            ]
        )
        self.output_projection = (
            nn.Identity()
            if self.output_dim == atom_s
            else nn.Linear(atom_s, self.output_dim, bias=False)
        )

    def encode(self, c: Tensor, p: Tensor, atom_mask: Tensor, block_index: Tensor) -> Tensor:
        """Apply static blocks to c/p and return (B,N,output_dim) conditions."""
        atom_mask = atom_mask.bool()
        allowed_pairs = block_attention_mask(atom_mask, block_index)
        x = c.masked_fill(~atom_mask.unsqueeze(-1), 0)
        for block in self.blocks:
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, p, atom_mask, allowed_pairs, use_reentrant=False)
            else:
                x = block(x, p, atom_mask, allowed_pairs)
        return self.output_projection(x).masked_fill(~atom_mask.unsqueeze(-1), 0)

    def forward(self, feats: Mapping[str, Tensor]) -> Tensor:
        """Read only original chemical/reference features; preserve atom ordering."""
        c, p = self.features(feats)
        return self.encode(c, p, feats["atom_pad_mask"], feats["block_index"])
