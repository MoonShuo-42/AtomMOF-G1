"""Static condition Transformer: project original c/p before block-local attention."""

from typing import Literal, Mapping, Optional

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from src.models.simple.transformers import Mlp

from .attention import BlockPairAttention
from .features import AtomConditionFeatures
from .pairs import local_pair_layout, prepared_local_pairs

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

    def forward(
        self,
        x: Tensor,
        p: Tensor,
        atom_mask: Tensor,
        allowed_pairs: Tensor,
        pair_indices: Tensor,
        block_layout=None,
    ) -> Tensor:
        """Update atom features using static pair bias and ordinary residuals."""
        x = x + self.attn(self.norm1(x), atom_mask, p, allowed_pairs, pair_indices, block_layout)
        x = x + self.mlp(self.norm2(x))
        return x.masked_fill(~atom_mask.bool().unsqueeze(-1), 0)


class ConditionEncoder(nn.Module):
    """Encode each input block independently into per-atom conditions.

    Builds original AtomMOF-small c/p at 256/128 channels, then projects them
    to hidden_dim=128 / pair_dim=64 before one static four-head Transformer.
    The default output is 128 channels, with no additional output projection.
    Set output_dim=None to retain hidden_dim channels. Raw feature widths and
    internal Transformer widths can be configured independently.

    forward(feats) computes only valid within-block pairs. Pair MLP
    and pair projection remain compact; attention can use dense per-head bias
    or CPU-planned component buckets through the execution policy.
    No noise, lattice, time, positional index embedding, or target is read.
    Parameters are newly initialized, not automatically loaded from small.ckpt.
    """

    _execution_flags = {"packed_block_attention": "packed_block_attention"}

    def __init__(
        self,
        atom_s: int = 256,
        atom_z: int = 128,
        output_dim: Optional[int] = 128,
        depth: int = 1,
        heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_impl: AttentionImpl = "pytorch",
        activation_checkpointing: bool = False,
        *,
        hidden_dim: int = 128,
        pair_dim: int = 64,
    ):
        super().__init__()
        if atom_s <= 0 or atom_z <= 0 or hidden_dim <= 0 or pair_dim <= 0:
            raise ValueError("atom_s, atom_z, hidden_dim and pair_dim must be positive")
        if heads <= 0 or hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by positive heads")
        if depth < 1 or int(hidden_dim * mlp_ratio) < 1:
            raise ValueError("depth and MLP hidden width must be positive")
        if output_dim is not None and output_dim <= 0:
            raise ValueError("output_dim must be positive or None")
        self.atom_s = atom_s
        self.atom_z = atom_z
        self.hidden_dim = hidden_dim
        self.pair_dim = pair_dim
        self.output_dim = hidden_dim if output_dim is None else output_dim
        self.activation_checkpointing = activation_checkpointing
        self.features = AtomConditionFeatures(atom_s, atom_z)
        self.single_projection = (
            nn.Identity() if atom_s == hidden_dim else nn.Linear(atom_s, hidden_dim, bias=False)
        )
        self.pair_projection = (
            nn.Identity() if atom_z == pair_dim else nn.Linear(atom_z, pair_dim, bias=False)
        )
        self.blocks = nn.ModuleList(
            [
                StaticConditionBlock(
                    dim=hidden_dim,
                    dim_pair=pair_dim,
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
            if self.output_dim == hidden_dim
            else nn.Linear(hidden_dim, self.output_dim, bias=False)
        )

    def _encode(
        self,
        c: Tensor,
        p: Tensor,
        atom_mask: Tensor,
        allowed_pairs: Tensor,
        pair_indices: Tensor,
        block_layout=None,
    ) -> Tensor:
        x = self.single_projection(c.masked_fill(~atom_mask.unsqueeze(-1), 0))
        p = self.pair_projection(p)
        for block in self.blocks:
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(
                    block,
                    x,
                    p,
                    atom_mask,
                    allowed_pairs,
                    pair_indices,
                    block_layout,
                    use_reentrant=False,
                )
            else:
                x = block(x, p, atom_mask, allowed_pairs, pair_indices, block_layout)
        return self.output_projection(x).masked_fill(~atom_mask.unsqueeze(-1), 0)

    def forward(self, feats: Mapping[str, Tensor]) -> Tensor:
        """Read only original chemical/reference features; preserve atom ordering."""
        mask = feats["atom_pad_mask"].bool()
        layout = (
            prepared_local_pairs(feats)
            if getattr(self, "prepared_layouts", False)
            else local_pair_layout(mask, feats["block_index"])
        )
        c, p = self.features(feats, layout.indices)
        block_layout = None
        if getattr(self, "packed_block_attention", False):
            block_layout = feats.get("g1_block_layout")
            if block_layout is None:
                from src.data.block_layout import move_block_layout, prepare_block_layout

                block_layout = prepare_block_layout(
                    mask.detach().cpu(),
                    feats["block_index"].detach().cpu(),
                    layout.indices.detach().cpu(),
                )
                block_layout = move_block_layout(block_layout, mask.device)
        return self._encode(c, p, mask, layout.allowed, layout.indices, block_layout)
