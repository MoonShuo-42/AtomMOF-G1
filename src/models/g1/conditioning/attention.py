"""Same-block attention using AtomMOF's pair bias and feature gate."""

import torch
from torch import Tensor

from src.models.simple.attention import AttentionPairBias


def block_attention_mask(atom_mask: Tensor, block_index: Tensor) -> Tensor:
    """Return allowed query/key pairs (B,N,N), including safe padding rows.

    Valid atoms can read all valid atoms in their own block, including self.
    Each padded query temporarily reads only itself, avoiding an all-masked
    softmax on every backend, even for an entirely padded sample. Its output
    is subsequently zeroed; valid queries can never read a padded key.
    """
    atom_mask = atom_mask.bool()
    if atom_mask.ndim != 2 or block_index.shape != atom_mask.shape:
        raise ValueError("atom_mask and block_index must have the same (B,N) shape")
    if atom_mask.shape[0] == 0 or atom_mask.shape[1] == 0:
        raise ValueError("ConditionEncoder requires nonempty batch and atom dimensions")
    allowed = (
        atom_mask[:, :, None]
        & atom_mask[:, None, :]
        & (block_index[:, :, None] == block_index[:, None, :])
    )
    diagonal = torch.eye(atom_mask.shape[1], device=atom_mask.device, dtype=torch.bool)
    return allowed | (~atom_mask[:, :, None] & diagonal[None])


class BlockPairAttention(AttentionPairBias):
    """Reuse original projections/backends, with a hard query/key block mask.

    The original sigmoid feature gate is retained. It is unrelated to DiT's
    time-conditioned residual gates, which the static encoder does not use.
    """

    def forward(self, s: Tensor, mask: Tensor, z: Tensor, allowed_pairs: Tensor) -> Tensor:
        """Attend within blocks and return exactly zero at padded query rows."""
        batch_size, num_atoms, _ = s.shape
        s = self.norm_s(s)
        q = self.proj_q(s).view(batch_size, num_atoms, self.num_heads, self.head_dim)
        k = self.proj_k(s).view(batch_size, num_atoms, self.num_heads, self.head_dim)
        v = self.proj_v(s).view(batch_size, num_atoms, self.num_heads, self.head_dim)
        gate = self.proj_g(s).sigmoid()

        # Mask AFTER adding pair bias: zeroing z would not mask the QK term.
        attn_bias = self.proj_z(z).masked_fill(~allowed_pairs[:, None], float("-inf"))
        output = self.attention_fn(q, k, v, attn_bias.contiguous())
        output = output.reshape(batch_size, num_atoms, self.c_s)
        output = self.proj_o(gate * output)
        return output.masked_fill(~mask.bool().unsqueeze(-1), 0)
