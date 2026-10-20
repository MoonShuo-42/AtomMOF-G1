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

    def forward(
        self,
        s: Tensor,
        mask: Tensor,
        z: Tensor,
        allowed_pairs: Tensor,
        pair_indices: Tensor,
        block_layout=None,
    ) -> Tensor:
        """Attend within blocks and return exactly zero at padded query rows."""
        batch_size, num_atoms, _ = s.shape
        s = self.norm_s(s)
        q, k, v, gate = self.project_qkvg(s)

        # Mask AFTER adding pair bias: zeroing z would not mask the QK term.
        # LayerNorm/projection are pointwise over pairs. Scatter only H channels,
        # never a dense (B,N,N,64/128) tensor. Padding diagonal bias can be zero:
        # a padded query reads only itself and its output is discarded.
        compact_bias = self.proj_z(z)  # (num_local_pairs, heads)
        if block_layout is not None:
            output = self._packed_attention(q, k, v, compact_bias, block_layout)
            output = self.proj_o(gate * output)
            return output.masked_fill(~mask.bool().unsqueeze(-1), 0)
        flat_bias = compact_bias.new_zeros(
            (batch_size * num_atoms * num_atoms, self.num_heads)
        )
        bias = flat_bias.index_copy(0, pair_indices, compact_bias)
        bias = bias.view(batch_size, num_atoms, num_atoms, self.num_heads).permute(
            0, 3, 1, 2
        )
        attn_bias = bias.masked_fill(~allowed_pairs[:, None], float("-inf"))
        output = self.attention_fn(q, k, v, attn_bias.contiguous())
        output = output.reshape(batch_size, num_atoms, self.c_s)
        output = self.proj_o(gate * output)
        return output.masked_fill(~mask.bool().unsqueeze(-1), 0)

    def _packed_attention(self, q, k, v, compact_bias, layout):
        """One attention call per length bucket, never one call per component."""
        batch, atoms, heads, dim = q.shape
        q_flat, k_flat, v_flat = (
            x.reshape(batch * atoms, heads, dim) for x in (q, k, v)
        )
        rows, destinations = [], []
        for bucket in layout:
            indices, valid, pair_positions, pair_indices, output_positions = bucket
            count, length = indices.shape
            packed_q, packed_k, packed_v = (
                x[indices] for x in (q_flat, k_flat, v_flat)
            )
            bias = compact_bias.new_zeros((count * length * length, heads))
            bias = bias.index_copy(0, pair_positions, compact_bias[pair_indices])
            bias = bias.reshape(count, length, length, heads).permute(0, 3, 1, 2)
            # Every component has a real key. Padded queries are discarded later.
            bias = bias.masked_fill(~valid[:, None, None, :], float("-inf"))
            values = self.attention_fn(packed_q, packed_k, packed_v, bias.contiguous())
            rows.append(values.reshape(-1, self.c_s)[output_positions])
            destinations.append(indices.flatten()[output_positions])
        output = q.new_zeros((batch * atoms, self.c_s))
        if rows:
            output = output.index_copy(0, torch.cat(destinations), torch.cat(rows))
        else:
            # All-padding batches still connect every projection for DDP.
            output = output + (q.sum() + k.sum() + v.sum() + compact_bias.sum()) * 0
        return output.reshape(batch, atoms, self.c_s)
