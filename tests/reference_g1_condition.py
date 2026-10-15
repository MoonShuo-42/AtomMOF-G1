"""Dense mathematical reference used only by tests and component benchmarks.

Training and prediction never import this module. The dense equations verify
the compact production path; this is not a selectable model architecture.
"""

import torch

from src.models.g1.conditioning.attention import block_attention_mask


def dense_features_reference(features, feats):
    """Evaluate the original c/p equations on all pairs using the supplied weights."""
    ref_pos = feats["ref_pos"]
    atom_mask = feats["atom_pad_mask"].bool()
    block_index = feats["block_index"]
    c = features.single_features(feats)
    d = ref_pos.unsqueeze(1) - ref_pos.unsqueeze(2)
    d_norm = 1 / (1 + torch.sum(d * d, dim=-1, keepdim=True))
    same_block = (
        (
            atom_mask[:, :, None]
            & atom_mask[:, None, :]
            & (block_index[:, :, None] == block_index[:, None, :])
        )
        .float()
        .unsqueeze(-1)
    )
    p = features.embed_atompair_ref_pos(d) * same_block
    p = p + features.embed_atompair_ref_dist(d_norm) * same_block
    p = p + features.embed_atompair_mask(same_block) * same_block
    p = p + features.embed_bond_feats(feats["bond_types"])
    p = p + features.c_to_p_trans_q(c.unsqueeze(2))
    p = p + features.c_to_p_trans_k(c.unsqueeze(1))
    p = p + features.p_mlp(p)
    return c, p


def dense_attention_reference(attn, s, mask, p, allowed):
    """Reference the full per-pair bias with the same attention backend."""
    batch, atoms, _ = s.shape
    s = attn.norm_s(s)
    q = attn.proj_q(s).view(batch, atoms, attn.num_heads, attn.head_dim)
    k = attn.proj_k(s).view(batch, atoms, attn.num_heads, attn.head_dim)
    v = attn.proj_v(s).view(batch, atoms, attn.num_heads, attn.head_dim)
    bias = attn.proj_z(p).masked_fill(~allowed[:, None], float("-inf"))
    output = attn.attention_fn(q, k, v, bias.contiguous()).reshape(batch, atoms, attn.c_s)
    output = attn.proj_o(attn.proj_g(s).sigmoid() * output)
    return output.masked_fill(~mask[..., None], 0)


def dense_condition_reference(encoder, feats):
    """Differentiable oracle, independent of compact layout/scatter and checkpointing."""
    mask = feats["atom_pad_mask"].bool()
    allowed = block_attention_mask(mask, feats["block_index"])
    c, p = dense_features_reference(encoder.features, feats)
    x = encoder.single_projection(c.masked_fill(~mask[..., None], 0))
    p = encoder.pair_projection(p)
    for block in encoder.blocks:
        x = x + dense_attention_reference(block.attn, block.norm1(x), mask, p, allowed)
        x = x + block.mlp(block.norm2(x))
        x = x.masked_fill(~mask[..., None], 0)
    return encoder.output_projection(x).masked_fill(~mask[..., None], 0)
