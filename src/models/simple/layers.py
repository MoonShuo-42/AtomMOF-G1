from einops import rearrange

import torch
from torch import nn
from torch.nn import Module, Embedding
from torch.nn.functional import one_hot

import src.models.layers.initialize as init
from src.data import const
from src.models.modules.utils import LinearNoBias
from src.models.simple.transformers import DiT, PositionalEmbedder


class AtomAttentionEncoder(Module):
    def __init__(
        self, 
        atom_s, 
        atom_z,
        atom_feature_dim, 
        atom_encoder_depth, 
        atom_encoder_heads, 
        attention_impl,
        positional_encoding=False,
        activation_checkpointing=False
    ):
        super().__init__()
        
        self.embed_atom_features = LinearNoBias(atom_feature_dim, atom_s)
        self.embed_atompair_ref_pos = LinearNoBias(3, atom_z)
        self.embed_atompair_ref_dist = LinearNoBias(1, atom_z)
        self.embed_atompair_mask = LinearNoBias(1, atom_z)
        self.embed_bond_feats = Embedding(len(const.bond_types), atom_z)

        self.positional_encoding = positional_encoding
        if self.positional_encoding:
            self.pos_emb = PositionalEmbedder(atom_s)

        self.c_to_p_trans_k = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(atom_s, atom_z),
        )
        init.final_init_(self.c_to_p_trans_k[1].weight)

        self.c_to_p_trans_q = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(atom_s, atom_z),
        )
        init.final_init_(self.c_to_p_trans_q[1].weight)

        self.p_mlp = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(atom_z, atom_z),
            nn.ReLU(),
            LinearNoBias(atom_z, atom_z),
            nn.ReLU(),
            LinearNoBias(atom_z, atom_z),
        )
        init.final_init_(self.p_mlp[5].weight)

        self.x_to_q_trans = LinearNoBias(3, atom_s)
        init.final_init_(self.x_to_q_trans.weight)

        self.l_to_q_trans = LinearNoBias(6, atom_s)
        init.final_init_(self.l_to_q_trans.weight)

        self.atom_encoder = DiT(
            dim=atom_s,
            dim_pair=atom_z,
            depth=atom_encoder_depth,
            heads=atom_encoder_heads,
            attention_impl=attention_impl,
            activation_checkpointing=activation_checkpointing,
        )

    def forward(
        self, 
        x_t, 
        l_t, 
        t, 
        feats,
        multiplicity=1
    ):
        B, N, _ = feats["ref_pos"].shape
        atom_mask = feats["atom_pad_mask"].bool()

        # Create atom single conditioning
        atom_ref_pos = feats["ref_pos"]
        atom_uid = feats["block_index"]
        # TODO: Try embedding each feature separately before concatenation (e.g., positional encoding)
        atom_feats = [
            atom_ref_pos,
            feats["ref_charge"].unsqueeze(-1),
            feats["atom_pad_mask"].unsqueeze(-1),
            feats["ref_element"].unsqueeze(-1),
        ]
        
        # Add block type features (# TODO: Experiment with removing this)
        atom_block_feats = one_hot(feats["block_type"], num_classes=len(const.block_types)).float()
        atom_feats.append(atom_block_feats)

        atom_feats = torch.cat(atom_feats, dim=-1)
        c = self.embed_atom_features(atom_feats)

        # Embed offsets between atom reference positions
        atom_ref_pos_queries = atom_ref_pos.unsqueeze(2) # (B, N, 1, 3)
        atom_ref_pos_keys = atom_ref_pos.unsqueeze(1) # (B, 1, N, 3)

        # Pairwise distance vectors
        d = atom_ref_pos_keys - atom_ref_pos_queries # (B, N, N, 3)
        d_norm = torch.sum(d * d, dim=-1, keepdim=True)
        d_norm = 1 / (1 + d_norm)

        # Pairwise mask to zero out interactions between different blocks or with padded atoms
        atom_mask_queries = atom_mask.view(B, N, 1)
        atom_mask_keys = atom_mask.view(B, 1, N)
        atom_uid_queries = atom_uid.view(B, N, 1)
        atom_uid_keys = atom_uid.view(B, 1, N)

        # v[b, i, j, 0] = 1 if
        # (1) atom i is not padded, 
        # (2) atom j is not padded, 
        # (3) atom i and j belong to the same block
        v = (
            (
                atom_mask_queries
                & atom_mask_keys
                & (atom_uid_queries == atom_uid_keys)
            )
            .float()
            .unsqueeze(-1) # (B, N, N, 1)
        )

        # Initial pairwise embedding
        p = self.embed_atompair_ref_pos(d) * v
        p = p + self.embed_atompair_ref_dist(d_norm) * v
        p = p + self.embed_atompair_mask(v) * v

        # Add bond features
        p = p + self.embed_bond_feats(feats["bond_types"])

        # Add combined single conditioning to pair representation
        p = p + self.c_to_p_trans_q(c.view(B, N, 1, c.shape[-1]))
        p = p + self.c_to_p_trans_k(c.view(B, 1, N, c.shape[-1]))
        p = p + self.p_mlp(p)
        
        q = c

        # Multiplicity for different versions
        q = q.repeat_interleave(multiplicity, 0)
        p = p.repeat_interleave(multiplicity, 0)
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        # Add noisy positions
        q = q + self.x_to_q_trans(x_t)

        # Add noisy lattice
        l_embed = self.l_to_q_trans(l_t)
        q = q + rearrange(l_embed, "b d -> b 1 d")

        # Add positional encoding
        if self.positional_encoding:
            pos_emb = self.pos_emb(feats["atom_pos_index"])
            q = q + pos_emb

        # Pass through transformer
        q = self.atom_encoder(q, t, atom_mask, p) # (B, N, atom_s)

        return q, p
    

class AtomAttentionDecoder(Module):
    def __init__(
        self, 
        atom_s, 
        atom_decoder_depth, 
        atom_decoder_heads, 
        attention_impl,
        activation_checkpointing=False
    ):
        super().__init__()
        self.atom_decoder = DiT(
            dim=atom_s,
            depth=atom_decoder_depth,
            heads=atom_decoder_heads,
            attention_impl=attention_impl,
            activation_checkpointing=activation_checkpointing,
        )

        # Projection heads
        self.feats_to_coords = nn.Sequential(
            nn.LayerNorm(atom_s), LinearNoBias(atom_s, 3)
        )
        init.final_init_(self.feats_to_coords[1].weight)

        self.feats_to_lattice = nn.Sequential(
            nn.LayerNorm(atom_s), LinearNoBias(atom_s, 6)
        )
        init.final_init_(self.feats_to_lattice[1].weight)

    def forward(
        self, 
        x, 
        t, 
        feats, 
        multiplicity=1
    ):
        atom_mask = feats["atom_pad_mask"]
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        x = self.atom_decoder(x, t, atom_mask)

        # Position prediction
        r_update = self.feats_to_coords(x)

        # Lattice prediction
        num_atoms = atom_mask.sum(dim=1, keepdim=True) # (B, 1)
        x_global = torch.sum(x * atom_mask[..., None], dim=1) / num_atoms
        l_update = self.feats_to_lattice(x_global)

        return r_update, l_update


class TokenTransformer(Module):
    def __init__(
        self, 
        token_s, 
        token_z,
        token_transformer_depth, 
        token_transformer_heads, 
        attention_impl,
        activation_checkpointing=False
    ):
        super().__init__()

        self.s_init = nn.Linear(token_s, token_s, bias=False)
        self.z_init_1 = nn.Linear(token_s, token_z, bias=False)
        self.z_init_2 = nn.Linear(token_s, token_z, bias=False)

        # Normalization layers
        self.s_mlp = nn.Sequential(
            nn.LayerNorm(token_s), nn.Linear(token_s, token_s, bias=False)
        )
        self.z_mlp = nn.Sequential(
            nn.LayerNorm(token_z), nn.Linear(token_z, token_z, bias=False)
        )

        self.token_transformer = DiT(
            dim=token_s, 
            dim_pair=token_z,
            depth=token_transformer_depth, 
            heads=token_transformer_heads, 
            attention_impl=attention_impl,
            activation_checkpointing=activation_checkpointing
        )

    def forward(
        self,
        x,
        z,
        t,
        feats,
        multiplicity=1
    ):
        token_mask = feats["atom_pad_mask"] # TODO: Rename to token_pad_mask
        token_mask = token_mask.repeat_interleave(multiplicity, 0)

        # Initialize single embeddings
        s_init = self.s_init(x)  # (batch_size, num_tokens, token_s)

        # Initialize pairwise embeddings
        z_init = z + (
            self.z_init_1(s_init)[:, :, None]
            + self.z_init_2(s_init)[:, None, :]
        ) # (batch_size, num_tokens, num_tokens, token_z)

        s = self.s_mlp(s_init)
        z = self.z_mlp(z_init)
        x = self.token_transformer(s, t, token_mask, z)

        return x
