import warnings
from typing import Optional, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from einops import rearrange
from einops.layers.torch import Rearrange

import src.models.layers.initialize as init
from src.utils.tensor_typing import Float, Bool

# Check if xformers is available
try:
    from xformers.ops import memory_efficient_attention
    xformers_installed = True
except ImportError:
    xformers_installed = False


class AttentionPairBias(nn.Module):
    """Attention pair bias layer."""

    def __init__(
        self,
        c_s: int,
        c_z: Optional[int],
        num_heads: int,
        inf: float = 1e6,
        initial_norm: bool = True,
        attention_impl: Literal["manual", "pytorch", "xformers"] = "xformers",
    ) -> None:
        """Initialize the attention pair bias layer.

        Parameters
        ----------
        c_s : int
            The input sequence dimension.
        c_z : int, optional
            The input pairwise dimension. If None, the pairwise bias is not used.
        num_heads : int
            The number of heads.
        inf : float, optional
            The inf value, by default 1e6
        initial_norm: bool, optional
            Whether to apply layer norm to the input, by default True
        attention_impl: Literal["manual", "pytorch", "xformers"], optional
            The attention implementation to use, by default "pytorch".
            If "xformers" is chosen but not installed, it falls back to "pytorch".
        """
        super().__init__()

        assert c_s % num_heads == 0

        self.c_s = c_s
        self.num_heads = num_heads
        self.head_dim = c_s // num_heads
        self.inf = inf

        self.initial_norm = initial_norm
        if self.initial_norm:
            self.norm_s = nn.LayerNorm(c_s)
        else:
            self.norm_s = nn.Identity()

        self.proj_q = nn.Linear(c_s, c_s)
        self.proj_k = nn.Linear(c_s, c_s, bias=False)
        self.proj_v = nn.Linear(c_s, c_s, bias=False)
        self.proj_g = nn.Linear(c_s, c_s, bias=False)
        self.proj_o = nn.Linear(c_s, c_s, bias=False)
        # init.final_init_(self.proj_o.weight)

        # Attention implementation
        if attention_impl == "manual":
            self.attention_fn = self._manual_attention
        elif attention_impl == "pytorch":
            self.attention_fn = self._pytorch_attention
        elif attention_impl == "xformers":
            if xformers_installed:
                self.attention_fn = self._xformers_attention
            else:
                warnings.warn("xformers is not installed. Using PyTorch.")
                self.attention_fn = self._pytorch_attention
        else:
            raise ValueError(f"Unknown attention implementation: {attention_impl}")

        # (Optional) pairwise features
        self.use_pair_bias = c_z is not None
        if self.use_pair_bias:
            self.proj_z = nn.Sequential(
                nn.LayerNorm(c_z),
                nn.Linear(c_z, num_heads, bias=False),
                Rearrange("b ... h -> b h ..."),
            )

    def _manual_attention(
        self, 
        q: Float['b n h d'], 
        k: Float['b n h d'], 
        v: Float['b n h d'], 
        attn_mask: Float['b h n n'],
    ):
        attn_scores = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
        attn_scores = attn_scores / (self.head_dim**0.5)
        attn_scores = attn_scores + attn_mask
        
        attn_weights = attn_scores.softmax(dim=-1)
        
        o = torch.einsum("bhij,bjhd->bihd", attn_weights, v.float()).to(v.dtype)
        return o

    def _pytorch_attention(
        self, 
        q: Float['b n h d'], 
        k: Float['b n h d'], 
        v: Float['b n h d'], 
        attn_mask: Float['b h n n'],
    ):
        # Reshape for scaled_dot_product_attention: (b, n, h, d) -> (b, h, n, d)
        q, k, v = map(lambda t: rearrange(t, 'b n h d -> b h n d'), (q, k, v))

        o = F.scaled_dot_product_attention(
            q.float(), k.float(), v.float(), attn_mask=attn_mask.float()
        ).to(v.dtype)
        
        # Transpose back to (b, n, h, d_h)
        return o.transpose(1, 2)

    def _xformers_attention(
        self,
        q: Float['b n h d_h'],
        k: Float['b n h d_h'],
        v: Float['b n h d_h'],
        attn_mask: Float['b h n n'],
    ) -> Tensor:
        # Expects input shape of (b, n, h, d_h) which is already satisfied
        o = memory_efficient_attention(q, k, v, attn_bias=attn_mask)
        return o
    
    def _create_attn_mask(
        self, 
        mask: Bool['b n'], 
        pair_bias: Optional[Float['b n n c_z']] = None
    ) -> Float['b h n n']:

        B, N = mask.shape
        
        # Attention mask broadcasted to (b, 1, 1, n)
        attn_mask = (1 - mask[:, None, None, :].float()) * -self.inf
        attn_mask = attn_mask.expand(B, self.num_heads, N, N)

        # (Optional) Add pair bias
        if self.use_pair_bias:
            assert pair_bias is not None, "pair_bias must be provided if use_pair_bias is True"
            z_bias = self.proj_z(pair_bias) # Shape (b, h, n, n)
            attn_mask = attn_mask + z_bias

        return attn_mask.contiguous() # contiguous for xformers
    
    def forward(
        self,
        s: Tensor,
        mask: Tensor,
        z: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            s: Input sequence tensor (B, N, D).
            mask: Sequence mask tensor (B, N).
            z: Optional pairwise bias tensor (B, N, N, D).

        Returns:
            The output sequence tensor.
        """
        B, N, _ = s.shape

        # Initial normalization and projections (common logic)
        s = self.norm_s(s)
        q = self.proj_q(s).view(B, N, self.num_heads, self.head_dim)
        k = self.proj_k(s).view(B, N, self.num_heads, self.head_dim)
        v = self.proj_v(s).view(B, N, self.num_heads, self.head_dim)
        g = self.proj_g(s).sigmoid()

        # Attention mask
        attn_mask = self._create_attn_mask(mask, z)

        # Attention computation
        o = self.attention_fn(q, k, v, attn_mask)

        # Final gating and projection
        o = o.reshape(B, N, self.c_s)
        o = self.proj_o(g * o)

        return o