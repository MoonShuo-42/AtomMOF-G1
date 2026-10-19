"""Fuse current hidden, conditional, geometric and time features into router keys."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RouterQueryEncoder(nn.Module):
    """Shared MLP + LayerNorm: [H,c,rho,time,g_ok] -> per-atom routing keys."""

    def __init__(
        self,
        h_dim: int = 256,
        c_dim: int = 128,
        rho_dim: int = 11,
        time_dim: int = 32,
        hidden_dim: int = 128,
        key_dim: int = 32,
    ):
        super().__init__()
        dimensions = (h_dim, c_dim, rho_dim, time_dim, hidden_dim, key_dim)
        if any(not isinstance(dim, int) or dim < 1 for dim in dimensions):
            raise ValueError("router feature dimensions must be positive integers")
        self.h_dim, self.c_dim, self.rho_dim, self.time_dim = h_dim, c_dim, rho_dim, time_dim
        self.input_dim = h_dim + c_dim + rho_dim + time_dim + 1
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, key_dim)
        self.norm = nn.LayerNorm(key_dim)

    def build_inputs(
        self,
        h: Tensor,
        c: Tensor,
        rho: Tensor,
        time_embedding: Tensor,
        geometry_ok: Tensor,
        atom_mask: Tensor,
    ) -> Tensor:
        """Construct (B,N,input_dim), removing padding and unavailable geometry first."""
        if h.ndim != 3 or h.shape[-1] != self.h_dim:
            raise ValueError(f"h must have shape (B,N,{self.h_dim})")
        batch, atoms = h.shape[:2]
        if batch < 1 or atoms < 1 or atom_mask.shape != (batch, atoms):
            raise ValueError("atom_mask must match nonempty h batch/atom dimensions")
        if c.shape != (batch, atoms, self.c_dim) or rho.shape != (batch, atoms, self.rho_dim):
            raise ValueError("c/rho must match h batch/atom dimensions and configured widths")
        if time_embedding.shape != (batch, self.time_dim):
            raise ValueError(f"time_embedding must have shape (B,{self.time_dim})")
        if geometry_ok.shape == (batch, 1):
            geometry_ok = geometry_ok[:, 0]
        if geometry_ok.shape != (batch,):
            raise ValueError("geometry_ok must have shape (B,) or (B,1)")
        tensors = (c, rho, time_embedding, geometry_ok, atom_mask)
        if (
            any(tensor.device != h.device for tensor in tensors)
            or self.fc1.weight.device != h.device
        ):
            raise ValueError("router inputs and parameters must share a device")
        if not all(tensor.is_floating_point() for tensor in (h, c, rho, time_embedding)):
            raise TypeError("h, c, rho and time_embedding must be floating-point tensors")
        dtype = (
            torch.float64
            if torch.float64
            in (h.dtype, c.dtype, rho.dtype, time_embedding.dtype, self.fc1.weight.dtype)
            else torch.float32
        )
        mask = atom_mask.bool()[..., None]
        available = geometry_ok.bool()[:, None, None]
        # torch.where, not multiplication: masked NaN/Inf must not contaminate the MLP.
        h = torch.where(mask, h.to(dtype), 0)
        c = torch.where(mask, c.to(dtype), 0)
        rho = torch.where(mask & available, rho.to(dtype), 0)
        time = torch.where(mask, time_embedding.to(dtype)[:, None, :], 0)
        flag = (mask & available).to(dtype)
        return torch.cat((h, c, rho, time, flag), -1)

    def forward(
        self,
        h: Tensor,
        c: Tensor,
        rho: Tensor,
        time_embedding: Tensor,
        geometry_ok: Tensor,
        atom_mask: Tensor,
    ) -> Tensor:
        """Return (B,N,key_dim); all padding keys and their input gradients are zero."""
        features = self.build_inputs(h, c, rho, time_embedding, geometry_ok, atom_mask)
        # rho contains high-order contractions. Keep this small MLP in >=FP32,
        # even under AMP or when the enclosing model's parameters use half precision.
        dtype = features.dtype
        with torch.autocast(device_type=h.device.type, enabled=False):
            hidden = F.silu(F.linear(features, self.fc1.weight.to(dtype), self.fc1.bias.to(dtype)))
            keys = F.linear(hidden, self.fc2.weight.to(dtype), self.fc2.bias.to(dtype))
            keys = F.layer_norm(
                keys,
                self.norm.normalized_shape,
                self.norm.weight.to(dtype),
                self.norm.bias.to(dtype),
                self.norm.eps,
            )
        return keys.masked_fill(~atom_mask.bool()[..., None], 0)
