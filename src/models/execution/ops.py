"""Differentiable algebraic rewrites that preserve parameter names and shapes."""
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def repeat_batch(x: Tensor, multiplicity: int, *, enabled: bool = False) -> Tensor:
    # Returning the input is safe: callers use out-of-place updates.
    return x if enabled and multiplicity == 1 else x.repeat_interleave(multiplicity, 0)


def normalize_pairs(z: Tensor, norm: nn.LayerNorm) -> Tensor:
    """Only the parameter-free part is shared; each layer keeps its affine weights."""
    with torch.autocast(device_type=z.device.type, enabled=False):
        value = z.float() if z.dtype in (torch.float16, torch.bfloat16) else z
        return F.layer_norm(value, norm.normalized_shape, eps=norm.eps)


def project_normalized_pairs(z: Tensor, norm: nn.LayerNorm, projection: nn.Linear) -> Tensor:
    """W(gamma*z_hat+beta)+b, without materializing gamma*z_hat+beta.

    Effective weights are constructed afresh and remain connected to autograd.
    Never cache them across optimizer updates or merge the learned parameters.
    """
    with torch.autocast(device_type=z.device.type, enabled=False):
        weight = projection.weight.to(z.dtype)
        effective = weight if norm.weight is None else weight * norm.weight.to(z.dtype)[None]
        bias = None if projection.bias is None else projection.bias.to(z.dtype)
        if norm.bias is not None:
            offset = F.linear(norm.bias.to(z.dtype), weight)
            bias = offset if bias is None else bias + offset
    return F.linear(z, effective, bias)


def content_autocast(module: nn.Module, x: Tensor):
    """Opt-in AMP for learned content operations; geometry stays in its own FP32 scope."""
    enabled = (
        getattr(module, "mixed_precision", False)
        and torch.is_autocast_enabled(x.device.type)
        and x.dtype != torch.float64
    )
    return torch.autocast(device_type=x.device.type, enabled=enabled)
