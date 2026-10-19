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


class _SharedCast(torch.autograd.Function):
    """Share cast storage, but return each branch's gradient in the source dtype.

    Letting branches use one ordinary BF16 cast node would accumulate their
    gradients in BF16. These separate identity-in-backward nodes preserve the
    original FP32 fan-in while their forward outputs alias the same storage.
    """

    @staticmethod
    def forward(ctx, source: Tensor, converted: Tensor):
        ctx.source_dtype = source.dtype
        return converted

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.to(ctx.source_dtype), None


def pair_projection_copy(z: Tensor, enabled: bool) -> Tensor | None:
    """One detached storage copy; _SharedCast supplies its full gradient path."""
    if not enabled or not torch.is_autocast_enabled(z.device.type) or z.dtype != torch.float32:
        return None
    dtype = torch.get_autocast_dtype(z.device.type)
    if dtype not in (torch.float16, torch.bfloat16):
        return None
    with torch.no_grad():
        return z.to(dtype)


def project_normalized_pairs(
    z: Tensor,
    norm: nn.LayerNorm,
    projection: nn.Linear,
    projection_copy: Tensor | None = None,
) -> Tensor:
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
    if projection_copy is not None:
        z = _SharedCast.apply(z, projection_copy)
    return F.linear(z, effective, bias)


def content_autocast(module: nn.Module, x: Tensor):
    """Opt-in AMP for learned content operations; geometry stays in its own FP32 scope."""
    enabled = (
        getattr(module, "mixed_precision", False)
        and torch.is_autocast_enabled(x.device.type)
        and x.dtype != torch.float64
    )
    return torch.autocast(device_type=x.device.type, enabled=enabled)
