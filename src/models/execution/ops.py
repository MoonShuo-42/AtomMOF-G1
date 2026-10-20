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


def project_normalized_pairs(
    z: Tensor,
    norm: nn.LayerNorm,
    projection: nn.Linear,
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
    return F.linear(z, effective, bias)


def project_pair_bias_group(z: Tensor, layers) -> tuple[Tensor, ...]:
    """One GEMM for several independent LayerNorm-affine/linear projections.

    Only the parameter-free normalized input is shared. Concatenation retains
    every original parameter and its gradient, with no cache across forwards.
    """
    weights, biases, sizes = [], [], []
    with torch.autocast(device_type=z.device.type, enabled=False):
        for norm, projection in layers:
            weight = projection.weight.to(z.dtype)
            weights.append(weight if norm.weight is None else weight * norm.weight.to(z.dtype))
            bias = None if projection.bias is None else projection.bias.to(z.dtype)
            if norm.bias is not None:
                offset = F.linear(norm.bias.to(z.dtype), weight)
                bias = offset if bias is None else bias + offset
            biases.append(bias)
            sizes.append(projection.out_features)
    bias = None
    if any(value is not None for value in biases):
        bias = torch.cat(
            [
                value if value is not None else z.new_zeros(size)
                for value, size in zip(biases, sizes)
            ]
        )
    return F.linear(z, torch.cat(weights), bias).split(sizes, dim=-1)


def project_pair_geometry(d: Tensor, inverse_distance: Tensor, mask: Tensor, layers) -> Tensor:
    """Fuse the three bias-free geometric projections using their existing weights.

    Mask in the five-channel input space rather than materializing three wide
    pair tensors. Bond and single-feature contributions remain unmasked.
    """
    features = torch.cat((d * mask, inverse_distance * mask, mask * mask), dim=-1)
    weight = torch.cat([layer.weight for layer in layers], dim=1)
    return F.linear(features, weight)
