"""Validate component compatibility and explicit context boundaries."""

import torch
from torch import Tensor, nn

from .outputs import G1Context
from .routing.groups import GroupLayout


def require_module(name: str, module: nn.Module | None) -> None:
    if module is not None and not isinstance(module, nn.Module):
        raise TypeError(f"{name} must be an nn.Module instance or None; instantiate configs first")


def validate_components(condition_encoder, router, group_interaction, readback) -> None:
    """Component replacements must preserve these public interfaces/dimension attributes."""
    query = router.query_encoder
    pool = group_interaction.content_pool
    attention = readback.attention
    checks = (
        (condition_encoder.output_dim, query.c_dim, "condition output / router c_dim"),
        (query.rho_dim, 11, "router rho_dim / periodic geometry rho"),
        (query.h_dim, pool.h_dim, "router / group interaction h_dim"),
        (query.h_dim, attention.h_dim, "router / readback h_dim"),
        (
            pool.value_projection.out_features,
            attention.group_dim,
            "S_plus / readback group_dim",
        ),
        (router.pool_eps, attention.pool_eps, "router / readback pool_eps"),
    )
    for source, target, description in checks:
        if source != target:
            raise ValueError(f"incompatible {description}: {source} != {target}")


def validate_mask(atom_mask: Tensor) -> Tensor:
    if not isinstance(atom_mask, Tensor) or atom_mask.ndim != 2 or min(atom_mask.shape) < 1:
        raise ValueError("atom_pad_mask must be nonempty (B,N)")
    return atom_mask.bool()


def validate_condition(c_cond: Tensor, atom_mask: Tensor, c_dim: int) -> None:
    if c_cond.shape != (*atom_mask.shape, c_dim):
        raise ValueError(f"c_cond must be (B,N,{c_dim}) matching atom_mask")
    if not c_cond.is_floating_point():
        raise TypeError("c_cond must be floating-point")
    if c_cond.device != atom_mask.device:
        raise ValueError("c_cond and atom_mask must share a device")


def validate_context(H: Tensor, t: Tensor, context: G1Context, h_dim: int, c_dim: int) -> None:
    if not isinstance(context, G1Context):
        raise TypeError(
            "context must be a G1Context from prepare or an explicit compatible context"
        )
    if not isinstance(context.group_layout, GroupLayout):
        raise TypeError("context.group_layout must be a prepared GroupLayout")
    mask = validate_mask(context.atom_mask)
    batch, atoms = mask.shape
    if H.shape != (batch, atoms, h_dim):
        raise ValueError(f"H must be (B,N,{h_dim}) matching the context")
    if not H.is_floating_point() or not t.is_floating_point():
        raise TypeError("H and t must be floating-point")
    if t.shape not in ((batch,), (batch, 1)):
        raise ValueError("t must be (B,) or (B,1)")
    if H.device != mask.device or t.device != H.device:
        raise ValueError("H, t and context must share a device")
    validate_condition(context.c_cond, mask, c_dim)
    E, rho, geometry_ok = context.geometry
    if E.shape != (batch, atoms, atoms, 16) or rho.shape != (batch, atoms, 11):
        raise ValueError("context geometry E/rho must match (B,N,N,16)/(B,N,11)")
    if geometry_ok.shape != (batch,) or geometry_ok.dtype != torch.bool:
        raise ValueError("context geometry_ok must be bool (B,)")
    if not E.is_floating_point() or not rho.is_floating_point():
        raise TypeError("context E and rho must be floating-point")
    if any(x.device != H.device for x in (E, rho, geometry_ok)):
        raise ValueError("context geometry and H must share a device")
