"""Opt-in, first-order edge pooling with recomputed message workspaces.

Linear/SiLU/Linear/heads and scalar/vector reductions retain their separate
operations. Chunking changes reduction accumulation order, so this path has its
own numerical gate and is deliberately disabled by default. No full edge heads
or weighted values are saved: backward recomputes one chunk at a time.
"""

import torch
from torch.nn import functional as F


def edge_sum(atom_projection, env_projection, relation, atom, env, inplace=False):
    """Only mutate a fresh gather result, never an input or shared projection."""
    hidden = atom_projection.index_select(0, atom)
    if inplace:
        hidden.add_(env_projection.index_select(0, env))
        hidden.add_(relation)
        return hidden
    hidden = hidden + env_projection.index_select(0, env)
    return hidden + relation


def _heads(atom, env, relation, parameters, inplace):
    rw, rb, lw, lb, hw, hb = parameters
    return _heads_projected(atom, env, F.linear(relation, rw, rb), (lw, lb, hw, hb), inplace)


def _heads_projected(atom, env, projected, parameters, inplace):
    lw, lb, hw, hb = parameters
    hidden = atom + env
    if inplace:
        hidden.add_(projected)
    else:
        hidden = hidden + projected
    return F.linear(F.linear(F.silu(hidden), lw, lb), hw, hb)


def _logits(heads, omega):
    safe = torch.where(omega > 0, omega, 1)
    return (heads[:, 0] + safe.log()).masked_fill(omega <= 0, float("-inf"))


class _StreamPool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mode, chunk_size, inplace, *tensors):
        atom_projection, env_projection, chi, omega, atom, env, *parameters = tensors
        count = env_projection.shape[0] if mode == "environment" else atom_projection.shape[0]
        width = parameters[-1].numel() - 1
        rows = atom.numel()

        def chunks():
            for start in range(0, max(rows, 1), chunk_size):
                part = slice(start, start + chunk_size)
                a, e = atom[part], env[part]
                heads = _heads(
                    atom_projection.index_select(0, a), env_projection.index_select(0, e),
                    chi[part], parameters, inplace,
                )
                yield part, a, e, heads

        maximum = atom_projection.new_zeros(count)
        if mode == "readback":
            # A detached global maximum, with the null logit 0 present once.
            for part, a, _, heads in chunks():
                maximum.scatter_reduce_(0, a, _logits(heads, omega[part]),
                                        reduce="amax", include_self=True)
            denominator = (-maximum).exp()
        else:
            denominator = atom_projection.new_zeros(count)
        numerator = atom_projection.new_zeros((count, width))
        for part, a, e, heads in chunks():
            if mode == "environment":
                index = e
                weights = omega[part] * torch.sigmoid(heads[:, 0])
            else:
                index = a
                weights = (_logits(heads, omega[part]) - maximum.index_select(0, a)).exp()
            denominator.index_add_(0, index, weights)
            numerator.index_add_(0, index, weights[:, None] * heads[:, 1:])
        output = numerator / denominator[:, None] if mode == "readback" else numerator
        ctx.mode, ctx.chunk_size, ctx.inplace = mode, chunk_size, inplace
        ctx.save_for_backward(*tensors, maximum, denominator, numerator)
        return (output, denominator) if mode == "readback" else (denominator, output)

    @staticmethod
    def backward(ctx, grad_first, grad_second):
        if torch.is_grad_enabled():
            raise RuntimeError("G1new stream_edge_pooling supports first-order gradients only")
        saved = ctx.saved_tensors
        tensors, maximum, denominator, numerator = saved[:-3], *saved[-3:]
        atom_projection, env_projection, chi, omega, atom, env, *parameters = tensors
        needs = ctx.needs_input_grad[3:]
        gradients = [torch.zeros_like(t) if need else None for t, need in zip(tensors, needs)]
        if ctx.mode == "environment":
            grad_mass = torch.zeros_like(denominator) if grad_first is None else grad_first
            grad_numerator = torch.zeros_like(numerator) if grad_second is None else grad_second
        else:
            grad_output = torch.zeros_like(numerator) if grad_first is None else grad_first
            # Replay the original node-level division backward. Moving division
            # outside the feature reduction is algebraically equivalent but
            # changes FP32 rounding and failed the four-site CUDA gradient gate.
            with torch.enable_grad():
                node_numerator = numerator.detach().requires_grad_()
                node_denominator = denominator.detach().requires_grad_()
                grad_numerator, grad_mass = torch.autograd.grad(
                    node_numerator / node_denominator[:, None],
                    (node_numerator, node_denominator), grad_output,
                )
            # The second return is private to the wrapper, but account for it
            # if a direct internal caller differentiates the denominator.
            if grad_second is not None:
                grad_mass = grad_mass + grad_second
        for start in range(0, max(atom.numel(), 1), ctx.chunk_size):
            part = slice(start, start + ctx.chunk_size)
            a, e = atom[part], env[part]
            local = [
                atom_projection.index_select(0, a).detach(),
                env_projection.index_select(0, e).detach(),
                chi[part].detach(), omega[part].detach(),
                *[p.detach() for p in parameters],
            ]
            positions = [0, 1, 2, 3, *range(6, 12)]
            targets, target_positions = [], []
            for value, pos in zip(local, positions):
                if needs[pos]:
                    value.requires_grad_()
                    targets.append(value)
                    target_positions.append(pos)
            if not targets:
                continue
            with torch.enable_grad():
                heads = _heads(local[0], local[1], local[2], local[4:], ctx.inplace)
                if ctx.mode == "environment":
                    index = e
                    weights = local[3] * torch.sigmoid(heads[:, 0])
                else:
                    index = a
                    weights = (_logits(heads, local[3]) - maximum.index_select(0, a)).exp()
                weighted = weights[:, None] * heads[:, 1:]
                values = torch.autograd.grad(
                    (weights, weighted), targets,
                    (grad_mass.index_select(0, index), grad_numerator.index_select(0, index)),
                    allow_unused=False,
                )
            for pos, value in zip(target_positions, values):
                if pos == 0:
                    gradients[pos].index_add_(0, a, value)
                elif pos == 1:
                    gradients[pos].index_add_(0, e, value)
                elif pos in (2, 3):
                    gradients[pos][part].copy_(value)
                else:
                    gradients[pos].add_(value)
        return (None, None, None, *gradients)


def stream_pool(mode, atom_projection, env_projection, chi, omega, atom, env,
                messages, heads, relation_part, chunk_size, inplace=False):
    """Return environment (mass, numerator), or normalized atom readback.

    This independently recomputes edge messages during backward, regardless of
    the enclosing module checkpoint switch. It is intended for eager execution;
    higher-order differentiation is explicitly rejected.
    """
    start = sum(messages.widths[:relation_part])
    end = start + messages.widths[relation_part]
    result = _StreamPool.apply(
        mode, chunk_size, inplace, atom_projection, env_projection, chi, omega, atom, env,
        messages.first.weight[:, start:end], messages.first.bias,
        messages.last.weight, messages.last.bias, heads.weight, heads.bias,
    )
    return result[0] if mode == "readback" else result
