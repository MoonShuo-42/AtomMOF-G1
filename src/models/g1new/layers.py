"""Small shared MLPs; split projections avoid repeating atom features per edge."""

import torch
from torch import nn
from torch.nn import functional as F


def mlp(in_dim, hidden_dim, out_dim):
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, out_dim))


class JointMLP(nn.Module):
    """Exactly Linear(concat(parts))->SiLU->Linear, with the same parameters.

    Atom/environment contributions may be projected before gathering. Only the
    small edge relation projection then runs T times. Bias is added exactly once.
    No low-rank factorization, weight tying between inputs, or reduced precision."""

    def __init__(self, widths, hidden, output):
        super().__init__()
        self.widths = tuple(widths)
        self.first = nn.Linear(sum(widths), hidden)
        self.last = nn.Linear(hidden, output)

    def project(self, x, part, *, bias=False):
        start = sum(self.widths[:part])
        return F.linear(
            x,
            self.first.weight[:, start : start + self.widths[part]],
            self.first.bias if bias else None,
        )

    def finish(self, summed):
        return self.last(F.silu(summed))

    def fused_heads(self, heads):
        """Compose consecutive affine maps, retaining BOTH original parameters.

        There is no activation between last and heads. Compute the small weight
        product once per call, not the intermediate 64-channel tensor per edge.
        Autograd differentiates through this product; nothing is detached/cached
        across forwards and the nonlinear SiLU remains in its original location.
        """
        weight = heads.weight @ self.last.weight
        bias = F.linear(self.last.bias, heads.weight, heads.bias)

        def apply(hidden):
            return F.linear(F.silu(hidden), weight, bias)

        return apply


def edge_slices(length, chunk_size):
    # One empty invocation keeps all learned paths connected on empty ranks.
    for start in range(0, max(length, 1), chunk_size):
        yield slice(start, start + chunk_size)


def recompute(function, enabled, *args):
    if enabled and torch.is_grad_enabled():
        from torch.utils.checkpoint import checkpoint

        # All functions here are deterministic and contain no dropout/randomness.
        return checkpoint(function, *args, use_reentrant=False, preserve_rng_state=False)
    return function(*args)


def split_apply(function, chunk_size, checkpointing, *values):
    """Chunk an edge map with one SplitBackward per input, not T-sized slice grads.

    Individual x[start:end] operations each allocate a full input-sized gradient.
    split() joins the chunk gradients once. Keep an empty call for DDP connectivity.
    """
    parts = [value.split(chunk_size, dim=0) for value in values]
    outputs = [recompute(function, checkpointing, *group) for group in zip(*parts)]
    # cat([x]) still allocates/copies x. The common single-chunk and empty cases
    # can return the result directly, retaining the same autograd connections.
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
