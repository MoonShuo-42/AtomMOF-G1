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
