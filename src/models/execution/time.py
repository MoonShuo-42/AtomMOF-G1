"""Forward-local reuse of fixed time features; learned MLPs remain independent."""
from contextlib import contextmanager
from contextvars import ContextVar

import torch

_FEATURES = ContextVar("atommof_time_features", default=None)


@contextmanager
def time_feature_scope(enabled):
    # Never retain an autograd graph on a module or across network calls.
    token = _FEATURES.set({} if enabled else None)
    try:
        yield
    finally:
        _FEATURES.reset(token)


def shared_time_features(spec, inputs, factory):
    cache = _FEATURES.get()
    if cache is None:
        return factory()
    key = (spec, tuple(id(x) for x in inputs), torch.is_grad_enabled())
    if key not in cache:
        # Strong references prevent tensor-id reuse during the scope.
        cache[key] = (inputs, factory())
    return cache[key][1]


def constant_frequencies(module, key, factory):
    cached = getattr(module, "_time_frequencies", None)
    if cached is None or cached[0] != key:
        # Plain, non-differentiable runtime cache: no checkpoint/state_dict change.
        # Key includes calculation dtype/device, so .to() never downcasts constants.
        with torch.inference_mode(False), torch.no_grad():
            module._time_frequencies = (key, factory())
    return module._time_frequencies[1]
