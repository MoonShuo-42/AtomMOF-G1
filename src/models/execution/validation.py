"""Combine device predicate checks within one forward, without dropping errors."""
from contextlib import contextmanager
from contextvars import ContextVar

import torch

_pending = ContextVar("atommof_device_checks", default=None)


def check_tensor(predicate, message):
    pending = _pending.get()
    if pending is None:
        if not bool(predicate):
            raise ValueError(message)
    else:
        pending.append((predicate.detach(), message))


@contextmanager
def deferred_tensor_checks(enabled):
    if not enabled or _pending.get() is not None:
        yield
        return
    checks = []
    token = _pending.set(checks)
    try:
        yield
    finally:
        _pending.reset(token)
    # Same-device forward only. One scalar host read on the successful path.
    if checks:
        valid = torch.stack([value for value, _ in checks])
        if not bool(valid.all()):
            failed = valid.cpu().tolist()
            raise ValueError(
                "; ".join(message for ok, (_, message) in zip(failed, checks) if not ok)
            )
