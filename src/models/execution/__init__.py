"""Execution policies; no model parameters or persistent tensors live here."""
from .policy import configure_execution

__all__ = ["configure_execution"]
