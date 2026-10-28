"""A single optional insertion, separate from the legacy four-bridge scheduler."""

from copy import deepcopy

from hydra.utils import instantiate
from torch import nn

from src.models.execution.ops import repeat_batch

from .module import G1NewModule


class G1NewBridge(nn.Module):
    def __init__(self, h_dim, depth, insert_after=12, module=None):
        super().__init__()
        if type(insert_after) is not int or not 1 <= insert_after <= depth:
            raise ValueError("g1new.insert_after must be a one-based layer <= trunk depth")
        self.insert_after = insert_after
        self.module = G1NewModule(h_dim=h_dim) if module is None else instantiate(deepcopy(module))
        if not isinstance(self.module, G1NewModule) or self.module.h_dim != h_dim:
            raise ValueError("g1new module.h_dim must equal token_s")

    def encode_condition(self, feats):
        return self.module.encode_condition(feats)

    def prepare(self, feats, x_t, l_t, multiplicity=1, condition=None):
        if type(multiplicity) is not int or multiplicity < 1:
            raise ValueError("multiplicity must be a positive integer")
        enabled = getattr(self, "avoid_repeat_copies", False)
        # Preserve repeat_interleave ordering used by both training and sampling.
        expanded = {
            name: repeat_batch(feats[name], multiplicity, enabled=enabled)
            for name in ("atom_pad_mask", "ref_element", "bond_types", "block_index")
        }
        if condition is None:
            condition = self.encode_condition(feats)
        condition = repeat_batch(condition, multiplicity, enabled=enabled)
        edges = self.module.geometry(x_t, l_t, expanded)
        return expanded, edges, condition

    def forward(self, layer, hidden, times, context):
        if layer != self.insert_after:
            return hidden
        return self.module(hidden, times, *context)
