"""Exact periodic local edges, streamed with explicit fail-fast resource budgets.

No nearest-neighbour cap, fixed 27-cell assumption, or silent geometry dropout.
Discrete search uses detached float64; selected geometry is recomputed with
autograd in FP32 (or FP64 for numerical tests). Different images remain edges.
"""

import math
from typing import NamedTuple

import torch
from torch import Tensor, nn

from .lattice import lattice_parameters_to_cell, validate_cell


class SearchBudgetExceeded(RuntimeError):
    """An exact search exceeds its configured workspace/work budget."""


class LocalPairs(NamedTuple):
    anchors: Tensor  # (M,), flattened original B*N atom addresses
    atom: Tensor  # (T,), flattened receiving/partner atom addresses
    env: Tensor  # (T,), packed environment addresses
    shift: Tensor  # (T,3), integer periodic images
    bond: Tensor  # (T,), only the closest known-bond image is nonzero
    distance: Tensor  # (T,)
    direction: Tensor  # (T,3)
    same_block: Tensor  # (T,)
    omega: Tensor  # (T,)
    geometry_ok: Tensor  # (B,), invalid input geometry disables this branch


class _Budget:
    def __init__(self, searches, edges):
        self.max_searches, self.max_edges = searches, edges
        self.searches = self.edges = 0

    def consume(self, count, kind):
        limit = self.max_searches if kind == "searches" else self.max_edges
        value = getattr(self, kind) + count
        if value > limit:
            raise SearchBudgetExceeded(
                f"g1new exact periodic search exceeds {kind} budget: {value} > {limit}. "
                "No candidates were truncated and no sample was silently bypassed. "
                "Inspect current cell geometry, reduce batch size, or explicitly "
                "increase the corresponding g1new geometry resource budget."
            )
        setattr(self, kind, value)


class LocalGeometry(nn.Module):
    def __init__(
        self,
        cutoff=0.8,
        inner_cutoff=0.6,
        pair_chunk_size=65536,
        image_chunk_size=262144,
        max_search_candidates=32000000,
        max_edges=2000000,
        min_cell_singular_value=1e-6,
        max_cell_condition=1e6,
        bounds_margin=1e-9,
    ):
        super().__init__()
        if not (math.isfinite(cutoff) and 0 <= inner_cutoff < cutoff):
            raise ValueError("Require finite 0 <= inner_cutoff < cutoff (nm)")
        for name, value in dict(
            pair_chunk_size=pair_chunk_size,
            image_chunk_size=image_chunk_size,
            max_search_candidates=max_search_candidates,
            max_edges=max_edges,
        ).items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not math.isfinite(min_cell_singular_value)
            or min_cell_singular_value <= 0
            or not math.isfinite(max_cell_condition)
            or max_cell_condition < 1
            or not math.isfinite(bounds_margin)
            or bounds_margin < 0
        ):
            raise ValueError("Invalid cell/search numerical thresholds")
        self.cutoff, self.inner_cutoff = cutoff, inner_cutoff
        self.optimized = True
        self.pair_chunk_size, self.image_chunk_size = pair_chunk_size, image_chunk_size
        self.max_search_candidates, self.max_edges = max_search_candidates, max_edges
        self.min_cell_singular_value, self.max_cell_condition = (
            min_cell_singular_value,
            max_cell_condition,
        )
        self.bounds_margin = bounds_margin

    def _boxes(self, delta, inverse, radius, budget):
        fractional = torch.bmm(delta[:, None], inverse).squeeze(1)
        extent = radius[:, None] * torch.linalg.vector_norm(inverse, dim=-2)
        margin = self.bounds_margin + 16 * torch.finfo(torch.float64).eps * (
            1 + fractional.abs() + extent
        )
        lo = torch.ceil(-fractional - extent - margin)
        hi = torch.floor(-fractional + extent + margin)
        widths = (hi - lo + 1).clamp_min(0)
        counts = widths.prod(-1)
        representable = (
            torch.isfinite(lo).all()
            & torch.isfinite(hi).all()
            & (lo.abs() < 2**52).all()
            & (hi.abs() < 2**52).all()
            & torch.isfinite(counts).all()
        )
        if self.optimized:
            # One bounded host read instead of independent synchronizations for
            # each predicate and the candidate count. Decisions remain float64.
            safe, total = torch.stack((representable.to(counts.dtype), counts.sum())).tolist()
        else:
            safe, total = bool(representable), float(counts.sum())
        if not safe:
            raise SearchBudgetExceeded(
                "g1new periodic integer bounds are not safely representable"
            )
        budget.consume(total, "searches")  # check BEFORE float-to-int conversion/allocation
        counts = counts.long()
        return lo.long(), widths.long(), counts.cumsum(0), int(total)

    @staticmethod
    def _decode(indices, pair, boxes):
        lower, widths, ends, _ = boxes
        starts = ends - widths.prod(-1)
        offset = indices - starts[pair]
        w = widths[pair]
        shift = torch.stack(
            (offset // (w[:, 1] * w[:, 2]), (offset // w[:, 2]) % w[:, 1], offset % w[:, 2]), -1
        )
        return shift + lower[pair]

    def _enumerate(self, boxes):
        for start in range(0, boxes[3], self.image_chunk_size):
            indices = torch.arange(
                start, min(start + self.image_chunk_size, boxes[3]), device=boxes[0].device
            )
            pair = torch.searchsorted(boxes[2], indices, right=True)
            yield indices, pair, self._decode(indices, pair, boxes)

    def _nearest(self, delta, cells, inverse, budget):
        """Exact nearest image, lexicographically first shift for exact ties."""
        frac = torch.bmm(delta[:, None], inverse).squeeze(1)
        initial = -frac.round()
        upper = torch.linalg.vector_norm(
            delta + torch.bmm(initial[:, None], cells).squeeze(1), dim=-1
        )
        boxes = self._boxes(delta, inverse, upper, budget)
        best = delta.new_full((len(delta),), float("inf"))
        sentinel = torch.iinfo(torch.long).max
        winner = torch.full((len(delta),), sentinel, dtype=torch.long, device=delta.device)
        for index, pair, shift in self._enumerate(boxes):
            r = delta[pair] + torch.bmm(shift.to(cells.dtype)[:, None], cells[pair]).squeeze(1)
            d2 = r.square().sum(-1)
            partial = best.new_full(best.shape, float("inf")).scatter_reduce_(
                0, pair, d2, reduce="amin", include_self=True
            )
            candidate = torch.where(d2 == partial[pair], index, sentinel)
            first = winner.new_full(winner.shape, sentinel).scatter_reduce_(
                0, pair, candidate, reduce="amin", include_self=True
            )
            take = (partial < best) | ((partial == best) & (first < winner))
            best, winner = torch.where(take, partial, best), torch.where(take, first, winner)
        if bool((winner == sentinel).any()):
            raise RuntimeError(
                "g1new nearest-image search failed to cover its initial upper bound"
            )
        return (
            self._decode(winner, torch.arange(len(delta), device=delta.device), boxes),
            best.sqrt(),
        )

    @torch.no_grad()
    def _search(self, x, cell, mask, elements, bonds, valid):
        if not self.optimized:
            return self._search_reference(x, cell, mask, elements, bonds, valid)
        _, atoms = mask.shape
        heavy = mask & (elements != 1)
        anchors = heavy.flatten().nonzero(as_tuple=True)[0]
        budget = _Budget(self.max_search_candidates, self.max_edges)
        x64, cell64 = x.reshape(-1, 3).double(), cell.double()
        inv = torch.linalg.inv(cell64)
        parts = [[], [], [], []]

        # Same support as the reference checks, once per forward. Padding,
        # invalid samples and hydrogen anchors do not introduce new errors.
        supported = valid[:, None, None] & mask[:, :, None] & heavy[:, None, :]
        if bool((((bonds < 0) | (bonds > 5)) & supported).any()):
            raise ValueError("g1new requires bond_types in 0..5")
        flat_bonds = bonds.reshape(-1)

        def append(a, e, n, bond):
            budget.consume(a.numel(), "edges")
            if a.numel():
                for target, value in zip(parts, (a, e, n, bond)):
                    target.append(value)

        for start in range(0, anchors.numel() * atoms, self.pair_chunk_size):
            ids = torch.arange(
                start, min(start + self.pair_chunk_size, anchors.numel() * atoms), device=x.device
            )
            env, local = ids // atoms, ids % atoms
            batch = anchors.index_select(0, env) // atoms
            keep = (valid[batch] & mask[batch, local]).nonzero(as_tuple=True)[0]
            if not keep.numel():
                continue
            # Compact once and reuse the integer layout; boolean indexing each
            # component separately would repeat nonzero/host synchronization.
            ids = ids.index_select(0, keep)
            env, local = ids // atoms, ids % atoms
            anchor = anchors.index_select(0, env)
            batch = anchor // atoms
            partner = batch * atoms + local
            delta = x64.index_select(0, partner) - x64.index_select(0, anchor)
            cells, inverse = cell64.index_select(0, batch), inv.index_select(0, batch)
            types = flat_bonds.index_select(0, partner * atoms + anchor % atoms)
            chemical = (types > 0).nonzero(as_tuple=True)[0]
            nearest = torch.zeros_like(delta, dtype=torch.long)
            nearest_d = delta.new_full((len(delta),), float("inf"))
            if chemical.numel():
                shifts, distances = self._nearest(
                    delta.index_select(0, chemical), cells.index_select(0, chemical),
                    inverse.index_select(0, chemical), budget,
                )
                nearest.index_copy_(0, chemical, shifts)
                nearest_d.index_copy_(0, chemical, distances)

            boxes = self._boxes(delta, inverse, delta.new_full((len(delta),), self.cutoff), budget)
            for _, pair, shifts in self._enumerate(boxes):
                a, anchor_image = partner.index_select(0, pair), anchor.index_select(0, pair)
                r = delta.index_select(0, pair) + torch.bmm(
                    shifts.to(cells.dtype)[:, None], cells.index_select(0, pair)
                ).squeeze(1)
                within = torch.linalg.vector_norm(r, dim=-1) < self.cutoff
                central = (a == anchor_image) & (shifts == 0).all(-1)
                take = (within & ~central).nonzero(as_tuple=True)[0]
                pair, shifts = pair.index_select(0, take), shifts.index_select(0, take)
                bond = torch.where(
                    (shifts == nearest.index_select(0, pair)).all(-1),
                    types.index_select(0, pair), 0,
                )
                append(partner.index_select(0, pair), env.index_select(0, pair), shifts, bond)

            extra = ((types > 0) & (nearest_d >= self.cutoff)
                     & ~((partner == anchor) & (nearest == 0).all(-1))).nonzero(as_tuple=True)[0]
            append(
                partner.index_select(0, extra), env.index_select(0, extra),
                nearest.index_select(0, extra), types.index_select(0, extra),
            )
        empty = torch.empty(0, dtype=torch.long, device=x.device)
        result = [torch.cat(p) if p else empty for p in parts]
        if not parts[2]:
            result[2] = empty.reshape(0, 1).expand(0, 3)
        return (anchors, *result)

    @torch.no_grad()
    def _search_reference(self, x, cell, mask, elements, bonds, valid):
        batch, atoms = mask.shape
        anchors = (mask & (elements != 1)).flatten().nonzero(as_tuple=True)[0]
        budget = _Budget(self.max_search_candidates, self.max_edges)
        x64, cell64 = x.reshape(-1, 3).double(), cell.double()
        inv = torch.linalg.inv(cell64)
        atom_parts, env_parts, shift_parts, bond_parts = [], [], [], []

        def append(atom, env, shifts, types):
            budget.consume(atom.numel(), "edges")
            if atom.numel():
                atom_parts.append(atom)
                env_parts.append(env)
                shift_parts.append(shifts)
                bond_parts.append(types)

        # Stream the M*N index space; do not allocate a B*N*M*images tensor.
        for start in range(0, anchors.numel() * atoms, self.pair_chunk_size):
            ids = torch.arange(
                start, min(start + self.pair_chunk_size, anchors.numel() * atoms), device=x.device
            )
            env, local_atom = ids // atoms, ids % atoms
            b = anchors[env] // atoms
            keep = valid[b] & mask[b, local_atom]
            env, local_atom, b = env[keep], local_atom[keep], b[keep]
            if not env.numel():
                continue
            partner = b * atoms + local_atom
            anchor = anchors[env]
            delta = x64[partner] - x64[anchor]
            cells, inverse = cell64[b], inv[b]
            types = bonds[b, local_atom, anchor % atoms]
            if bool(((types < 0) | (types > 5)).any()):
                raise ValueError("g1new requires bond_types in 0..5")
            chemical = types > 0
            nearest = torch.zeros_like(delta, dtype=torch.long)
            nearest_d = delta.new_full((len(delta),), float("inf"))
            if bool(chemical.any()):
                shifts, distances = self._nearest(
                    delta[chemical], cells[chemical], inverse[chemical], budget
                )
                nearest[chemical], nearest_d[chemical] = shifts, distances
            boxes = self._boxes(delta, inverse, delta.new_full((len(delta),), self.cutoff), budget)
            for _, pair, shifts in self._enumerate(boxes):
                r = delta[pair] + torch.bmm(shifts.to(cells.dtype)[:, None], cells[pair]).squeeze(
                    1
                )
                within = torch.linalg.vector_norm(r, dim=-1) < self.cutoff
                central = (partner[pair] == anchor[pair]) & (shifts == 0).all(-1)
                take = within & ~central
                pair, shifts = pair[take], shifts[take]
                edge_type = torch.where((shifts == nearest[pair]).all(-1), types[pair], 0)
                append(partner[pair], env[pair], shifts, edge_type)
            # Union without sorting/deduplicating a giant edge table: only append
            # chemical edges outside the spatial set. Each pair/image is unique.
            extra = chemical & (nearest_d >= self.cutoff)
            extra &= ~((partner == anchor) & (nearest == 0).all(-1))
            append(partner[extra], env[extra], nearest[extra], types[extra])
        empty = torch.empty(0, dtype=torch.long, device=x.device)
        return (
            anchors,
            torch.cat(atom_parts) if atom_parts else empty,
            torch.cat(env_parts) if env_parts else empty,
            torch.cat(shift_parts) if shift_parts else empty.reshape(0, 1).expand(0, 3),
            torch.cat(bond_parts) if bond_parts else empty,
        )

    def forward(self, x, lattice, feats):
        mask = feats["atom_pad_mask"].bool()
        if x.shape != (*mask.shape, 3) or lattice.shape != (mask.shape[0], 6):
            raise ValueError("g1new geometry expects matching expanded (B,N,3)/(B,6) inputs")
        if x.dtype not in (torch.float32, torch.float64) or lattice.dtype != x.dtype:
            raise ValueError("g1new geometry requires matching FP32 (or test FP64) inputs")
        with torch.autocast(device_type=x.device.type, enabled=False):
            cell, valid = lattice_parameters_to_cell(lattice)
            valid = valid & validate_cell(
                cell, self.min_cell_singular_value, self.max_cell_condition
            )
            finite = torch.isfinite(x).all(-1)
            valid = valid & (finite | ~mask).all(-1)
            safe_x = torch.where((finite & mask & valid[:, None])[..., None], x, 0)
            safe_cell = torch.where(
                valid[:, None, None], cell, torch.eye(3, device=x.device, dtype=x.dtype)
            )
            anchors, atom, env, shift, bond = self._search(
                safe_x, safe_cell, mask, feats["ref_element"], feats["bond_types"], valid
            )
            # Rebuild only chosen displacements with live tensors. No detached
            # distances or lattice inverses enter the differentiable messages.
            flat = safe_x.reshape(-1, 3)
            displacements = []
            for start in range(0, max(atom.numel(), 1), self.image_chunk_size):
                a, e, n = (
                    atom[start : start + self.image_chunk_size],
                    env[start : start + self.image_chunk_size],
                    shift[start : start + self.image_chunk_size],
                )
                image = torch.bmm(n.to(x.dtype)[:, None], safe_cell[a // mask.shape[1]]).squeeze(1)
                displacements.append(flat[a] + image - flat[anchors[e]])
            r = torch.cat(displacements)
            distance = torch.linalg.vector_norm(r, dim=-1)
            direction = r / distance.clamp_min(1e-6)[:, None]
            phase = ((distance - self.inner_cutoff) / (self.cutoff - self.inner_cutoff)).clamp(
                0, 1
            )
            omega = 0.5 * (1 + torch.cos(math.pi * phase))
            omega = torch.where(bond > 0, torch.ones_like(omega), omega)
            blocks = feats["block_index"].reshape(-1)
            same = (blocks[atom] == blocks[anchors[env]]).to(x.dtype)
            return LocalPairs(
                anchors, atom, env, shift, bond, distance, direction, same, omega, valid
            )
