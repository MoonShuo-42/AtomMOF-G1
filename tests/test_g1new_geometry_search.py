"""Optional merged periodic search: exact edges, ties, ordering and budgets."""

from itertools import product

import pytest
import torch

from src.models.g1new.geometry import LocalGeometry, SearchBudgetExceeded
from src.models.g1new.geometry import pairs as pair_module


@pytest.fixture(autouse=True, scope="module")
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _random_case(seed):
    rng = torch.Generator().manual_seed(seed)
    # Skew row cells, translated coordinates, padding, hydrogen and invalid rows.
    cell = torch.tensor([
        [[1.1, 0, 0], [.75, .65, 0], [-.45, .3, .9]],
        [[.8, 0, 0], [-.6, .7, 0], [.25, -.4, 1.2]],
    ], dtype=torch.float64)
    fractional = 3 * torch.rand((2, 5, 3), generator=rng, dtype=torch.float64) - 1.5
    x = fractional @ cell
    mask = torch.tensor([[True, True, True, True, False], [True] * 5])
    elements = torch.tensor([[6, 8, 1, 26, 0], [6, 1, 6, 8, 30]])
    bonds = torch.randint(0, 6, (2, 5, 5), generator=rng)
    valid = torch.tensor([True, seed % 2 == 0])
    return x, cell, mask, elements, bonds, valid


def _brute_force(case, cutoff, pair_chunk):
    x, cell, mask, elements, bonds, valid = case
    atoms = mask.shape[1]
    anchors = (mask & (elements != 1)).flatten().nonzero(as_tuple=True)[0]
    shifts = torch.tensor(list(product(range(-8, 9), repeat=3)), dtype=torch.long)
    out = [[], [], [], []]
    for start in range(0, len(anchors) * atoms, pair_chunk):
        spatial, extra = [], []
        for pair_id in range(start, min(start + pair_chunk, len(anchors) * atoms)):
            env, local = divmod(pair_id, atoms)
            anchor = int(anchors[env])
            batch = anchor // atoms
            if not valid[batch] or not mask[batch, local]:
                continue
            atom = batch * atoms + local
            delta = x[batch, local] - x.reshape(-1, 3)[anchor]
            r = delta + shifts.to(x.dtype) @ cell[batch]
            squared = r.square().sum(-1)
            winner = int(squared.argmin())  # enumeration is lexicographic
            # Our fixed exhaustive box is far wider than each skew test cell's
            # nearest/cutoff support; ensure a boundary cannot be the winner.
            assert (shifts[winner].abs() < 8).all()
            known = int(bonds[batch, local, anchor % atoms])
            central = (atom == anchor) & (shifts == 0).all(-1)
            take = ((torch.linalg.vector_norm(r, dim=-1) < cutoff) & ~central).nonzero().flatten()
            for index in take.tolist():
                spatial.append((atom, env, shifts[index], known if index == winner else 0))
            if known and squared[winner].sqrt() >= cutoff and not central[winner]:
                extra.append((atom, env, shifts[winner], known))
        for record in spatial + extra:
            for destination, value in zip(out, record):
                destination.append(value)
    result = [torch.tensor(out[0]), torch.tensor(out[1]),
              torch.stack(out[2]) if out[2] else torch.empty((0, 3), dtype=torch.long),
              torch.tensor(out[3])]
    return (anchors, *[value.long() for value in result])


@pytest.mark.parametrize("seed", [10, 11, 12])
@pytest.mark.parametrize("pair_chunk,image_chunk", [(7, 19), (64, 113)])
def test_merged_edges_match_reference_and_skew_cell_exhaustive_search(seed, pair_chunk, image_chunk):
    case = _random_case(seed)
    actual = LocalGeometry(merge_search=True, pair_chunk_size=pair_chunk, image_chunk_size=image_chunk)
    old = LocalGeometry(pair_chunk_size=pair_chunk, image_chunk_size=image_chunk)
    reference = LocalGeometry(pair_chunk_size=pair_chunk, image_chunk_size=image_chunk)
    reference.optimized = False
    outputs = [geometry._search(*case) for geometry in (actual, old, reference)]
    outputs.append(_brute_force(case, actual.cutoff, pair_chunk))
    for expected in outputs[1:]:
        for a, b in zip(outputs[0], expected):
            assert torch.equal(a, b)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("cutoff", [.4, 1., 1.1])
def test_exact_tie_outside_cutoff_and_inside_cutoff_keep_lexicographic_image(device, cutoff):
    x = torch.tensor([[[0., 0, 0], [1., 0, 0]]], dtype=torch.float64, device=device)
    cell = torch.eye(3, dtype=x.dtype, device=device)[None] * 2
    mask = torch.ones((1, 2), dtype=torch.bool, device=device)
    elements = torch.full((1, 2), 6, dtype=torch.long, device=device)
    bonds = torch.tensor([[[0, 3], [3, 0]]], device=device)
    valid = torch.ones(1, dtype=torch.bool, device=device)
    options = dict(cutoff=cutoff, inner_cutoff=cutoff / 2, pair_chunk_size=4, image_chunk_size=1)
    merged = LocalGeometry(merge_search=True, **options)
    old = LocalGeometry(**options)
    result = merged._search(x, cell, mask, elements, bonds, valid)
    for a, b in zip(result, old._search(x, cell, mask, elements, bonds, valid)):
        assert torch.equal(a, b)
    _, atom, env, shift, bond = result
    chosen = (atom == 1) & (env == 0) & (bond == 3)
    assert shift[chosen].tolist() == [[-1, 0, 0]]


def test_search_candidate_budget_counts_actual_union_and_edges_fail_before_buffering(monkeypatch):
    observations = []
    original_budget = pair_module._Budget

    class RecordingBudget(original_budget):
        def __init__(self, *args):
            super().__init__(*args)
            observations.append(self)

    monkeypatch.setattr(pair_module, "_Budget", RecordingBudget)
    case = _random_case(10)
    settings = dict(pair_chunk_size=64, image_chunk_size=19)
    old = LocalGeometry(**settings)
    merged = LocalGeometry(merge_search=True, **settings)
    expected, actual = old._search(*case), merged._search(*case)
    old_count, merged_count = [int(item.searches) for item in observations]
    assert 0 < merged_count < old_count
    for a, b in zip(expected, actual):
        assert torch.equal(a, b)
    LocalGeometry(merge_search=True, max_search_candidates=merged_count, **settings)._search(*case)
    with pytest.raises(SearchBudgetExceeded, match="searches budget"):
        LocalGeometry(merge_search=True, max_search_candidates=merged_count - 1, **settings)._search(*case)
    edge_count = actual[1].numel()
    LocalGeometry(merge_search=True, max_edges=edge_count, **settings)._search(*case)
    with pytest.raises(SearchBudgetExceeded, match="edges budget"):
        LocalGeometry(merge_search=True, max_edges=edge_count - 1, **settings)._search(*case)


@pytest.mark.parametrize("empty", ["no_bonds", "hydrogen", "padding", "invalid"])
def test_empty_invalid_and_no_chemical_support_match_reference(empty):
    case = list(_random_case(10))
    if empty == "no_bonds":
        case[4].zero_()
    elif empty == "hydrogen":
        case[3].fill_(1)
    elif empty == "padding":
        case[2].zero_()
    else:
        case[5].zero_()
        case[4].fill_(-1)  # unsupported invalid samples must not create errors
    merged = LocalGeometry(merge_search=True)
    reference = LocalGeometry()
    reference.optimized = False
    for a, b in zip(merged._search(*case), reference._search(*case)):
        assert torch.equal(a, b)


def test_merged_flag_is_opt_in_and_reference_policy_bypasses_it(monkeypatch):
    assert LocalGeometry().merge_search is False
    with pytest.raises(TypeError, match="merge_search"):
        LocalGeometry(merge_search=1)
    model = LocalGeometry(merge_search=True)
    model.optimized = False
    monkeypatch.setattr(model, "_merged_search", lambda *a, **k: pytest.fail("merged search invoked"))
    model._search(*_random_case(10))


def test_live_geometry_values_and_coordinate_lattice_gradients_are_unchanged():
    torch.manual_seed(73)
    x = torch.rand((1, 4, 3), dtype=torch.float64)
    lattice = torch.tensor([[1.2, 1.5, 1.1, 1.3, 1.4, 1.1]], dtype=torch.float64)
    feats = dict(atom_pad_mask=torch.ones((1, 4), dtype=torch.bool),
                 ref_element=torch.tensor([[6, 1, 8, 26]]),
                 bond_types=torch.tensor([[[0, 1, 0, 0], [1, 0, 0, 0],
                                           [0, 0, 0, 4], [0, 0, 4, 0]]]),
                 block_index=torch.tensor([[0, 0, 1, 1]]))
    args = [[value.clone().requires_grad_() for value in (x, lattice)] for _ in range(2)]
    geometries = [LocalGeometry(merge_search=flag, pair_chunk_size=5, image_chunk_size=7)
                  for flag in (False, True)]
    outputs = [geometry(*inputs, feats) for geometry, inputs in zip(geometries, args)]
    for a, b in zip(*outputs):
        assert torch.equal(a, b)
    for output in outputs:
        loss = (output.distance.square().sum() + output.direction.square().sum()
                + output.omega.square().sum())
        loss.backward()
    for left, right in zip(*args):
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
