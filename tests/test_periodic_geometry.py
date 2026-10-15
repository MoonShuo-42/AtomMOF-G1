"""Numerical and behavioral acceptance tests for G1 periodic geometry."""

import ast
import itertools
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.models.g1.geometry import (
    PeriodicGeometry,
    PeriodicPairMoments,
    scalarize_geometry,
    unpack_second_moment,
)
from src.models.g1.geometry.lattice import lattice_parameters_to_cell


@pytest.fixture(scope="module", autouse=True)
def small_cpu_workloads():
    """Keep tiny geometry and gradient checks from oversubscribing CPU threads."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def brute_force(coords, cell, radius=0.6, image_range=6):
    """Independent NumPy oracle: exhaust a large integer cube, then filter the sphere."""
    coords, cell = np.asarray(coords), np.asarray(cell)
    shifts = np.array(list(itertools.product(range(-image_range, image_range + 1), repeat=3)))
    result = np.zeros((len(coords), len(coords), 16))
    for i, xi in enumerate(coords):
        for j, xj in enumerate(coords):
            displacement = xj - xi + shifts @ cell
            selected = np.linalg.norm(displacement, axis=-1) < radius
            if i == j:
                selected &= np.any(shifts != 0, axis=-1)
            y = displacement[selected] / radius
            q = np.einsum("ij,ij->i", y, y)
            w = (1 - q) ** 3
            strength = w.sum()
            beta = w / (strength + 0.1)
            radial = (
                beta[:, None] * np.exp(-0.5 * ((q[:, None] - np.arange(6) / 5) / 0.2) ** 2)
            ).sum(0)
            vector = np.einsum("i,ij->j", beta, y)
            second = np.einsum("i,ij,ik->jk", beta, y, y)
            result[i, j] = np.concatenate(
                (radial, vector, second[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]], [strength])
            )
    return torch.tensor(result, dtype=torch.float64)


def sample():
    """An oblique cell with several images, no cutoff ties, and three atoms."""
    params = torch.tensor([[0.75, 0.81, 0.93, 1.23, 1.38, 1.31]], dtype=torch.float64)
    cell, _ = lattice_parameters_to_cell(params)
    frac = torch.tensor(
        [[[0.07, 0.12, 0.16], [0.84, 0.19, 0.27], [0.37, 0.76, 0.69]]], dtype=torch.float64
    )
    return frac @ cell, params, cell, torch.ones(1, 3, dtype=torch.bool)


def test_lattice_gauge_matches_original_atommof_helper():
    """The new nm/radian cell has the same orientation as the original Å/degree helper."""
    # Execute the actual pure-Torch helper without importing unrelated pymatgen analysis.
    path = Path(__file__).resolve().parents[1] / "src/utils/data_utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lattice_params_to_matrix_torch"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    lattice_params_to_matrix_torch = namespace["lattice_params_to_matrix_torch"]

    _, params, cell, _ = sample()
    expected = (
        lattice_params_to_matrix_torch(params[:, :3] * 10, torch.rad2deg(params[:, 3:])) / 10
    )
    torch.testing.assert_close(cell, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(cell[:, 2, :2], torch.zeros(1, 2, dtype=cell.dtype), rtol=0, atol=0)


def test_documented_boundary_example_matches_all_sixteen_channels():
    """The +0.8 nm direct difference must become the -0.2 nm periodic neighbor."""
    coordinates = torch.tensor([[[0.1, 0, 0], [0.9, 0, 0]]], dtype=torch.float64)
    cell = torch.eye(3, dtype=torch.float64)[None]
    output = PeriodicGeometry().from_cell(coordinates, cell, torch.ones(1, 2))
    expected = torch.tensor(
        [
            0.7501836355186575,
            0.7930399347718736,
            0.3084097126990845,
            0.04412315452509808,
            0.002322258598971117,
            4.4963501400507596e-05,
            -0.29178776998917194,
            0,
            0,
            0.09726258999639066,
            0,
            0,
            0,
            0,
            0,
            0.7023319615912207,
        ],
        dtype=torch.float64,
    )
    assert output.geometry_ok.tolist() == [True]
    torch.testing.assert_close(output.E[0, 0, 1], expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "cell_values",
    [
        [[0.75, 0, 0.12], [0.25, 0.71, 0.09], [0, 0, 0.83]],
        [[1.0, 0, 0], [0.94, 0.19, 0], [0.03, 0.02, 1.2]],
    ],
)
def test_all_periodic_images_match_large_brute_force_cube(cell_values):
    """Oblique cells need reciprocal bounds, not a fixed nearest-image heuristic."""
    cell = torch.tensor(cell_values, dtype=torch.float64)[None]
    frac = torch.tensor(
        [[[0.13, 0.24, 0.31], [0.88, 0.71, 0.37], [0.52, 0.11, 0.92]]], dtype=torch.float64
    )
    coordinates = frac @ cell
    output = PeriodicGeometry(pair_chunk_size=2).from_cell(coordinates, cell, torch.ones(1, 3))
    assert output.geometry_ok.item()
    expected = brute_force(coordinates[0], cell[0])
    torch.testing.assert_close(output.E[0], expected, rtol=1e-11, atol=1e-12)


def test_nonzero_self_images_and_opposite_images_are_retained():
    """Exclude only identity self; opposite vectors cancel while second moments remain."""
    cell = torch.diag(torch.tensor([0.4, 2.0, 2.0], dtype=torch.float64))[None]
    coordinates = torch.zeros(1, 1, 3, dtype=torch.float64)
    output = PeriodicGeometry().from_cell(coordinates, cell, torch.ones(1, 1))
    expected = brute_force(coordinates[0], cell[0])
    torch.testing.assert_close(output.E[0], expected)
    assert output.E[0, 0, 0, 15] > 0
    torch.testing.assert_close(output.E[0, 0, 0, 6:9], torch.zeros(3, dtype=cell.dtype))
    assert output.E[0, 0, 0, 9] > 0


def test_distinct_coincident_atoms_remain_finite_and_are_not_removed():
    """Zero separation between different atoms is a valid edge, not identity self."""
    coordinates = torch.zeros(1, 2, 3, dtype=torch.float64, requires_grad=True)
    cell = torch.eye(3, dtype=torch.float64)[None].requires_grad_()
    output = PeriodicGeometry().from_cell(coordinates, cell, torch.ones(1, 2))
    assert output.E[0, 0, 1, 15] == 1
    assert output.E[0, 0, 0].count_nonzero() == 0
    (output.E.square().sum() + output.rho.square().sum()).backward()
    assert torch.isfinite(coordinates.grad).all() and torch.isfinite(cell.grad).all()


def test_joint_rotation_transforms_moments_and_preserves_rho():
    """Rotate both Cartesian coordinates and row cell; E is covariant, rho invariant."""
    coordinates, _, cell, mask = sample()
    rotation, _ = torch.linalg.qr(
        torch.tensor([[0.3, 0.9, 0.2], [0.8, 0.1, 0.6], [0.4, 0.5, 0.7]], dtype=torch.float64)
    )
    module = PeriodicGeometry()
    before = module.from_cell(coordinates, cell, mask)
    after = module.from_cell(coordinates @ rotation.T, cell @ rotation.T, mask)
    assert before.geometry_ok.item() and after.geometry_ok.item()
    torch.testing.assert_close(after.E[..., :6], before.E[..., :6], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(after.E[..., 15], before.E[..., 15], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(
        after.E[..., 6:9], before.E[..., 6:9] @ rotation.T, rtol=1e-10, atol=1e-12
    )
    expected_q = rotation @ unpack_second_moment(before.E[..., 9:15]) @ rotation.T
    torch.testing.assert_close(
        unpack_second_moment(after.E[..., 9:15]), expected_q, rtol=1e-10, atol=1e-12
    )
    torch.testing.assert_close(after.rho, before.rho, rtol=1e-10, atol=1e-12)


def test_periodic_representatives_translation_and_permutation_preserve_geometry():
    """Different image representatives must preserve the complete set of neighbors."""
    coordinates, _, cell, mask = sample()
    module = PeriodicGeometry(pair_chunk_size=2)
    before = module.from_cell(coordinates, cell, mask)
    shifts = torch.tensor([[[3, -2, 1], [-1, 2, -3], [2, 1, 0]]], dtype=cell.dtype)
    moved = coordinates + shifts @ cell + coordinates.new_tensor([1.37, -0.28, 0.61])
    permutation = torch.tensor([2, 0, 1])
    after = module.from_cell(moved[:, permutation], cell, mask)
    torch.testing.assert_close(
        after.E, before.E[:, permutation][:, :, permutation], rtol=1e-10, atol=1e-12
    )
    torch.testing.assert_close(after.rho, before.rho[:, permutation], rtol=1e-10, atol=1e-12)
    even = [*range(6), *range(9, 16)]
    torch.testing.assert_close(before.E[..., even], before.E.transpose(1, 2)[..., even])
    torch.testing.assert_close(before.E[..., 6:9], -before.E.transpose(1, 2)[..., 6:9])


def test_scalarization_includes_off_diagonal_terms_and_happens_after_sum():
    """Packed Q is not a six-vector dot product; chi and summation do not commute."""
    geometry = torch.tensor([*range(6), 1, 2, 3, 1, 2, 3, 0.1, 0.2, 0.3, 5], dtype=torch.float64)
    expected = torch.tensor([*range(6), 5, 14, 6, 14.28, 41.2], dtype=torch.float64)
    torch.testing.assert_close(scalarize_geometry(geometry), expected)
    coordinates, _, cell, mask = sample()
    output = PeriodicGeometry().from_cell(coordinates, cell, mask)
    torch.testing.assert_close(output.rho, scalarize_geometry(output.E.sum(2)))
    assert not torch.allclose(output.rho, scalarize_geometry(output.E).sum(2))


def test_padding_is_zero_and_adding_far_atoms_does_not_dilute_rho():
    """Neither padding nor remote atoms should change an existing local neighborhood."""
    module = PeriodicGeometry()
    cell = (torch.eye(3, dtype=torch.float64) * 5)[None]
    coordinates = torch.tensor([[[0, 0, 0], [0.2, 0.1, 0]]], dtype=torch.float64)
    before = module.from_cell(coordinates, cell, torch.ones(1, 2))
    extra = torch.tensor([[[2, 2, 2], [float("nan"), float("inf"), 0]]], dtype=torch.float64)
    extended = torch.cat((coordinates, extra), dim=1).requires_grad_()
    after = module.from_cell(extended, cell, torch.tensor([[1, 1, 1, 0]]))
    assert after.geometry_ok.item()
    torch.testing.assert_close(after.rho[:, :2], before.rho, rtol=0, atol=0)
    assert after.E[:, 3].count_nonzero() == 0
    assert after.E[:, :, 3].count_nonzero() == 0
    assert after.rho[:, 2:].count_nonzero() == 0
    after.rho.sum().backward()
    assert torch.isfinite(extended.grad).all()
    assert extended.grad[:, 3].count_nonzero() == 0


@pytest.mark.parametrize(
    "bad_params",
    [
        [0, 1, 1, 1.57, 1.57, 1.57],
        [-1, 1, 1, 1.57, 1.57, 1.57],
        [float("nan"), 1, 1, 1.57, 1.57, 1.57],
        [1, float("inf"), 1, 1.57, 1.57, 1.57],
        [1, 1, 1, 0.1, 0.1, 3.0],
        [1, 1, 1, 0, 1.57, 1.57],
        [1, 1, 1, math.pi, math.pi, math.pi],
        [1, 1, 1, 1.57, 1e-14, 1.57],
    ],
)
def test_invalid_parameters_disable_whole_sample_with_finite_zero_gradients(bad_params):
    """Invalid noisy cells are not repaired silently and do not poison other samples."""
    good = [1, 1, 1, math.pi / 2, math.pi / 2, math.pi / 2]
    params = torch.tensor([good, bad_params], dtype=torch.float64, requires_grad=True)
    coordinates = torch.tensor(
        [[[0.1, 0, 0], [0.9, 0, 0]]] * 2, dtype=torch.float64, requires_grad=True
    )
    output = PeriodicGeometry()(coordinates, params, torch.ones(2, 2))
    assert output.geometry_ok.tolist() == [True, False]
    assert output.E[1].count_nonzero() == 0 and output.rho[1].count_nonzero() == 0
    (output.E.square().sum() + output.rho.square().sum()).backward()
    assert torch.isfinite(params.grad).all() and torch.isfinite(coordinates.grad).all()
    assert params.grad[1].count_nonzero() == 0
    assert coordinates.grad[1].count_nonzero() == 0


def test_invalid_matrix_coordinates_and_empty_samples_have_distinct_availability():
    """A legal empty neighborhood is available; malformed/fully padded samples are not."""
    cell = torch.eye(3, dtype=torch.float64)[None].repeat(5, 1, 1)
    cell[1, 2] = cell[1, 1]  # Singular
    cell[2, 2, 2] = 1e-9  # Numerical degeneration
    coordinates = torch.zeros(5, 1, 3, dtype=torch.float64)
    coordinates[3, 0, 0] = float("nan")
    mask = torch.tensor([[1], [1], [1], [1], [0]])
    coordinates.requires_grad_()
    cell.requires_grad_()
    output = PeriodicGeometry().from_cell(coordinates, cell, mask)
    assert output.geometry_ok.tolist() == [True, False, False, False, False]
    assert output.E.count_nonzero() == 0 and output.rho.count_nonzero() == 0
    output.rho.sum().backward()
    assert torch.isfinite(coordinates.grad).all() and torch.isfinite(cell.grad).all()


@pytest.mark.parametrize("cap,side", [(125, 0.1), (27, 0.4)])
def test_candidate_overflow_never_leaves_partial_geometry(cap, side):
    """Even if a later chunk fails, the whole sample is unavailable, with zero gradients."""
    cell = (torch.eye(3, dtype=torch.float64) * side)[None].requires_grad_()
    coordinates = torch.tensor(
        [[[0, 0, 0], [side / 2] * 3]], dtype=torch.float64, requires_grad=True
    )
    output = PeriodicGeometry(max_images=cap, pair_chunk_size=1).from_cell(
        coordinates, cell, torch.ones(1, 2)
    )
    assert not output.geometry_ok.item()
    assert output.E.count_nonzero() == 0 and output.rho.count_nonzero() == 0
    output.E.sum().backward()
    assert cell.grad.count_nonzero() == 0 and coordinates.grad.count_nonzero() == 0


def test_smooth_cutoff_and_nontrivial_attenuation():
    """A single-image weight must decay, rather than canceling during normalization."""
    fractions = torch.tensor([0.95, 0.99, 0.999, 1.0, 1.001], dtype=torch.float64)
    distances = (fractions * 0.6).requires_grad_()
    displacement = torch.stack(
        (distances, torch.zeros_like(distances), torch.zeros_like(distances)), -1
    )[:, None]
    output = PeriodicPairMoments()(displacement, torch.ones(5, 1, dtype=torch.bool))
    assert output[0, 15] > output[1, 15] > output[2, 15] > 0
    assert output[0, :6].max() < 0.01
    assert output[2, :6].max() < 1e-7
    assert output[3:].count_nonzero() == 0
    output.sum().backward()
    assert torch.isfinite(distances.grad).all()
    assert distances.grad[3:].count_nonzero() == 0
    assert distances.grad[2].abs() < distances.grad[1].abs() < distances.grad[0].abs()


def test_coordinate_and_lattice_gradients_pass_finite_differences():
    """Validate live gradients through periodic shifts, all kernels, and rho contractions."""
    coordinates, params, _, mask = sample()
    coordinates = coordinates[:, :2].clone().requires_grad_()
    params = params.clone().requires_grad_()
    module = PeriodicGeometry(pair_chunk_size=2, activation_checkpointing=False)

    def values(x, lattice):
        output = module(x, lattice, mask[:, :2])
        return output.E, output.rho

    assert torch.autograd.gradcheck(
        values, (coordinates, params), eps=1e-6, atol=2e-5, rtol=2e-4, fast_mode=True
    )


def test_chunking_and_checkpointing_preserve_coordinate_and_cell_gradients():
    """Recomputation must preserve periodic cell gradients and chunk ordering."""
    coordinates, _, cell, mask = sample()
    expected = None
    torch.manual_seed(42)
    weights = torch.randn(1, 3, 3, 16, dtype=torch.float64)
    for size, enabled in [(4096, False), (2, False), (1, True)]:
        x = coordinates.clone().requires_grad_()
        lattice = cell.clone().requires_grad_()
        output = PeriodicGeometry(
            pair_chunk_size=size, activation_checkpointing=enabled
        ).from_cell(x, lattice, mask)
        loss = (output.E * weights).sum() + output.rho.square().mean()
        gradients = torch.autograd.grad(loss, (x, lattice))
        assert all(g.abs().sum() > 0 and torch.isfinite(g).all() for g in gradients)
        if expected is None:
            expected = (output, gradients)
        else:
            torch.testing.assert_close(output.E, expected[0].E)
            torch.testing.assert_close(output.rho, expected[0].rho)
            for actual, reference in zip(gradients, expected[1]):
                torch.testing.assert_close(actual, reference, rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_precision_policy_and_cpu_autocast(dtype):
    """Promote half inputs, preserve doubles, and keep geometry outside mixed precision."""
    coordinates, params, _, mask = sample()
    x, lattice = coordinates.to(dtype).requires_grad_(), params.to(dtype).requires_grad_()
    module = PeriodicGeometry()
    reference = module(x, lattice, mask)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = module(x, lattice, mask)
    assert output.E.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    torch.testing.assert_close(output.E, reference.E, rtol=0, atol=0)
    torch.testing.assert_close(output.rho, reference.rho, rtol=0, atol=0)
    (output.E.square().mean() + output.rho.square().mean()).backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(lattice.grad).all()


def test_units_config_and_parameter_free_interface():
    """The delivered config accepts flow units; external Å/degree conversion is explicit."""
    coordinates, params, cell, mask = sample()
    root = Path(__file__).resolve().parents[1]
    module = instantiate(OmegaConf.load(root / "configs/g1/geometry/periodic.yaml"))
    output = module(coordinates, params, mask)
    assert output.E.shape == (1, 3, 3, 16) and output.rho.shape == (1, 3, 11)
    assert list(module.parameters()) == []
    assert module.activation_checkpointing is False
    assert output.geometry_ok.tolist() == [True]
    converted = torch.cat((params[:, :3] * 10, torch.rad2deg(params[:, 3:])), -1)
    external = PeriodicGeometry(length_unit="angstrom", angle_unit="degree")
    for result in (
        external(coordinates * 10, converted, mask),
        external.from_cell(coordinates * 10, cell * 10, mask),
    ):
        torch.testing.assert_close(result.E, output.E, rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(result.rho, output.rho, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cutoff": 0},
        {"radial_sigma": 0},
        {"delta": 0},
        {"max_images": 0},
        {"pair_chunk_size": 0},
        {"min_cell_singular_value": 0},
        {"length_unit": "auto"},
    ],
)
def test_invalid_configuration_fails_explicitly(kwargs):
    """Invalid configuration must not silently alter the geometric definition."""
    with pytest.raises(ValueError):
        PeriodicGeometry(**kwargs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_batched_plan_matches_separate_samples_and_gradients(dtype, checkpoint):
    """Mixed cells/padding/overflow retain per-sample semantics across chunk boundaries."""
    x, _, cell, mask = sample()
    x, cell = x.to(dtype).repeat(4, 1, 1), cell.to(dtype).repeat(4, 1, 1)
    mask = mask.repeat(4, 1)
    cell[1] *= 1.4
    x[1] *= 1.4
    mask[1, 1] = False
    x[1, 1] = float("nan")
    cell[2] *= 0.05  # Candidate overflow invalidates this sample only.
    mask[3] = False
    x.requires_grad_()
    cell.requires_grad_()
    module = PeriodicGeometry(pair_chunk_size=5, activation_checkpointing=checkpoint)
    actual = module.from_cell(x, cell, mask)
    assert actual.geometry_ok.tolist() == [True, True, False, False]
    weights = torch.randn(actual.E.shape, dtype=dtype, generator=torch.Generator().manual_seed(81))
    grads = torch.autograd.grad((actual.E * weights).sum(), (x, cell))
    expected_outputs, expected_x, expected_cell = [], [], []
    for i in range(4):
        xi, ci = x[i : i + 1].detach().requires_grad_(), cell[i : i + 1].detach().requires_grad_()
        out = module.from_cell(xi, ci, mask[i : i + 1])
        dx, dc = torch.autograd.grad((out.E * weights[i : i + 1]).sum(), (xi, ci))
        expected_outputs.append(out.E)
        expected_x.append(dx)
        expected_cell.append(dc)
    torch.testing.assert_close(actual.E, torch.cat(expected_outputs))
    torch.testing.assert_close(grads[0], torch.cat(expected_x))
    torch.testing.assert_close(grads[1], torch.cat(expected_cell))
    assert torch.isfinite(grads[0]).all() and torch.isfinite(grads[1]).all()
    assert grads[0][2:].count_nonzero() == 0 and grads[1][2:].count_nonzero() == 0


def test_batched_search_runs_once_and_uses_exact_image_padding(monkeypatch):
    """All pair bounds are checked once, even when moments span many small chunks."""
    module = PeriodicGeometry(pair_chunk_size=3)
    x = torch.tensor([[[0.1, 0, 0], [0.9, 0, 0]]] * 4, dtype=torch.float64)
    cell = torch.eye(3, dtype=x.dtype)[None].repeat(4, 1, 1)
    mask = torch.ones(4, 2, dtype=torch.bool)
    cell[2] *= 0.01  # Entire sample rejected; no partial candidate truncation.
    mask[3] = False
    original_bounds, original_enumerate = module.image_search.bounds, module.image_search.enumerate
    bound_calls, slots, pairs = [], [], []

    def bounds(*args, **kwargs):
        bound_calls.append(args[0].shape)
        return original_bounds(*args, **kwargs)

    def enumerate_images(bounds, **kwargs):
        shifts, valid = original_enumerate(bounds, **kwargs)
        slots.append(shifts.shape[1])
        pairs.append(shifts.shape[0])
        assert torch.equal(valid.sum(-1), bounds.counts)
        return shifts, valid

    monkeypatch.setattr(module.image_search, "bounds", bounds)
    monkeypatch.setattr(module.image_search, "enumerate", enumerate_images)
    result = module.from_cell(x, cell, mask)
    assert bound_calls == [torch.Size([4, 4, 3])]
    assert slots and all(width == 1 for width in slots)
    assert sum(pairs) == 8 and max(pairs) <= 3
    assert result.geometry_ok.tolist() == [True, True, False, False]
    for i in range(2):
        torch.testing.assert_close(result.E[i], brute_force(x[i], cell[i]))


def test_entirely_unavailable_batch_does_not_enumerate_images(monkeypatch):
    module = PeriodicGeometry()

    def unexpected(*args, **kwargs):
        pytest.fail("Rejected samples must not launch image/moment computation")

    monkeypatch.setattr(module.image_search, "enumerate", unexpected)
    x = torch.zeros(3, 2, 3, requires_grad=True)
    cell = (torch.eye(3)[None].repeat(3, 1, 1) * 0.01).requires_grad_()
    out = module.from_cell(x, cell, torch.ones(3, 2))
    assert not out.geometry_ok.any()
    out.E.sum().backward()
    assert x.grad.count_nonzero() == 0 and cell.grad.count_nonzero() == 0


@pytest.mark.parametrize("side", [0.55, 1.5])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_image_budget_limits_chunks_without_dropping_pairs_or_gradients(
    side, checkpoint, monkeypatch
):
    generator = torch.Generator().manual_seed(97)
    x = (torch.rand(2, 9, 3, generator=generator, dtype=torch.float64) * side).requires_grad_()
    cell = (torch.eye(3, dtype=x.dtype)[None].repeat(2, 1, 1) * side).requires_grad_()
    mask = torch.ones(2, 9, dtype=torch.bool)
    mask[0, -2:] = False
    module = PeriodicGeometry(
        pair_chunk_size=32, image_chunk_size=64, activation_checkpointing=checkpoint
    )
    calls = []
    original = module.image_search.enumerate

    def enumerate_images(bounds, **kwargs):
        shifts, valid = original(bounds, **kwargs)
        calls.append(shifts.shape[:2])
        assert torch.equal(valid.sum(-1), bounds.counts)
        return shifts, valid

    monkeypatch.setattr(module.image_search, "enumerate", enumerate_images)
    actual = module.from_cell(x, cell, mask)
    expected = PeriodicGeometry(pair_chunk_size=3).from_cell(x, cell, mask)
    assert actual.geometry_ok.all()
    assert calls and all(pairs <= 32 and pairs * slots <= 64 for pairs, slots in calls)
    torch.testing.assert_close(actual.E, expected.E, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual.rho, expected.rho, rtol=1e-12, atol=1e-12)
    weights = torch.randn(actual.E.shape, generator=generator, dtype=x.dtype)
    a_grads = torch.autograd.grad((actual.E * weights).sum() + actual.rho.sum(), (x, cell))
    b_grads = torch.autograd.grad((expected.E * weights).sum() + expected.rho.sum(), (x, cell))
    for a, b in zip(a_grads, b_grads):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-11)


def test_geometry_chunk_policy_grows_only_when_image_work_allows():
    from src.models.g1.geometry.chunking import geometry_chunk_size

    assert geometry_chunk_size(32768, 524288, 1) == 32768
    assert geometry_chunk_size(32768, 524288, 125) == 4194
    assert geometry_chunk_size(4096, 524288, 1) == 4096
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            PeriodicGeometry(image_chunk_size=invalid)
