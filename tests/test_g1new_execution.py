"""Execution-only G1-New changes against the retained streaming implementation."""

from copy import deepcopy

import pytest
import torch
from hydra.utils import instantiate

from test_g1new import branch_inputs, edge_keys, new_config, perturb, run_branch
from src.models.g1new.bridge import G1NewBridge
from src.models.g1new.geometry import LocalGeometry, SearchBudgetExceeded
from src.models.g1new.geometry import LocalPairs
from src.models.g1new import G1NewModule
from src.models.g1new.relations import RelationEncoder


@pytest.fixture(autouse=True, scope="module")
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("rounds", [1, 2, 3])
@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_reference_outputs_and_all_gradients(rounds, checkpointing, dtype):
    torch.manual_seed(197)
    config = new_config(checkpoint=checkpointing, chunk=7)["module"]
    config.update(rounds=rounds, optimized=True)
    actual = instantiate(config).to(dtype).train()
    config["optimized"] = False
    reference = instantiate(config).to(dtype).train()
    perturb(actual)
    reference.load_state_dict(actual.state_dict(), strict=True)
    assert list(actual.state_dict()) == list(reference.state_dict())
    assert sum(p.numel() for p in actual.parameters()) == sum(
        p.numel() for p in reference.parameters()
    )
    H, t, feats, x, lattice = branch_inputs(dtype)
    left = [v.clone().requires_grad_() for v in (H, t, x, lattice)]
    right = [v.clone().requires_grad_() for v in (H, t, x, lattice)]
    a = run_branch(actual, left[0], left[1], feats, left[2], left[3])
    b = run_branch(reference, right[0], right[1], feats, right[2], right[3])
    tolerance = dict(rtol=5e-5, atol=5e-6) if dtype == torch.float32 else dict(rtol=1e-8, atol=1e-9)
    torch.testing.assert_close(a, b, **tolerance)
    weight = torch.randn_like(a)
    (a * weight).sum().backward()
    (b * weight).sum().backward()
    for p, q in zip(left, right):
        torch.testing.assert_close(p.grad, q.grad, **tolerance)
    for (name, p), (other_name, q) in zip(actual.named_parameters(), reference.named_parameters()):
        assert name == other_name and p.grad is not None and q.grad is not None
        torch.testing.assert_close(p.grad, q.grad, msg=name, **tolerance)


@pytest.mark.parametrize("rounds", [1, 2, 3])
def test_four_sites_shared_geometry_features_match_reference(rounds):
    torch.manual_seed(551)
    config = new_config(checkpoint=False, chunk=7)
    config.pop("enabled")
    config["insert_after"] = [6, 12, 18, 24]
    config["module"].update(rounds=rounds, optimized=True)
    actual = G1NewBridge(h_dim=16, depth=24, **config).double()
    config["module"]["optimized"] = False
    reference = G1NewBridge(h_dim=16, depth=24, **config).double()
    perturb(actual)
    reference.load_state_dict(actual.state_dict(), strict=True)
    H, t, feats, x, lattice = branch_inputs(torch.float64)
    args = [[z.clone().requires_grad_() for z in (H, t, x, lattice)] for _ in range(2)]
    outputs = []
    for model, (hidden, time, coords, cell) in zip((actual, reference), args):
        context = model.prepare(feats, coords, cell)
        assert (context.relation_features is not None) == model.first_bridge.optimized
        for position in model.insert_after:
            hidden = model(position, hidden, time, context)
        outputs.append(hidden)
    torch.testing.assert_close(*outputs, rtol=1e-8, atol=1e-9)
    weight = torch.randn_like(H)
    for output in outputs:
        (output * weight).sum().backward()
    for a, b in zip(*args):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-7, atol=1e-8)
    for (name, a), (_, b) in zip(actual.named_parameters(), reference.named_parameters()):
        torch.testing.assert_close(a.grad, b.grad, msg=name, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("chunk", [1, 7, 65536])
def test_periodic_edge_sets_and_live_geometry_match_across_chunks(chunk):
    torch.manual_seed(339)
    _, _, feats, x, lattice = branch_inputs(torch.float64)
    lattice[:, :3] = torch.tensor([0.55, 0.75, 1.1])
    lattice[:, 3:] = torch.tensor([1.25, 1.45, 1.1])
    feats["ref_element"][0, 1] = 1
    feats["bond_types"][0, 0, 1] = feats["bond_types"][0, 1, 0] = 1
    actual = LocalGeometry(pair_chunk_size=chunk, image_chunk_size=19)
    reference = LocalGeometry(pair_chunk_size=5, image_chunk_size=7)
    reference.optimized = False
    a, b = actual(x, lattice, feats), reference(x, lattice, feats)
    assert edge_keys(a) == edge_keys(b)
    assert a.atom.numel() == b.atom.numel() == len(edge_keys(a))
    for edges in (a, b):
        assert torch.isfinite(edges.distance).all()
        assert torch.isfinite(edges.direction).all()
    # Compare by identity, not enumeration/chunk order.
    def keyed(edges):
        keys = list(zip(edges.atom.tolist(), edges.env.tolist(), map(tuple, edges.shift.tolist())))
        return {k: torch.cat((edges.distance[i:i+1], edges.direction[i], edges.omega[i:i+1]))
                for i, k in enumerate(keys)}
    aa, bb = keyed(a), keyed(b)
    for key in aa:
        torch.testing.assert_close(aa[key], bb[key], rtol=0, atol=0)


@pytest.mark.parametrize("optimized", [False, True])
def test_bond_validation_has_reference_support(optimized):
    _, _, feats, x, lattice = branch_inputs(torch.float64)
    geo = LocalGeometry(pair_chunk_size=7)
    geo.optimized = optimized
    bad = deepcopy(feats)
    bad["bond_types"][0, 0, 0] = 7
    with pytest.raises(ValueError, match="bond_types"):
        geo(x, lattice, bad)
    # An invalid geometry sample must not introduce new bond validation failures.
    lattice[0, 0] = -1
    geo(x, lattice, bad)


@pytest.mark.parametrize("optimized", [False, True])
def test_large_search_cannot_bypass_resource_budgets(optimized):
    _, _, feats, x, lattice = branch_inputs(torch.float64)
    lattice[:, :3] = 0.01
    geo = LocalGeometry(max_search_candidates=100)
    geo.optimized = optimized
    with pytest.raises(SearchBudgetExceeded, match="budget"):
        geo(x, lattice, feats)


def synthetic_edges(count, atoms=32, anchors=16, dtype=torch.float32):
    """Many image edges with all six bond classes; no geometry-search timing."""
    return LocalPairs(
        torch.arange(anchors), torch.arange(count) % atoms,
        (torch.arange(count) // atoms) % anchors,
        torch.zeros(count, 3, dtype=torch.long), torch.arange(count) % 6,
        torch.rand(count, dtype=dtype).requires_grad_(),
        torch.randn(count, 3, dtype=dtype).requires_grad_(),
        torch.ones(count, dtype=dtype), torch.rand(count, dtype=dtype).requires_grad_(),
        torch.tensor([True]),
    )


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_all_six_bond_projections_match_dense_encoder(checkpointing, dtype):
    torch.manual_seed(631)
    actual = RelationEncoder().to(dtype)
    reference = deepcopy(actual)
    reference.optimized = False
    edges = synthetic_edges(61, dtype=dtype)
    other = edges._replace(distance=edges.distance.detach().clone().requires_grad_(),
                           direction=edges.direction.detach().clone().requires_grad_())
    a = actual(edges, 13, checkpointing)
    b = reference(other, 7, checkpointing)
    tolerance = dict(rtol=5e-5, atol=5e-6) if dtype == torch.float32 else dict(rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(a, b, **tolerance)
    weight = torch.randn_like(a)
    (a * weight).sum().backward()
    (b * weight).sum().backward()
    for p, q in zip((edges.distance, edges.direction), (other.distance, other.direction)):
        torch.testing.assert_close(p.grad, q.grad, **tolerance)
    for (name, p), (_, q) in zip(actual.named_parameters(), reference.named_parameters()):
        assert p.grad is not None and q.grad is not None
        torch.testing.assert_close(p.grad, q.grad, msg=name, **tolerance)


def test_checkpoint_does_not_save_full_edge_hidden_heads_or_values():
    torch.manual_seed(819)
    count, atoms = 1031, 32
    totals = []
    for checkpointing in (False, True):
        model = G1NewModule(rounds=3, edge_chunk_size=257,
                            activation_checkpointing=checkpointing).train()
        edges = synthetic_edges(count, atoms)
        H = torch.randn(1, atoms, 256, requires_grad=True)
        t = torch.tensor([0.4], requires_grad=True)
        condition = torch.randn(1, atoms, 128, requires_grad=True)
        exclude = {v.untyped_storage().data_ptr()
                   for v in (*model.parameters(), *model.buffers(), H, t, condition, *edges)}
        saved = {}

        def pack(value):
            storage = value.untyped_storage()
            if storage.data_ptr() not in exclude:
                saved[storage.data_ptr()] = storage.nbytes()
            return value

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
            output = model(H, t, {"atom_pad_mask": torch.ones(1, atoms, dtype=torch.bool)},
                           edges, condition)
        totals.append(sum(saved.values()))
        if checkpointing:
            # Check backing storage, not view shape: small checkpoint slices of
            # a full T*128 buffer would otherwise hide the earlier regression.
            for width in (64, 65, 128):
                assert count * width * 4 not in saved.values()
        output.square().sum().backward()
        assert all(p.grad is not None for name, p in model.named_parameters()
                   if not name.startswith("condition_encoder."))
    assert totals[1] < totals[0] * 0.5


@pytest.mark.parametrize("checkpointing", [False, True])
def test_three_adam_updates_keep_parameter_and_optimizer_states(checkpointing):
    torch.manual_seed(381)
    config = new_config(checkpoint=checkpointing, chunk=7)["module"]
    config.update(rounds=3, optimized=True)
    actual = instantiate(config).double()
    config["optimized"] = False
    reference = instantiate(config).double()
    perturb(actual)
    reference.load_state_dict(actual.state_dict(), strict=True)
    optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0, foreach=False)
                  for m in (actual, reference)]
    for _ in range(3):
        H, t, feats, x, lattice = branch_inputs(torch.float64)
        weight = torch.randn_like(H)
        for model, optimizer in zip((actual, reference), optimizers):
            optimizer.zero_grad(set_to_none=True)
            output = run_branch(model, H, t, feats, x, lattice)
            (output * weight).sum().backward()
            optimizer.step()
        for (name, p), (_, q) in zip(actual.named_parameters(), reference.named_parameters()):
            torch.testing.assert_close(p, q, msg=name, rtol=1e-7, atol=1e-8)
            a, b = optimizers[0].state[p], optimizers[1].state[q]
            assert set(a) == set(b)
            for key in a:
                torch.testing.assert_close(a[key], b[key], msg=name + "." + key,
                                           rtol=1e-7, atol=1e-8)


def test_numerical_report_does_not_replace_elementwise_gate_with_global_l2():
    from scripts.benchmark_g1new_execution import compare_tensors

    row = compare_tensors("cancellation", torch.tensor([1000., 0.]),
                          torch.tensor([1000., 0.0003]))
    assert row["relative_l2"] < 0.003
    assert not row["passed"] and row["failed_elements"] == 1
    assert row["worst"]["flat_index"] == 1
    assert row["worst"]["allowed_error"] == pytest.approx(0.0002)
    assert row["max_tolerance_ratio"] == pytest.approx(1.5)


@pytest.mark.parametrize("actual", [[1.001, 0.0001], [1.01, 0.], [float("nan"), 0.],
                                    [float("inf"), 0.]])
def test_numerical_report_preserves_tolerance_and_rejects_nonfinite(actual):
    import json
    from scripts.benchmark_g1new_execution import compare_tensors

    reference, value = torch.tensor([1., 0.]), torch.tensor(actual)
    row = compare_tensors("test", reference, value)
    expected = bool(torch.allclose(value, reference, rtol=0.003, atol=0.0002)
                    and torch.isfinite(value).all())
    assert row["passed"] == expected
    json.dumps(row, allow_nan=False)


@pytest.mark.parametrize("changes", [dict(heads=True), dict(bonds=True), dict(environment=True),
                                     dict(heads=True, bonds=True, environment=True)])
def test_diagnostic_factors_preserve_equations_parameters_and_precision(changes):
    from scripts.benchmark_g1new_execution import diagnostic_reassociation

    torch.manual_seed(643)
    config = new_config(checkpoint=False, chunk=7)
    config.pop("enabled")
    config["insert_after"] = [6, 12, 18, 24]
    expected = G1NewBridge(h_dim=16, depth=24, **config).double()
    perturb(expected)
    actual = deepcopy(expected)
    policy = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
              torch.backends.cudnn.allow_tf32)
    diagnostic_reassociation(actual, **changes)
    assert policy == (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                      torch.backends.cudnn.allow_tf32)
    for name, parameter in expected.state_dict().items():
        assert torch.equal(parameter, actual.state_dict()[name])
    H, t, feats, x, lattice = branch_inputs(torch.float64)
    weight = torch.randn_like(H)
    outputs = []
    for model in (expected, actual):
        context = model.prepare(feats, x, lattice)
        hidden = H
        for position in model.insert_after:
            hidden = model(position, hidden, t, context)
        outputs.append(hidden)
        (hidden * weight).sum().backward()
    torch.testing.assert_close(*outputs, rtol=1e-8, atol=1e-9)
    for (name, p), (_, q) in zip(expected.named_parameters(), actual.named_parameters()):
        torch.testing.assert_close(p.grad, q.grad, msg=name, rtol=1e-7, atol=1e-8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires actual CUDA FP32/high/TF32")
def test_cuda_numerical_gate_under_training_precision(tmp_path):
    from scripts.benchmark_g1new_execution import verify

    verify(tmp_path, 2, "cuda")


@pytest.mark.parametrize("profile", ["training", "deterministic"])
def test_diagnostic_policy_restores_flags_even_on_error(profile, monkeypatch):
    import os
    from scripts.benchmark_g1new_execution import diagnostic_policy

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    flags = (torch.are_deterministic_algorithms_enabled(),
             torch.is_deterministic_algorithms_warn_only_enabled(),
             torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    precision = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                 torch.backends.cudnn.allow_tf32)
    with pytest.raises(RuntimeError, match="test exit"):
        with diagnostic_policy(profile):
            if profile == "deterministic":
                assert torch.are_deterministic_algorithms_enabled()
                assert not torch.is_deterministic_algorithms_warn_only_enabled()
                assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
            assert precision == (torch.get_float32_matmul_precision(),
                                 torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
            raise RuntimeError("test exit")
    assert flags == (torch.are_deterministic_algorithms_enabled(),
                     torch.is_deterministic_algorithms_warn_only_enabled(),
                     torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_repeated_probes_record_real_self_comparisons_and_unchanged_fixture(tmp_path):
    import json
    from scripts.benchmark_g1new_execution import verify

    verify(tmp_path, 2, "cpu", diagnose=True, repeats=2)
    rows = json.loads((tmp_path / "diagnostic_summary_rounds2.json").read_text())
    assert len(rows) == 12
    assert all(r["passed"] for r in rows)
    first = json.loads((tmp_path / "numerical_rounds2.json").read_text())
    second = json.loads((tmp_path / "numerical_rounds2_repeat02.json").read_text())
    assert first["self_repeat_passed"] is None  # Do not claim equality by comparing a tensor to itself.
    assert second["self_repeat_passed"]
    assert all(r["max_abs"] == 0 for r in second["self_repeat_tensors"])
    assert first["fixture_sha256"] == second["fixture_sha256"]
    assert first["initial_state_sha256"] == second["initial_state_sha256"]
    assert first["rtol"] == second["rtol"] == 0.003
    assert first["atol"] == second["atol"] == 0.0002


def test_diagnostic_profiles_continue_after_first_failure(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from pathlib import Path
    from scripts import benchmark_g1new_execution as runner

    visited = []
    def fake_run(cmd, **kwargs):
        profile = cmd[cmd.index("--diagnostic-profile") + 1]
        visited.append(profile)
        directory = Path(cmd[cmd.index("--output") + 1])
        directory.mkdir()
        bad = profile == "training"
        values = [dict(case=case, repeat=trial + 1, passed=not bad,
                       failed_tensors=int(bad), failed_elements=int(bad),
                       self_failed_elements=int(bad))
                  for case in ("reference_repeat", "optimized", "heads_reassociated",
                               "bonds_preprojected", "environment_reassociated", "legacy_all")
                  for trial in range(2)]
        (directory / "diagnostic_summary_rounds2.json").write_text(json.dumps(values))
        (directory / "numerical_rounds2.json").write_text(json.dumps(dict(
            fixture_sha256="same-inputs", initial_state_sha256="same-weights",
            matmul_precision="high", matmul_tf32=True, cudnn_tf32=True, rtol=0.003, atol=0.0002
        )))
        return SimpleNamespace(returncode=int(bad))

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    args = SimpleNamespace(rounds=[2], repeats=2, verify_device="cpu", dry_run=False)
    with pytest.raises(RuntimeError, match="profiles completed with failures"):
        runner.diagnose_profiles(tmp_path, args)
    assert visited == ["training", "deterministic"]
    result = json.loads((tmp_path / "diagnostic_profiles.json").read_text())
    assert result["identical_fixtures_and_precision"]
    assert all(r["complete"] for r in result["runs"])
    assert [r["returncode"] for r in result["runs"]] == [1, 0]
