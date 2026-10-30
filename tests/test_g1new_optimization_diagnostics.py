"""Candidate benchmarks must gate the actual mode and use a matched control."""

import argparse
import json
from types import SimpleNamespace

import pytest
import torch

from scripts import benchmark_g1new_execution as runner


def arguments(**kwargs):
    values = dict(rounds=[2], epochs=2, repeats=2, offline=True, dry_run=False,
                  activation_checkpointing=False, include_reference=False,
                  lmdb_dir="/tmp/bwdb", candidate=[])
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_combinations_are_explicit_and_canonical():
    assert runner.parse_candidate("all") == runner.OPTIMIZATIONS
    assert runner.parse_candidate("merge_search,inplace_edge_sum,merge_search") == (
        "inplace_edge_sum", "merge_search"
    )
    with pytest.raises(argparse.ArgumentTypeError):
        runner.parse_candidate("unknown")


def test_commands_match_checkpoint_policy_and_disable_unrequested_flags(tmp_path):
    options = runner.parse_candidate("checkpoint_relation_inputs,merge_search")
    candidate = runner.command(tmp_path, 2, False, 2, "/tmp/bwdb", True, candidate=options)
    control = runner.command(tmp_path, 2, False, 2, "/tmp/bwdb", True,
                             activation_checkpointing=True)
    prefix = "model.flow_model_args.g1new.module."
    assert prefix + "activation_checkpointing=true" in candidate
    assert prefix + "activation_checkpointing=true" in control
    assert "++" + prefix + "checkpoint_relation_inputs=true" in candidate
    assert "++" + prefix + "checkpoint_relation_inputs=false" in control
    assert "++" + prefix + "geometry.merge_search=true" in candidate
    assert "++" + prefix + "geometry.merge_search=false" in control
    assert "++" + prefix + "inplace_edge_sum=false" in candidate
    assert "++" + prefix + "stream_edge_pooling=false" in candidate
    for cmd in (candidate, control):
        assert "++trainer.precision=32-true" in cmd
        assert "trainer.accumulate_grad_batches=2" in cmd
        assert "data.dynamic.max_batch_size=32" in cmd
    with pytest.raises(ValueError, match="optimized=true"):
        runner.command(tmp_path, 2, True, 2, "/tmp/bwdb", True, candidate=options)


@pytest.mark.parametrize("failure", [False, True])
def test_each_candidate_requires_its_gate_before_training(tmp_path, monkeypatch, failure):
    events = []
    selections = [runner.parse_candidate(name) for name in (
        "inplace_edge_sum", "checkpoint_relation_inputs", "merge_search"
    )]

    def gate(cmd, **kwargs):
        name = cmd[cmd.index("--candidate") + 1]
        assert kwargs["check"]
        assert "--verify-only" in cmd
        events.append(("gate", name))
        if failure and name == "checkpoint_relation_inputs":
            raise runner.subprocess.CalledProcessError(1, cmd)
        return SimpleNamespace(returncode=0)

    def train(cmd, directory):
        events.append(("train", directory.name))

    monkeypatch.setattr(runner.subprocess, "run", gate)
    monkeypatch.setattr(runner, "run", train)
    args = arguments(candidate=selections)
    if failure:
        with pytest.raises(runner.subprocess.CalledProcessError):
            runner.benchmark_candidates(tmp_path, args)
        assert events == [
            ("gate", "inplace_edge_sum"), ("train", "rounds2_optimized"),
            ("train", "rounds2_candidate_inplace_edge_sum"),
            ("gate", "checkpoint_relation_inputs"),
        ]
    else:
        runner.benchmark_candidates(tmp_path, args)
        assert events == [
            ("gate", "inplace_edge_sum"), ("train", "rounds2_optimized"),
            ("train", "rounds2_candidate_inplace_edge_sum"),
            ("gate", "checkpoint_relation_inputs"), ("train", "rounds2_optimized_checkpointed"),
            ("train", "rounds2_candidate_checkpoint_relation_inputs"),
            ("gate", "merge_search"), ("train", "rounds2_candidate_merge_search"),
        ]
        plan = json.loads((tmp_path / "benchmark_plan.json").read_text())
        assert plan["rounds2_candidate_checkpoint_relation_inputs"]["control"] == "rounds2_optimized_checkpointed"
        assert plan["rounds2_candidate_checkpoint_relation_inputs"]["activation_checkpointing"]
        assert plan["rounds2_candidate_merge_search"]["control"] == "rounds2_optimized"


@pytest.mark.parametrize("name", [*runner.OPTIMIZATIONS, "all"])
def test_candidate_gate_checks_training_backward_and_multiple_chunks(tmp_path, monkeypatch, name):
    # Real CPU forward/backward protects checkpoint wiring and exact edge order;
    # server CUDA uses the same gate without any relaxed tolerance.
    import torch.utils.checkpoint as checkpoint_module

    calls = []
    checkpoint = checkpoint_module.checkpoint

    def measured_checkpoint(*args, **kwargs):
        calls.append(True)
        return checkpoint(*args, **kwargs)

    monkeypatch.setattr(checkpoint_module, "checkpoint", measured_checkpoint)
    threads = torch.get_num_threads()
    precision = torch.get_float32_matmul_precision()
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    candidate = runner.parse_candidate(name)
    try:
        runner.verify(tmp_path, 2, "cpu", repeats=2, candidate=candidate)
    finally:
        torch.set_num_threads(threads)
        torch.set_float32_matmul_precision(precision)
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
    first = json.loads((tmp_path / "numerical_rounds2.json").read_text())
    repeat = json.loads((tmp_path / "numerical_rounds2_repeat02.json").read_text())
    assert first["passed"] and repeat["passed"] and repeat["self_repeat_passed"]
    assert first["identical_edge_order"] and first["identical_edge_sets"]
    assert first["edge_count"] > first["edge_chunk_size"]
    assert first["model_training"] and first["control"] == "current_optimized"
    assert first["candidate"] == list(candidate)
    assert first["rtol"] == 0.003 and first["atol"] == 0.0002
    should_checkpoint = "checkpoint_relation_inputs" in candidate
    assert first["activation_checkpointing"] == should_checkpoint
    assert bool(calls) == should_checkpoint
    assert first["fixture_sha256"] == repeat["fixture_sha256"]
    assert first["initial_state_sha256"] == repeat["initial_state_sha256"]
    assert any(row["name"] == "input.condition" for row in first["tensors"])
    assert any("relations.network" in row["name"] for row in first["tensors"])


def test_summary_uses_checkpoint_matched_control_and_checks_data_order(tmp_path):
    plan = {
        "rounds2_optimized_checkpointed": dict(control=None, candidate=[], activation_checkpointing=True),
        "rounds2_candidate_checkpoint_relation_inputs": dict(
            control="rounds2_optimized_checkpointed", candidate=["checkpoint_relation_inputs"],
            activation_checkpointing=True
        ),
    }
    (tmp_path / "benchmark_plan.json").write_text(json.dumps(plan))
    for name in plan:
        directory = tmp_path / name / "performance"
        directory.mkdir(parents=True)
        for rank in range(8):
            record = dict(epoch=0, rank=rank, world_size=8, batches_sha256="indices",
                          shapes_sha256="shapes", parameter_count=39479632, precision="32-true",
                          float32_matmul_precision="high", cuda_matmul_allow_tf32=True,
                          cudnn_allow_tf32=True, accumulate_grad_batches=2, total_batches=651,
                          train_seconds=100 if "candidate" in name else 110,
                          peak_allocated_bytes=10 * 2**30, peak_reserved_bytes=12 * 2**30)
            (directory / f"rank{rank}_epoch0.json").write_text(json.dumps(record))
    runner.summarize(tmp_path, 1)
    rows = json.loads((tmp_path / "summary.json").read_text())
    actual = next(r for r in rows if r["candidate"])
    assert actual["time_change_percent"] == pytest.approx(-100 / 11)
    assert actual["allocated_change_gib"] == 0
    bad = tmp_path / "rounds2_candidate_checkpoint_relation_inputs/performance/rank0_epoch0.json"
    record = json.loads(bad.read_text())
    record["batches_sha256"] = "different"
    bad.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="invalid A/B"):
        runner.summarize(tmp_path, 1)


def test_candidate_profiles_continue_after_failure_without_relaxing_gate(tmp_path, monkeypatch):
    from pathlib import Path

    selections = [runner.parse_candidate("inplace_edge_sum"),
                  runner.parse_candidate("checkpoint_relation_inputs")]
    visited = []

    def fake_profile(cmd, **kwargs):
        profile = cmd[cmd.index("--diagnostic-profile") + 1]
        candidate = runner.parse_candidate(cmd[cmd.index("--candidate") + 1])
        name = runner.candidate_name(candidate)
        visited.append((name, profile))
        directory = Path(cmd[cmd.index("--output") + 1]) / name
        directory.mkdir(parents=True)
        fails = profile == "training" and candidate == selections[0]
        rows = [dict(case=case, repeat=i + 1, passed=not fails,
                     failed_tensors=int(fails), failed_elements=int(fails),
                     self_repeat_passed=not fails, self_failed_elements=int(fails))
                for case in ("reference_repeat", "optimized") for i in range(2)]
        (directory / "diagnostic_summary_rounds2.json").write_text(json.dumps(rows))
        (directory / "numerical_rounds2.json").write_text(json.dumps(dict(
            fixture_sha256="same-inputs", initial_state_sha256="same-weights",
            matmul_precision="high", matmul_tf32=True, cudnn_tf32=True,
            rtol=0.003, atol=0.0002, edge_chunk_size=512,
            activation_checkpointing=runner.candidate_checkpointing(candidate)
        )))
        return SimpleNamespace(returncode=int(fails))

    monkeypatch.setattr(runner.subprocess, "run", fake_profile)
    with pytest.raises(RuntimeError, match="profiles completed with failures"):
        runner.diagnose_candidates(tmp_path, arguments(candidate=selections, verify_device="cpu"))
    assert visited == [(runner.candidate_name(candidate), profile)
                       for candidate in selections for profile in ("training", "deterministic")]
    report = json.loads((tmp_path / "diagnostic_profiles.json").read_text())
    assert report["identical_fixtures_and_precision"]
    assert report["identical_inputs_and_weights_across_candidates"]
    assert [r["returncode"] for r in report["runs"]] == [1, 0, 0, 0]
    assert all(r["complete"] for r in report["runs"])
