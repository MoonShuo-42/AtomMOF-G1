"""G1-New-only correctness gate and full-epoch B32/a2 timing on the server.

No backbone, precision, batch, geometry cutoff or architecture ablations.
Default: four sites, two internal rounds, two complete epochs on eight GPUs.
Use --include-reference for a same-code, same-data old/new implementation pair.
The numerical gate uses deterministic algorithms in an isolated child process;
training retains its original execution policy and the same FP32/high/TF32.
Use --candidate NAME (repeatable) for individual optimization A/B runs, or
--candidate inplace_edge_sum,merge_search / --candidate all for combinations.
Candidates are compared with the existing optimized implementation using the
same checkpointing policy. These gates do not certify normal CUDA repeatability;
--diagnose-only checks both execution policies in separate processes.
"""

import argparse
import hashlib
import json
import math
import os
import shlex
import signal
import statistics
import subprocess
import sys
import tarfile
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OPTIMIZATIONS = ("inplace_edge_sum", "checkpoint_relation_inputs",
                 "stream_edge_pooling", "merge_search")


def parse_candidate(value):
    """A repeated argument means separate runs; commas mean one combined run."""
    names = OPTIMIZATIONS if value == "all" else tuple(value.replace("+", ",").split(","))
    if not names or any(name not in OPTIMIZATIONS for name in names):
        raise argparse.ArgumentTypeError(
            "Expected all or comma-separated names from " + ", ".join(OPTIMIZATIONS)
        )
    return tuple(name for name in OPTIMIZATIONS if name in names)


def candidate_name(candidate):
    return "candidate_" + "_".join(candidate)


def candidate_checkpointing(candidate, requested=False):
    # Input recomputation is inert without activation checkpointing. Never
    # silently compare a recomputed candidate with a non-recomputed control.
    return requested or "checkpoint_relation_inputs" in candidate


def runtime_options(candidate=()):
    return {name: name in candidate for name in OPTIMIZATIONS}


def compare_tensors(name, reference, actual, rtol=0.003, atol=0.0002):
    """Keep the elementwise gate; explain local failures hidden by an L2 average."""
    import torch

    # The acceptance test uses the original tensor dtype, exactly as allclose
    # did before. Higher precision below is only for reporting error magnitudes.
    finite = torch.isfinite(reference) & torch.isfinite(actual)
    bad = ~torch.isclose(actual, reference, rtol=rtol, atol=atol) | ~finite
    reference, actual = reference.double(), actual.double()
    delta = (actual - reference).abs()
    tolerance = atol + rtol * reference.abs()
    ratio = (delta / tolerance).masked_fill(~finite, float("inf"))

    def number(value):
        result = float(value)
        return result if math.isfinite(result) else None

    worst = None
    if delta.numel():
        index = int(ratio.flatten().argmax())
        worst = dict(flat_index=index, reference=number(reference.flatten()[index]),
                     actual=number(actual.flatten()[index]),
                     abs_error=number(delta.flatten()[index]),
                     allowed_error=number(tolerance.flatten()[index]))
    return dict(
        name=name, max_abs=number(delta.max()) if delta.numel() else 0,
        relative_l2=number(delta.norm() / reference.norm().clamp_min(1e-12)),
        elements=delta.numel(), failed_elements=int(bad.sum()),
        max_tolerance_ratio=number(ratio.max()) if delta.numel() else 0,
        all_finite=bool(finite.all()), worst=worst, passed=not bool(bad.any()),
    )


def diagnostic_reassociation(model, *, heads=False, bonds=False, environment=False):
    """Restore selected affine reorders on a disposable probe, never training.

    Test one factor at a time without changing tensor dtype, TF32 or tolerances.
    The combined case restores all three reassociations, keeping current pooling.
    It does not reconstruct every operator/layout in the historical commit.
    """
    from types import MethodType
    import torch
    from torch.nn import functional as F
    from src.models.g1new.layers import split_apply

    def composed_heads(self, heads):
        weight = heads.weight @ self.last.weight
        bias = F.linear(self.last.bias, heads.weight, heads.bias)
        return lambda hidden: F.linear(F.silu(hidden), weight, bias)

    def projected_bonds(self, edges, chunk_size, checkpointing, geometry_features=None):
        if geometry_features is None:
            geometry_features = self.geometry_features(edges)
        first = self.network[0]
        table = F.linear(self.bond_embedding.weight, first.weight[:, 20:])

        def encode(geometry, bond):
            hidden = F.linear(geometry, first.weight[:, :20], first.bias)
            hidden = hidden + F.embedding(bond, table)
            return self.network[2](self.network[1](hidden))

        return split_apply(encode, chunk_size, checkpointing, geometry_features, edges.bond)

    def reordered_prepare(self, features, anchors, edges=None, chi=None, *,
                          checkpointing=False, anchor_features=None):
        if checkpointing or edges is None:
            raise ValueError("Diagnostic reassociation is a no-checkpoint probe only")
        atom = self.messages.project(features, 0)
        if anchor_features is None:
            anchor_features = features.index_select(0, anchors)
        anchor = self.messages.project(anchor_features, 1)
        return (atom.index_select(0, edges.atom) + anchor.index_select(0, edges.env)
                + self.messages.project(chi, 3, bias=True))

    def reordered_update(self, state, projected, edges, chi, chunk_size, checkpointing, *,
                         head_map=None):
        if checkpointing:
            raise ValueError("Diagnostic reassociation is a no-checkpoint probe only")
        hidden = projected + self.messages.project(state, 2).index_select(0, edges.env)
        if head_map is None:
            head_map = self.messages.head_map(self.heads)
        scores = split_apply(head_map, chunk_size, False, hidden)
        pooled = self._pool(scores, edges.omega, edges.env, state.shape[0])
        mass, numerator = pooled
        u = numerator / (1 + mass[:, None])
        q = torch.cat((state, u, torch.log1p(mass)[:, None]), -1)
        return state + (mass / (1 + mass))[:, None] * self.update(self.update_norm(q))

    for site in model.bridges.values():
        if heads:
            for operator in (site.environment.messages, site.readback.messages):
                operator.head_map = MethodType(composed_heads, operator)
        if bonds:
            site.relations.forward = MethodType(projected_bonds, site.relations)
        if environment:
            site.environment.prepare = MethodType(reordered_prepare, site.environment)
            site.environment.forward = MethodType(reordered_update, site.environment)


@contextmanager
def diagnostic_policy(profile):
    """Change reproducibility, never dtype/TF32, in a diagnostic subprocess only."""
    if profile not in ("training", "deterministic"):
        raise ValueError("Unknown diagnostic profile")
    import torch

    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark, cudnn_deterministic = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    if profile == "deterministic" and torch.cuda.is_initialized():
        raise RuntimeError("Start a fresh diagnostic subprocess before enabling CUDA determinism")
    try:
        if profile == "deterministic":
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic
        if workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace


def tensor_fingerprint(tensors):
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}\n".encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def verify(directory, rounds, device, diagnose=False, repeats=1, *, candidate=(),
           activation_checkpointing=False):
    """Nonzero residuals, all four sites, live geometry and all changed weights.

    Conditions are supplied identically to isolate the modified local branch;
    the unchanged condition encoder is covered by the repository integration tests.
    CUDA uses exactly the training FP32/high/TF32 policy, not a double-only check.
    """
    from copy import deepcopy
    import torch
    from src.models.g1new.bridge import G1NewBridge

    if repeats < 1:
        raise ValueError("repeats must be positive")
    candidate = tuple(candidate)
    activation_checkpointing = candidate_checkpointing(candidate, activation_checkpointing)
    if any(name not in OPTIMIZATIONS for name in candidate):
        raise ValueError("Unknown candidate optimization")
    torch.manual_seed(271)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the server numerical gate")
    if device == "cpu":
        torch.set_num_threads(1)
    models = []
    for index, optimized in enumerate((False, True)):
        # Candidate A/B isolates only the requested runtime flags, including
        # matching checkpointing and chunk sizes on its current-optimized control.
        optimized = bool(candidate) or optimized
        options = runtime_options(candidate if index else ())
        config = dict(
            _target_="src.models.g1new.G1NewModule", optimized=optimized,
            rounds=rounds, activation_checkpointing=activation_checkpointing,
            # The synthetic fixture has ~2.4k edges: force multiple chunks in
            # candidate gates so streaming/recomputation boundaries are exercised.
            edge_chunk_size=512 if candidate else (65536 if optimized else 8192),
            geometry=dict(
                _target_="src.models.g1new.geometry.LocalGeometry",
                pair_chunk_size=65536 if optimized else 4096,
                image_chunk_size=262144 if optimized else 65536,
                merge_search=options.pop("merge_search"),
            ),
            **options,
        )
        models.append(G1NewBridge(h_dim=256, depth=24, module=config).to(device).train())
    with torch.no_grad():
        for site in models[0].bridges.values():
            site.output_projection.weight.normal_(std=0.03)
    models[1].load_state_dict(models[0].state_dict(), strict=True)
    assert list(models[0].state_dict()) == list(models[1].state_dict())
    assert sum(p.numel() for p in models[0].parameters()) == 1103312
    B, N = 2, 48
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    mask[1, -4:] = False
    elements = torch.full((B, N), 6, device=device)
    elements[:, ::5] = 1
    bonds = torch.zeros(B, N, N, dtype=torch.long, device=device)
    ids = torch.arange(N - 1, device=device)
    # Exercise every learned bond row, not only the unknown/single-bond rows.
    bonds[:, ids, ids + 1] = bonds[:, ids + 1, ids] = 1 + ids % 5
    feats = dict(atom_pad_mask=mask, ref_element=elements, bond_types=bonds,
                 block_index=torch.arange(N, device=device).expand(B, -1) // 8)
    source = [
        torch.randn(B, N, 256, device=device),
        torch.tensor([0.3, 0.7], device=device),
        torch.rand(B, N, 3, device=device) * 1.4,
        torch.tensor([[1.4, 1.5, 1.6, 1.2, 1.4, 1.5]], device=device).expand(B, -1),
        torch.randn(B, N, 128, device=device),
    ]
    weight = torch.randn_like(source[0])
    fixture = dict(feats)
    fixture.update({f"input_{i}": value for i, value in enumerate(source)})
    fixture["loss_weight"] = weight
    fixture_hash = tensor_fingerprint(fixture)
    state_hash = tensor_fingerprint(models[0].state_dict())

    def evaluate(model):
        model.zero_grad(set_to_none=True)
        H, t, x, lattice, c = values = [v.clone().requires_grad_() for v in source]
        context = model.prepare(feats, x, lattice, condition=c)
        edge = context.edges
        edge_table = list(zip(
            edge.atom.tolist(), edge.env.tolist(), map(tuple, edge.shift.tolist()),
            edge.bond.tolist(),
        ))
        output = H
        for position in model.insert_after:
            output = model(position, output, t, context)
        (output * weight).sum().backward()
        tensors = {"output": output.detach()}
        tensors.update({"input." + name: value.grad for name, value in zip(
            ("H", "t", "coords", "lattice", "condition"), values
        )})
        tensors.update({name: p.grad for name, p in model.named_parameters()
                        if ".condition_encoder." not in name})
        if any(value is None for value in tensors.values()):
            raise RuntimeError("Disconnected input/parameter in the numerical gate")
        return {key: value.detach().cpu().clone() for key, value in tensors.items()}, edge_table

    reference, reference_edges = evaluate(models[0])
    # A reference that fails against itself cannot certify the optimized path.
    cases = [("reference_repeat", {}), ("optimized", {})]
    if diagnose and not candidate and not activation_checkpointing:
        cases = [*cases,
                 ("heads_reassociated", dict(heads=True)),
                 ("bonds_preprojected", dict(bonds=True)),
                 ("environment_reassociated", dict(environment=True)),
                 ("legacy_all", dict(heads=True, bonds=True, environment=True))]
    # TF32 and changed reduction order need a measured, explicit tolerance.
    # Report every tensor and fail; never widen tolerances or silently fall back.
    outcomes = []
    for case, changes in cases:
        model = models[0] if case == "reference_repeat" else deepcopy(models[1])
        if changes:
            diagnostic_reassociation(model, **changes)
        first, first_edges = (reference, reference_edges) if case == "reference_repeat" else (None, None)
        for trial in range(repeats):
            actual, edge_table = evaluate(model)
            # A self-comparison is only meaningful after an independent evaluation.
            self_rows = None if first is None else [
                compare_tensors(name, value, actual[name]) for name, value in first.items()
            ]
            self_passed = None if self_rows is None else (
                (first_edges == edge_table if candidate else
                 sorted(first_edges) == sorted(edge_table))
                and all(x["passed"] for x in self_rows)
            )
            report = dict(rounds=rounds, case=case, repeat=trial + 1, device=device,
                          rtol=0.003, atol=0.0002,
                          matmul_precision=torch.get_float32_matmul_precision(),
                          matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                          cudnn_tf32=torch.backends.cudnn.allow_tf32,
                          deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                          deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                          cudnn_benchmark=torch.backends.cudnn.benchmark,
                          cudnn_deterministic=torch.backends.cudnn.deterministic,
                          cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                          fixture_sha256=fixture_hash, initial_state_sha256=state_hash,
                          torch_version=torch.__version__, cuda_version=torch.version.cuda,
                          gpu_name=torch.cuda.get_device_name() if device == "cuda" else None,
                          candidate=list(candidate), candidate_options=runtime_options(candidate),
                          control="current_optimized" if candidate else "reference",
                          activation_checkpointing=activation_checkpointing,
                          model_training=model.training,
                          edge_chunk_size=next(iter(model.bridges.values())).edge_chunk_size,
                          shape=[B, N], edge_count=len(edge_table),
                          identical_edge_order=reference_edges == edge_table,
                          identical_edge_sets=sorted(reference_edges) == sorted(edge_table),
                          self_repeat_passed=self_passed, self_repeat_tensors=self_rows,
                          tensors=[compare_tensors(name, value, actual[name])
                                   for name, value in reference.items()])
            report["passed"] = (report["identical_edge_sets"]
                                and (not candidate or report["identical_edge_order"])
                                and all(x["passed"] for x in report["tensors"]))
            suffix = "" if case == "optimized" else "_" + case
            if trial:
                suffix += f"_repeat{trial + 1:02d}"
            (directory / f"numerical_rounds{rounds}{suffix}.json").write_text(
                json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
            )
            failures = [x["name"] for x in report["tensors"] if not x["passed"]]
            count = sum(x["failed_elements"] for x in report["tensors"])
            print(f"Numerical check rounds={rounds}, case={case}, repeat={trial + 1}: "
                  f"passed={report['passed']}, failed_elements={count}, "
                  f"self_repeat_passed={self_passed}, failures={failures}", flush=True)
            outcomes.append(dict(case=case, repeat=trial + 1, passed=report["passed"],
                                 failed_tensors=len(failures), failed_elements=count,
                                 self_repeat_passed=self_passed,
                                 self_failed_elements=None if self_rows is None else
                                 sum(x["failed_elements"] for x in self_rows)))
            if first is None:
                first, first_edges = actual, edge_table
        if tensor_fingerprint(model.state_dict()) != state_hash:
            raise RuntimeError("Numerical probe changed model parameters or buffers")
        del model, actual, first
    if diagnose:
        (directory / f"diagnostic_summary_rounds{rounds}.json").write_text(
            json.dumps(outcomes, indent=2), encoding="utf-8"
        )
    if any(not row["passed"] or row["self_repeat_passed"] is False
           for row in outcomes if row["case"] in ("optimized", "reference_repeat")):
        raise RuntimeError("Numerical gate failed; inspect numerical_rounds*.json before training")
    del models, reference
    if device == "cuda":
        torch.cuda.empty_cache()


def command(directory, rounds, reference, epochs, lmdb_dir, offline, *, candidate=(),
            activation_checkpointing=False):
    if reference and candidate:
        raise ValueError("Candidate optimizations require optimized=true")
    activation_checkpointing = candidate_checkpointing(candidate, activation_checkpointing)
    prefix = "model.flow_model_args.g1new.module."
    return [
        sys.executable, "-u", "src/train.py", "experiment=g1new_small_stage1",
        "execution=optimized", f"data.lmdb_dir={lmdb_dir}",
        f"hydra.run.dir={directory.as_posix()}", f"task_name=bench_g1new_{directory.name}",
        "logger=wandb", "+logger.wandb.entity=moonshuo-42-meta",
        "logger.wandb.project=AtomMOF-9", f"logger.wandb.offline={str(offline).lower()}",
        "trainer=ddp", "trainer.devices=8", "trainer.num_nodes=1",
        f"trainer.max_epochs={epochs}", "++trainer.precision=32-true",
        "trainer.accumulate_grad_batches=2", "++trainer.num_sanity_val_steps=0",
        "data.sampler=dynamic", "data.num_workers=8", "data.max_num_atoms=200",
        "data.dynamic.scaling=square", "data.dynamic.max_num_atoms_square=3000000",
        "data.dynamic.max_batch_size=32", "data.dynamic.skip_too_big=false",
        "data.sample_limit_dict.train=null", "data.sample_limit_dict.val=1000",
        "model.token_s=256", "model.token_z=128",
        "model.flow_model_args.token_transformer_heads=8",
        "model.flow_model_args.g1.enabled=false", "model.flow_model_args.g1new.enabled=true",
        "model.flow_model_args.g1new.insert_after=[6,12,18,24]",
        "model.flow_model_args.activation_checkpointing=false",
        prefix + f"activation_checkpointing={str(activation_checkpointing).lower()}",
        prefix + "condition_encoder.activation_checkpointing=false",
        prefix + f"rounds={rounds}", prefix + f"optimized={str(not reference).lower()}",
        prefix + f"edge_chunk_size={8192 if reference else 65536}",
        prefix + f"geometry.pair_chunk_size={4096 if reference else 65536}",
        prefix + f"geometry.image_chunk_size={65536 if reference else 262144}",
        *["++" + prefix + ("geometry." if name == "merge_search" else "")
          + f"{name}={str(enabled).lower()}" for name, enabled in runtime_options(candidate).items()],
        "model.training_args.lr_scheduler=af3", "model.training_args.max_lr=0.0001",
        "model.training_args.lr_warmup_no_steps=10000",
        "model.validation_args.sample_every_n_epochs=10", "model.validation_args.num_workers=1",
        "+callbacks.execution_benchmark._target_=src.callbacks.execution_benchmark.ExecutionBenchmark",
        "+callbacks.execution_benchmark.warmup_batches=20",
        "pretrained_ckpt_path=null", "resume_ckpt_path=null", "seed=42",
    ]


def run(cmd, directory):
    directory.mkdir()
    (directory / "command.json").write_text(json.dumps(cmd, indent=2), encoding="utf-8")
    print("\n" + shlex.join(cmd), flush=True)
    env = dict(os.environ, PROJECT_ROOT=str(ROOT), HYDRA_FULL_ERROR="1",
               CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7")
    # A process group allows Ctrl-C to stop this experiment's DDP children too.
    with (directory / "train.log").open("w", encoding="utf-8") as log:
        child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1,
                                 start_new_session=os.name != "nt")
        try:
            for line in child.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = child.wait()
        except BaseException:
            if child.poll() is None:
                if os.name == "nt":
                    child.terminate()
                else:
                    os.killpg(child.pid, signal.SIGINT)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    if os.name == "nt":
                        child.kill()
                    else:
                        os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            raise
        finally:
            child.stdout.close()
    if code:
        raise RuntimeError(f"Training failed with exit {code}; see {directory / 'train.log'}")


def summarize(root, expected_epochs):
    rows, fingerprints = [], {}
    plan_path = root / "benchmark_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else {}
    policy = ("precision", "float32_matmul_precision", "cuda_matmul_allow_tf32",
              "cudnn_allow_tf32", "parameter_count", "accumulate_grad_batches")
    for path in sorted(root.glob("rounds*_*/performance")):
        data = [json.loads(p.read_text(encoding="utf-8")) for p in path.glob("rank*_epoch*.json")]
        for epoch in range(expected_epochs):
            ranks = sorted((r for r in data if r["epoch"] == epoch), key=lambda r: r["rank"])
            if [r["rank"] for r in ranks] != list(range(8)):
                print(f"Incomplete {path.parent.name}, epoch {epoch + 1}; no complete-epoch claim")
                continue
            if any(r["world_size"] != 8 or not r.get("batches_sha256") for r in ranks):
                raise RuntimeError("Missing distributed/data-order evidence")
            if any(r["parameter_count"] != 39479632 or r["precision"] != "32-true"
                   or r["float32_matmul_precision"] != "high"
                   or not r["cuda_matmul_allow_tf32"] or not r["cudnn_allow_tf32"]
                   or r["accumulate_grad_batches"] != 2 for r in ranks):
                raise RuntimeError("Architecture/precision/accumulation differs from the control")
            row = dict(run=path.parent.name, epoch=epoch + 1,
                       batches=sorted({r["total_batches"] for r in ranks}),
                       seconds=max(r["train_seconds"] for r in ranks),
                       allocated_gib=max(r["peak_allocated_bytes"] for r in ranks) / 2**30,
                       reserved_gib=max(r["peak_reserved_bytes"] for r in ranks) / 2**30)
            row["below_120s"] = row["seconds"] < 120
            if path.parent.name in plan:
                mode = plan[path.parent.name]
                row.update(control=mode["control"], candidate=mode["candidate"],
                           activation_checkpointing=mode["activation_checkpointing"])
            rows.append(row)
            fingerprints[(row["run"], epoch)] = [
                [r[k] for k in (*policy, "total_batches", "batches_sha256", "shapes_sha256")]
                for r in ranks
            ]
    text = ["Training segment only; includes first 20 batches, excludes validation/save.",
            "Max over eight ranks. These are unprofiled full epochs, not extrapolations.",
            "A deterministic gate does not certify normal-CUDA self-repeat variation; use --diagnose-only.",
            "| run | epoch | batches/rank | seconds | allocated GiB | reserved GiB | <120s |",
            "|---|---:|---|---:|---:|---:|---|"]
    for r in rows:
        text.append(f"| {r['run']} | {r['epoch']} | {r['batches']} | {r['seconds']:.2f} | "
                    f"{r['allocated_gib']:.2f} | {r['reserved_gib']:.2f} | {r['below_120s']} |")
        ref = r.get("control") or r["run"].replace("_optimized", "_reference")
        key = (ref, r["epoch"] - 1)
        if (r.get("control") or r["run"].endswith("_optimized")) and key in fingerprints:
            if fingerprints[key] != fingerprints[(r["run"], r["epoch"] - 1)]:
                raise RuntimeError("Reference/optimized data order or precision differs; invalid A/B")
            if r.get("control"):
                control = next(x for x in rows if x["run"] == ref and x["epoch"] == r["epoch"])
                if r["activation_checkpointing"] != control["activation_checkpointing"]:
                    raise RuntimeError("Candidate/control checkpointing policies differ; invalid A/B")
                r["time_change_percent"] = 100 * (r["seconds"] / control["seconds"] - 1)
                r["allocated_change_gib"] = r["allocated_gib"] - control["allocated_gib"]
                text.append(f"  vs {ref}: time {r['time_change_percent']:+.2f}%, "
                            f"allocated {r['allocated_change_gib']:+.2f} GiB "
                            f"(checkpointing={r['activation_checkpointing']}).")
    for name in sorted({r["run"] for r in rows}):
        values = [r["seconds"] for r in rows if r["run"] == name]
        text.append(f"{name}: mean={statistics.mean(values):.2f}s, complete epochs={len(values)}")
    if not rows:
        text = ["No complete training epochs were recorded. There are no timing/memory results.",
                "Inspect numerical_rounds*.json under verification_rounds* if the gate failed."]
    report = "\n".join(text) + "\n"
    (root / "summary.log").write_text(report, encoding="utf-8")
    (root / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(report, flush=True)


def package(root):
    with tarfile.open(root / "reports.tar.gz", "w:gz") as archive:
        for path in sorted(root.rglob("*")):
            rel = path.relative_to(root)
            if (path.is_file() and not path.is_symlink()
                    and not {"checkpoints", "wandb"}.intersection(rel.parts)
                    and path.suffix in {".json", ".log", ".yaml", ".txt", ".patch", ".py"}):
                archive.add(path, arcname=str(rel), recursive=False)
    print(f"Download: {root / 'reports.tar.gz'}", flush=True)


def diagnose_profiles(root, args):
    """Independent processes: deterministic diagnostics cannot affect training."""
    runs, rows, fixtures = [], [], {}
    lines = ["Diagnostic only: no training epochs or performance measurements.",
             "Same FP32/high/TF32 and elementwise tolerance in both profiles.",
             "Counts below are ranges across repeated evaluations, not relaxed gates.",
             "| profile | rounds | case | repeats | failed tensors | failed elements | self failures |",
             "|---|---:|---|---:|---|---|---|"]
    for profile in ("training", "deterministic"):
        for rounds in dict.fromkeys(args.rounds):
            directory = root / f"{profile}_rounds{rounds}"
            cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--diagnose-only",
                   "--diagnostic-profile", profile, "--repeats", str(args.repeats),
                   "--verify-device", args.verify_device, "--rounds", str(rounds),
                   "--output", str(directory)]
            if getattr(args, "activation_checkpointing", False):
                cmd.append("--activation-checkpointing")
            print("\n" + shlex.join(cmd), flush=True)
            if args.dry_run:
                continue
            result = subprocess.run(cmd, cwd=ROOT, check=False)
            run = dict(profile=profile, rounds=rounds, returncode=result.returncode)
            runs.append(run)
            path = directory / f"diagnostic_summary_rounds{rounds}.json"
            if not path.exists():
                run["complete"] = False
                lines.append(f"{profile}, rounds={rounds}: interrupted/error before complete results; "
                             f"see {directory.name}/diagnostic_error.json")
                continue
            values = json.loads(path.read_text(encoding="utf-8"))
            case_count = 2 if getattr(args, "activation_checkpointing", False) else 6
            run["complete"] = len(values) == case_count * args.repeats
            for case in dict.fromkeys(row["case"] for row in values):
                selected = [r for r in values if r["case"] == case]
                record = dict(profile=profile, rounds=rounds, case=case, repeats=len(selected),
                              all_passed=all(r["passed"] for r in selected),
                              failed_tensors=[r["failed_tensors"] for r in selected],
                              failed_elements=[r["failed_elements"] for r in selected],
                              self_failed_elements=[r["self_failed_elements"] for r in selected
                                                    if r["self_failed_elements"] is not None])
                rows.append(record)
                def span(name):
                    v = record[name]
                    return f"{min(v)}..{max(v)}" if v else "not measured"
                lines.append(f"| {profile} | {rounds} | {case} | {len(selected)} | "
                             f"{span('failed_tensors')} | {span('failed_elements')} | "
                             f"{span('self_failed_elements')} |")
            evidence = json.loads((directory / f"numerical_rounds{rounds}.json").read_text(
                encoding="utf-8"
            ))
            fixtures[(profile, rounds)] = (
                evidence["fixture_sha256"], evidence["initial_state_sha256"],
                evidence["matmul_precision"], evidence["matmul_tf32"], evidence["cudnn_tf32"],
                evidence["rtol"], evidence["atol"],
            )
    if args.dry_run:
        return
    identical = all(fixtures.get(("training", n)) is not None
                    and fixtures.get(("training", n)) == fixtures.get(("deterministic", n))
                    for n in dict.fromkeys(args.rounds))
    lines.append(f"Identical inputs/initial weights/precision/tolerances across profiles: {identical}")
    report = dict(runs=runs, identical_fixtures_and_precision=identical, cases=rows)
    (root / "diagnostic_profiles.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (root / "summary.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    if not identical or any(r["returncode"] != 0 or not r["complete"] for r in runs):
        raise RuntimeError("Diagnostic profiles completed with failures; inspect summary.log and reports")


def diagnose_candidates(root, args):
    """Report production repeatability separately from each deterministic gate."""
    runs, rows, fixtures = [], [], {}
    selections = list(dict.fromkeys(args.candidate))
    for candidate in selections:
        name = candidate_name(candidate)
        for profile in ("training", "deterministic"):
            for rounds in dict.fromkeys(args.rounds):
                directory = root / f"{profile}_rounds{rounds}_{name}"
                cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--diagnose-only",
                       "--diagnostic-profile", profile, "--candidate", ",".join(candidate),
                       "--repeats", str(args.repeats), "--verify-device", args.verify_device,
                       "--rounds", str(rounds), "--output", str(directory)]
                if args.activation_checkpointing:
                    cmd.append("--activation-checkpointing")
                print("\n" + shlex.join(cmd), flush=True)
                if args.dry_run:
                    continue
                result = subprocess.run(cmd, cwd=ROOT, check=False)
                result_dir = directory / name
                summary = result_dir / f"diagnostic_summary_rounds{rounds}.json"
                values = json.loads(summary.read_text(encoding="utf-8")) if summary.exists() else []
                runs.append(dict(candidate=name, profile=profile, rounds=rounds,
                                 returncode=result.returncode,
                                 complete=len(values) == 2 * args.repeats))
                rows.extend(dict(candidate=name, profile=profile, rounds=rounds, **row)
                            for row in values)
                evidence = result_dir / f"numerical_rounds{rounds}.json"
                if evidence.exists():
                    report = json.loads(evidence.read_text(encoding="utf-8"))
                    fixtures[(name, profile, rounds)] = [report[key] for key in (
                        "fixture_sha256", "initial_state_sha256", "matmul_precision", "matmul_tf32",
                        "cudnn_tf32", "rtol", "atol", "activation_checkpointing", "edge_chunk_size"
                    )]
    if args.dry_run:
        return
    identical = all(
        fixtures.get((candidate_name(c), "training", n)) is not None
        and fixtures.get((candidate_name(c), "training", n))
        == fixtures.get((candidate_name(c), "deterministic", n))
        for c in selections for n in dict.fromkeys(args.rounds)
    )
    # Flags change execution only: compare fixture/state fingerprints across
    # candidates too, even when their checkpointing policies legitimately differ.
    same_inputs = all(len({tuple(v[:2]) for (name, profile, n), v in fixtures.items() if n == rounds}) == 1
                      for rounds in dict.fromkeys(args.rounds))
    lines = ["Diagnostic only: no training/performance measurements.",
             "Normal CUDA repeatability and the deterministic gate are reported separately.",
             "Same FP32/high/TF32 and unchanged elementwise tolerances; no failure is suppressed.",
             "| candidate | profile | rounds | case | repeat | failed tensors | failed elements | self passed |",
             "|---|---|---:|---|---:|---:|---:|---|"]
    lines.extend(f"| {r['candidate']} | {r['profile']} | {r['rounds']} | {r['case']} | "
                 f"{r['repeat']} | {r['failed_tensors']} | {r['failed_elements']} | "
                 f"{r['self_repeat_passed']} |" for r in rows)
    lines += [f"Identical fixtures/precision across profiles: {identical}",
              f"Identical inputs/weights across candidates: {same_inputs}"]
    (root / "diagnostic_profiles.json").write_text(json.dumps(dict(
        runs=runs, identical_fixtures_and_precision=identical,
        identical_inputs_and_weights_across_candidates=same_inputs, cases=rows
    ), indent=2), encoding="utf-8")
    (root / "summary.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    if not identical or not same_inputs or any(r["returncode"] or not r["complete"] for r in runs):
        raise RuntimeError("Candidate diagnostic profiles completed with failures; inspect reports")


def benchmark_candidates(root, args):
    """Gate each candidate first, then run matching controls and its full epochs."""
    plan, completed = {}, set()

    def train_mode(rounds, name, candidate=(), checkpointing=False, reference=False, control=None):
        run_name = f"rounds{rounds}_{name}"
        if run_name in completed:
            return
        run_dir = root / run_name
        plan[run_name] = dict(candidate=list(candidate), activation_checkpointing=checkpointing,
                              reference=reference, control=control)
        (root / "benchmark_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        cmd = command(run_dir, rounds, reference, args.epochs, args.lmdb_dir, args.offline,
                      candidate=candidate, activation_checkpointing=checkpointing)
        if args.dry_run:
            print(shlex.join(cmd), flush=True)
        else:
            run(cmd, run_dir)
        completed.add(run_name)

    for rounds in dict.fromkeys(args.rounds):
        if args.include_reference:
            # The existing reference is a separate comparison, not the control
            # for candidates whose checkpoint policy might differ.
            gate = [sys.executable, "-u", str(Path(__file__).resolve()), "--verify-only",
                    "--verify-device", "cuda", "--rounds", str(rounds), "--repeats", str(args.repeats),
                    "--output", str(root / f"verification_rounds{rounds}_legacy")]
            if args.activation_checkpointing:
                gate.append("--activation-checkpointing")
            if not args.dry_run:
                subprocess.run(gate, cwd=ROOT, check=True,
                               env=dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"))
            train_mode(rounds, "reference", checkpointing=args.activation_checkpointing, reference=True)
        for candidate in dict.fromkeys(args.candidate):
            name = candidate_name(candidate)
            checkpointing = candidate_checkpointing(candidate, args.activation_checkpointing)
            gate = [sys.executable, "-u", str(Path(__file__).resolve()), "--verify-only",
                    "--verify-device", "cuda", "--candidate", ",".join(candidate),
                    "--rounds", str(rounds), "--repeats", str(args.repeats),
                    "--output", str(root / f"verification_rounds{rounds}_{name}")]
            if args.activation_checkpointing:
                gate.append("--activation-checkpointing")
            print("\n" + shlex.join(gate), flush=True)
            if not args.dry_run:
                subprocess.run(gate, cwd=ROOT, check=True,
                               env=dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"))
            control_name = "optimized_checkpointed" if checkpointing else "optimized"
            reference = f"rounds{rounds}_reference" if (args.include_reference
                        and checkpointing == args.activation_checkpointing) else None
            train_mode(rounds, control_name, checkpointing=checkpointing, control=reference)
            train_mode(rounds, name, candidate, checkpointing,
                       control=f"rounds{rounds}_{control_name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, nargs="+", default=[2], choices=(1, 2, 3))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--candidate", type=parse_candidate, action="append", default=[],
                        metavar="NAME[,NAME...]",
                        help="Repeat for separate A/Bs; comma-join for a combination; all enables all four")
    parser.add_argument("--activation-checkpointing", action="store_true",
                        help="Use matching G1new checkpointing for gate/control/candidate; "
                             "checkpoint_relation_inputs also enables it automatically")
    parser.add_argument("--offline", action="store_true", help="W&B offline; training is unchanged")
    gate_mode = parser.add_mutually_exclusive_group()
    gate_mode.add_argument("--verify-only", action="store_true",
                           help="Deterministic reference/optimized gate; never train")
    gate_mode.add_argument("--diagnose-only", action="store_true",
                           help="Repeat the reference and isolate three operation reorders; never train")
    parser.add_argument("--diagnostic-profile", choices=("both", "training", "deterministic"),
                        default="both", help="Only for --diagnose-only; both uses fresh child processes")
    parser.add_argument("--repeats", type=int, default=5,
                        help="Repeated comparisons per numerical gate/diagnostic case (default 5)")
    parser.add_argument("--verify-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--lmdb-dir", default="/data/ai4science/AtomMOF/AtomMOF-main/data/bwdb")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.repeats < 2:
        parser.error("--repeats must be at least two to measure self-repeat variation")
    if not args.diagnose_only and args.diagnostic_profile != "both":
        parser.error("--diagnostic-profile requires --diagnose-only; it cannot change training")
    if args.verify_device == "cpu" and not (args.verify_only or args.diagnose_only or args.dry_run):
        parser.error("Server training requires the CUDA numerical gate")
    prefix = "diagnose_g1new_numerics_" if args.diagnose_only else "bench_g1new_execution_b32_a2_8gpu_"
    root = (args.output or ROOT / "logs" / (
        prefix + datetime.now().strftime("%Y%m%d_%H%M%S")
    )).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    for filename, git_args in (("git_commit.txt", ["rev-parse", "HEAD"]),
                               ("git_status.txt", ["status", "--short"]),
                               ("tracked_changes.patch", ["diff", "HEAD"])):
        result = subprocess.run(["git", *git_args], cwd=ROOT, capture_output=True)
        (root / filename).write_bytes(result.stdout)
    sources = list((ROOT / "src/models/g1new").rglob("*.py"))
    sources += list((ROOT / "configs/g1new").rglob("*.yaml"))
    (root / "source_sha256.json").write_text(json.dumps({
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
    }, indent=2), encoding="utf-8")
    try:
        if args.diagnose_only and args.diagnostic_profile == "both":
            if args.candidate:
                diagnose_candidates(root, args)
            else:
                diagnose_profiles(root, args)
            return
        if args.verify_only or args.diagnose_only:
            if not args.dry_run:
                profile = "deterministic" if args.verify_only else args.diagnostic_profile
                # Enter once before CUDA initialization, also for --rounds 1 2 3.
                # This process exits before timed training is launched.
                with diagnostic_policy(profile):
                    for rounds in dict.fromkeys(args.rounds):
                        if args.candidate:
                            for candidate in dict.fromkeys(args.candidate):
                                directory = root / candidate_name(candidate)
                                directory.mkdir(exist_ok=True)
                                verify(directory, rounds, args.verify_device, diagnose=args.diagnose_only,
                                       repeats=args.repeats, candidate=candidate,
                                       activation_checkpointing=args.activation_checkpointing)
                        else:
                            verify(root, rounds, args.verify_device, diagnose=args.diagnose_only,
                                   repeats=args.repeats,
                                   activation_checkpointing=args.activation_checkpointing)
            return
        if args.candidate:
            benchmark_candidates(root, args)
            return
        for rounds in dict.fromkeys(args.rounds):
            if not args.dry_run:
                # Do not set deterministic flags/workspace in this parent or
                # the training child. The strict gate uses the same precision
                # and tolerance, but a repeatable reduction policy. Its CUDA
                # context/allocator is released before timed DDP training.
                gate = [
                    sys.executable, "-u", str(Path(__file__).resolve()),
                    "--verify-only", "--verify-device", "cuda", "--rounds", str(rounds),
                    "--repeats", str(args.repeats),
                    "--output", str(root / f"verification_rounds{rounds}"),
                ]
                if args.activation_checkpointing:
                    gate.append("--activation-checkpointing")
                subprocess.run(gate, cwd=ROOT,
                               env=dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"), check=True)
            for reference in ([True, False] if args.include_reference else [False]):
                run_dir = root / f"rounds{rounds}_{'reference' if reference else 'optimized'}"
                cmd = command(run_dir, rounds, reference, args.epochs, args.lmdb_dir, args.offline,
                              activation_checkpointing=args.activation_checkpointing)
                if args.dry_run:
                    print(shlex.join(cmd), flush=True)
                else:
                    run(cmd, run_dir)
    except Exception as exc:
        (root / "diagnostic_error.json").write_text(json.dumps(dict(
            exception_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()
        ), indent=2), encoding="utf-8")
        raise
    finally:
        # Package partial evidence even after an interrupt/failure. Incomplete
        # epochs/ranks cannot enter the latency table as a completed epoch.
        try:
            if not (args.verify_only or args.diagnose_only or args.dry_run):
                summarize(root, args.epochs)
        finally:
            package(root)


if __name__ == "__main__":
    main()
