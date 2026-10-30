"""G1-New-only correctness gate and full-epoch B32/a2 timing on the server.

No backbone, precision, batch, geometry cutoff or architecture ablations.
Default: four sites, two internal rounds, two complete epochs on eight GPUs.
Use --include-reference for a same-code, same-data old/new implementation pair.
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
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


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
    """Restore 11d7480 operations on a disposable probe, never a training model.

    Test one factor at a time without changing tensor dtype, TF32 or tolerances.
    The combined case reconstructs all three reassociations from the failed run.
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
        mass, numerator = pooled[:, 0], pooled[:, 1:]
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


def verify(directory, rounds, device, diagnose=False):
    """Nonzero residuals, all four sites, live geometry and all changed weights.

    Conditions are supplied identically to isolate the modified local branch;
    the unchanged condition encoder is covered by the repository integration tests.
    CUDA uses exactly the training FP32/high/TF32 policy, not a double-only check.
    """
    from copy import deepcopy
    import torch
    from src.models.g1new.bridge import G1NewBridge

    torch.manual_seed(271)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the server numerical gate")
    if device == "cpu":
        torch.set_num_threads(1)
    models = []
    for optimized in (False, True):
        config = dict(
            _target_="src.models.g1new.G1NewModule", optimized=optimized,
            rounds=rounds, activation_checkpointing=False,
            edge_chunk_size=65536 if optimized else 8192,
            geometry=dict(
                _target_="src.models.g1new.geometry.LocalGeometry",
                pair_chunk_size=65536 if optimized else 4096,
                image_chunk_size=262144 if optimized else 65536,
            ),
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
    cases = [("optimized", {})]
    if diagnose:
        cases = [("reference_repeat", {}), *cases,
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
        actual, edge_table = evaluate(model)
        report = dict(rounds=rounds, case=case, device=device, rtol=0.003, atol=0.0002,
                      matmul_precision=torch.get_float32_matmul_precision(),
                      matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                      cudnn_tf32=torch.backends.cudnn.allow_tf32,
                      torch_version=torch.__version__, cuda_version=torch.version.cuda,
                      gpu_name=torch.cuda.get_device_name() if device == "cuda" else None,
                      shape=[B, N], edge_count=len(edge_table),
                      identical_edge_order=reference_edges == edge_table,
                      identical_edge_sets=sorted(reference_edges) == sorted(edge_table),
                      tensors=[compare_tensors(name, value, actual[name])
                               for name, value in reference.items()])
        report["passed"] = report["identical_edge_sets"] and all(x["passed"] for x in report["tensors"])
        suffix = "" if case == "optimized" else "_" + case
        (directory / f"numerical_rounds{rounds}{suffix}.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
        failures = [x["name"] for x in report["tensors"] if not x["passed"]]
        print(f"Numerical check rounds={rounds}, case={case}: "
              f"passed={report['passed']}, failures={failures}", flush=True)
        outcomes.append(dict(case=case, passed=report["passed"], failed_tensors=len(failures),
                             failed_elements=sum(x["failed_elements"] for x in report["tensors"])))
        del model, actual
    if diagnose:
        (directory / f"diagnostic_summary_rounds{rounds}.json").write_text(
            json.dumps(outcomes, indent=2), encoding="utf-8"
        )
    if any(not row["passed"] for row in outcomes if row["case"] in ("optimized", "reference_repeat")):
        raise RuntimeError("Numerical gate failed; inspect numerical_rounds*.json before training")
    del models, reference
    if device == "cuda":
        torch.cuda.empty_cache()


def command(directory, rounds, reference, epochs, lmdb_dir, offline):
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
        prefix + "activation_checkpointing=false",
        prefix + "condition_encoder.activation_checkpointing=false",
        prefix + f"rounds={rounds}", prefix + f"optimized={str(not reference).lower()}",
        prefix + f"edge_chunk_size={8192 if reference else 65536}",
        prefix + f"geometry.pair_chunk_size={4096 if reference else 65536}",
        prefix + f"geometry.image_chunk_size={65536 if reference else 262144}",
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
            rows.append(row)
            fingerprints[(row["run"], epoch)] = [
                [r[k] for k in (*policy, "total_batches", "batches_sha256", "shapes_sha256")]
                for r in ranks
            ]
    text = ["Training segment only; includes first 20 batches, excludes validation/save.",
            "Max over eight ranks. These are unprofiled full epochs, not extrapolations.",
            "| run | epoch | batches/rank | seconds | allocated GiB | reserved GiB | <120s |",
            "|---|---:|---|---:|---:|---:|---|"]
    for r in rows:
        text.append(f"| {r['run']} | {r['epoch']} | {r['batches']} | {r['seconds']:.2f} | "
                    f"{r['allocated_gib']:.2f} | {r['reserved_gib']:.2f} | {r['below_120s']} |")
        ref = r["run"].replace("_optimized", "_reference")
        key = (ref, r["epoch"] - 1)
        if r["run"].endswith("_optimized") and key in fingerprints:
            if fingerprints[key] != fingerprints[(r["run"], r["epoch"] - 1)]:
                raise RuntimeError("Reference/optimized data order or precision differs; invalid A/B")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, nargs="+", default=[2], choices=(1, 2, 3))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--offline", action="store_true", help="W&B offline; training is unchanged")
    gate_mode = parser.add_mutually_exclusive_group()
    gate_mode.add_argument("--verify-only", action="store_true")
    gate_mode.add_argument("--diagnose-only", action="store_true",
                           help="Repeat the reference and isolate three operation reorders; never train")
    parser.add_argument("--verify-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--lmdb-dir", default="/data/ai4science/AtomMOF/AtomMOF-main/data/bwdb")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
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
        for rounds in dict.fromkeys(args.rounds):
            if not args.dry_run:
                if args.verify_only or args.diagnose_only:
                    verify(root, rounds, args.verify_device, diagnose=args.diagnose_only)
                else:
                    # Exit the verification process before training: its CUDA
                    # context and allocator must not occupy GPU 0 during timing.
                    subprocess.run([
                        sys.executable, "-u", str(Path(__file__).resolve()),
                        "--verify-only", "--verify-device", "cuda", "--rounds", str(rounds),
                        "--output", str(root / f"verification_rounds{rounds}"),
                    ], cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"),
                       check=True)
            if args.verify_only or args.diagnose_only:
                continue
            for reference in ([True, False] if args.include_reference else [False]):
                run_dir = root / f"rounds{rounds}_{'reference' if reference else 'optimized'}"
                cmd = command(run_dir, rounds, reference, args.epochs, args.lmdb_dir, args.offline)
                if args.dry_run:
                    print(shlex.join(cmd), flush=True)
                else:
                    run(cmd, run_dir)
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
