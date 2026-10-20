"""Launch real BWDB/DDP training comparisons in independent output directories.

Example (server):
  python scripts/benchmark_execution.py --lmdb-dir /path/to/bwdb --model g1 --stage 1

Each profile starts from the same seed, architecture, optional checkpoint and
batch budget. Reports are TRAINING-WINDOW evidence, not full training/MR proof.
Use --batches 0 for full epochs. All 3 stages and both models must be measured.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    1: dict(atoms="200", square=3000000, batch=32, accum=2),
    2: dict(atoms="300", square=3000000, batch=16, accum=4),
    3: dict(atoms="null", square=500000, batch=16, accum=4),
}


def summarize(runs, reference_profile="reference"):
    """Only compare complete, equal-work windows; never infer unmeasured VRAM."""
    summaries = []
    for run in runs:
        rows = [
            json.loads(p.read_text())
            for p in sorted((Path(run["directory"]) / "performance").glob("rank*_epoch*.json"))
        ]
        if not rows or any(
            not row["measured_seconds"]
            or not row["batches_sha256"]
            or any(
                key not in row
                for key in (
                    "precision",
                    "float32_matmul_precision",
                    "cuda_matmul_allow_tf32",
                    "cudnn_allow_tf32",
                )
            )
            for row in rows
        ):
            raise RuntimeError("Missing timing, batch-ID or precision evidence; cannot compare")
        epochs = {
            e: sorted((x for x in rows if x["epoch"] == e), key=lambda x: x["rank"])
            for e in {x["epoch"] for x in rows}
        }
        for values in epochs.values():
            if len(values) != values[0]["world_size"]:
                raise RuntimeError("Missing DDP rank reports")
        entry = dict(
            run,
            epochs=epochs,
            seconds_per_batch=statistics.median(
                max(x["measured_seconds"] for x in v) / v[0]["measured_batches"]
                for v in epochs.values()
            ),
            peak_allocated_bytes=max(x["peak_allocated_bytes"] for x in rows),
            peak_reserved_bytes=max(x["peak_reserved_bytes"] for x in rows),
        )
        summaries.append(entry)
    reference = {x["repeat"]: x for x in summaries if x["profile"] == reference_profile}
    for x in summaries:
        ref = reference.get(x["repeat"])
        if ref is None:
            continue
        same = set(x["epochs"]) == set(ref["epochs"])
        for epoch, values in x["epochs"].items():
            other = ref["epochs"].get(epoch, [])
            same &= len(values) == len(other)
            for a, b in zip(values, other):
                same &= all(
                    a[k] == b[k]
                    for k in (
                        "rank",
                        "world_size",
                        "batches_sha256",
                        "shapes_sha256",
                        "parameter_count",
                        "precision",
                        "float32_matmul_precision",
                        "cuda_matmul_allow_tf32",
                        "cudnn_allow_tf32",
                        "optimizer_updates",
                        "accumulate_grad_batches",
                        "total_batches",
                        "warmup_batches",
                        "measured_batches",
                        "measured_samples",
                    )
                )
        x["equal_work_verified"] = bool(same)
        if same:
            x["training_window_speedup"] = ref["seconds_per_batch"] / x["seconds_per_batch"]
            x["peak_memory_ratio"] = (
                x["peak_allocated_bytes"] / ref["peak_allocated_bytes"]
                if ref["peak_allocated_bytes"]
                else None
            )
            x["window_targets_met"] = (
                x["training_window_speedup"] >= 2
                and x["peak_memory_ratio"] is not None
                and x["peak_memory_ratio"] <= 0.5
            )
    return summaries


def profile_overrides(profile):
    """Keep a same-work comparison against e509a4a operator execution.

    Reference omits the config-group override: execution=null is not a valid
    Hydra config-group value on the server's Hydra version.
    """
    if profile == "reference":
        return []
    if profile in ("math", "optimized"):
        return [f"execution={profile}"]
    options = {
        "optimized_previous": dict(
            pair_bias_chunk_size=1,
            fused_pair_features=False,
            packed_geometry=False,
            joint_time_metrics=False,
            bucket_geometry_images=False,
        ),
        "bucketed": dict(bucket_geometry_images=True),
    }[profile]
    return [
        "execution=optimized",
        *[
            f"model.flow_model_args.execution.{key}={str(value).lower()}"
            for key, value in options.items()
        ],
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmdb-dir", required=True, type=Path)
    parser.add_argument("--model", choices=["baseline", "g1"], default="g1")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=["reference", "math", "optimized_previous", "optimized", "bucketed"],
        default=["reference", "math", "optimized"],
    )
    parser.add_argument("--compare-to", choices=["reference", "optimized_previous"], default=None)
    parser.add_argument(
        "--batch-size", type=int, help="Override the per-GPU cap equally in all profiles"
    )
    parser.add_argument(
        "--accumulate-grad-batches", type=int, help="Override accumulation equally in all profiles"
    )
    parser.add_argument(
        "--compile-blocks",
        action="store_true",
        help="Compile only DiT blocks in optimized profile",
    )
    parser.add_argument(
        "--compile-pair-features",
        action="store_true",
        help="Compile only pair features in optimized profile",
    )
    parser.add_argument("--batches", type=int, default=200, help="0 means a full epoch")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--dry-run", action="store_true", help="Write exact commands without starting training"
    )
    parser.add_argument(
        "--compile", action="store_true", help="Compile DiT blocks in optimized profile only"
    )
    args = parser.parse_args()
    stage = dict(STAGES[args.stage])
    for key, value in (("batch", args.batch_size), ("accum", args.accumulate_grad_batches)):
        if value is not None:
            if value < 1:
                parser.error("batch size and accumulation must be positive")
            stage[key] = value
    if args.compare_to is not None and args.compare_to not in args.profiles:
        parser.error("--compare-to must be included in --profiles")
    args.compare_to = args.compare_to or "reference"
    if args.epochs < 1 or args.repeats < 1 or args.warmup < 1 or args.warmup % stage["accum"]:
        parser.error(
            "epochs/repeats positive; warmup must be positive and divisible by accumulation"
        )
    if args.batches and (args.batches <= args.warmup or args.batches % stage["accum"]):
        parser.error("batches must exceed warmup and be divisible by accumulation, or be 0")
    if not args.lmdb_dir.is_dir() or (args.checkpoint and not args.checkpoint.is_file()):
        parser.error("Dataset/checkpoint path does not exist")
    output = (
        args.output
        or ROOT
        / "logs"
        / f"execution_{args.model}_stage{args.stage}_{datetime.now():%Y%m%d_%H%M%S}"
    ).resolve()
    output.mkdir(parents=True, exist_ok=False)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    (output / "git_commit.txt").write_text(commit + "\n")
    env = dict(
        os.environ, CUDA_VISIBLE_DEVICES=args.devices, PROJECT_ROOT=str(ROOT), HYDRA_FULL_ERROR="1"
    )
    devices = len(args.devices.split(","))
    runs = []
    for repeat in range(args.repeats):
        # Rotate profile order to reduce systematic warm-cache/temperature bias.
        profiles = (
            args.profiles[repeat % len(args.profiles) :]
            + args.profiles[: repeat % len(args.profiles)]
        )
        for profile in profiles:
            directory = output / f"repeat{repeat}_{profile}"
            directory.mkdir()
            overrides = [
                "data=bwdb",
                f"data.lmdb_dir={json.dumps(str(args.lmdb_dir.resolve()))}",
                "trainer=ddp",
                f"trainer.devices={devices}",
                "trainer.num_nodes=1",
                f"trainer.max_epochs={args.epochs}",
                f"trainer.accumulate_grad_batches={stage['accum']}",
                f"++trainer.limit_train_batches={args.batches if args.batches else '1.0'}",
                "++trainer.limit_val_batches=0",
                "++trainer.num_sanity_val_steps=0",
                "++trainer.enable_checkpointing=false",
                "++trainer.enable_model_summary=false",
                "++trainer.enable_progress_bar=false",
                "logger=[]",
                "callbacks=benchmark",
                f"callbacks.execution_benchmark.warmup_batches={args.warmup}",
                "data.sampler=dynamic",
                f"data.num_workers={args.workers}",
                "++data.persistent_workers=false",  # identical worker RNG lifecycle across profiles
                f"data.max_num_atoms={stage['atoms']}",
                "data.dynamic.scaling=square",
                f"data.dynamic.max_num_atoms_square={stage['square']}",
                f"data.dynamic.max_batch_size={stage['batch']}",
                "data.dynamic.skip_too_big=false",
                "data.sample_limit_dict.train=null",
                "data.sample_limit_dict.val=1000",
                "model.token_s=256",
                "model.token_z=128",
                "model.flow_model_args.token_transformer_heads=8",
                "model.flow_model_args.activation_checkpointing=false",
                f"model.flow_model_args.g1.enabled={str(args.model=='g1').lower()}",
                "model.flow_model_args.g1.activation_checkpointing=false",
                "model.flow_model_args.g1.module.condition_encoder.activation_checkpointing=false",
                "model.flow_model_args.g1.module.geometry.activation_checkpointing=false",
                "model.training_args.lr_scheduler=af3",
                "model.training_args.max_lr=0.0001",
                "model.training_args.lr_warmup_no_steps=10000",
                "resume_ckpt_path=null",
                "seed=42",
                f"pretrained_ckpt_path={json.dumps(str(args.checkpoint.resolve())) if args.checkpoint else 'null'}",
                f"hydra.run.dir={json.dumps(str(directory))}",
                "extras.enforce_tags=false",
                "extras.print_config=false",
                "++trainer.precision=32-true",
            ]
            overrides.extend(profile_overrides(profile))
            if profile == "optimized":
                if args.compile or args.compile_blocks:
                    overrides.append("model.flow_model_args.execution.compile_blocks=true")
                if args.compile or args.compile_pair_features:
                    overrides.append("model.flow_model_args.execution.compile_pair_features=true")
            command = [sys.executable, "-u", "src/train.py", *overrides]
            (directory / "command.json").write_text(json.dumps(command, indent=2))
            if args.dry_run:
                print(directory / "command.json")
                continue
            start = time.perf_counter()
            with (directory / "train.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
                )
            if result.returncode:
                raise RuntimeError(
                    f"Benchmark failed ({result.returncode}); inspect {directory/'train.log'}"
                )
            runs.append(
                dict(
                    profile=profile,
                    repeat=repeat,
                    directory=str(directory),
                    process_seconds=time.perf_counter() - start,
                )
            )
            print(f"Completed {directory}", flush=True)
    if args.dry_run:
        return
    report = dict(
        scope="training windows only; excludes validation and full training convergence",
        model=args.model,
        stage=args.stage,
        git_commit=commit,
        reference_profile=args.compare_to,
        runs=summarize(runs, args.compare_to),
    )
    (output / "comparison.json").write_text(json.dumps(report, indent=2))
    print(f"Comparison: {output/'comparison.json'}")


if __name__ == "__main__":
    main()
