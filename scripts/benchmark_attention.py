"""Diagnose attention kernels and fused projection costs; not a whole-model benchmark.

Run on the server, for example:
  python scripts/benchmark_attention.py --atoms 200 --batch 32 --pair-dim 128
Use --pair-dim 0 --heads 4 for the decoder. Unsupported kernels are reported,
never replaced by a different backend or a detached/dropped pair bias.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch

from src.models.simple.attention import AttentionPairBias


def measure_case(
    *,
    device,
    batch,
    atoms,
    dim,
    heads,
    pair_dim,
    amp,
    fused,
    backend,
    warmup,
    iterations,
    compile_backend=None,
    profile=True,
):
    torch.manual_seed(42)
    model = AttentionPairBias(dim, pair_dim or None, heads, attention_impl="pytorch").to(device)
    model.native_sdpa = True
    model.fused_qkvg = fused
    model.broadcast_masks = True
    model.sdpa_backend = backend
    if compile_backend is not None:
        model.compile(backend=compile_backend, dynamic=True)
    x = torch.randn(batch, atoms, dim, device=device, requires_grad=True)
    z = (
        torch.randn(batch, atoms, atoms, pair_dim, device=device, requires_grad=True)
        if pair_dim
        else None
    )
    mask = torch.ones(batch, atoms, device=device)
    if atoms > 1:
        mask[0, -1] = 0

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def step():
        model.zero_grad(set_to_none=True)
        x.grad = None
        if z is not None:
            z.grad = None
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            out = model(x, mask, z)
            value = out.float().square().mean()
        value.backward()
        return value

    sync()
    start = time.perf_counter()
    for _ in range(warmup):
        step()
    sync()
    warmup_seconds = time.perf_counter() - start
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(iterations):
        value = step()
    sync()
    seconds = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    grads = [x.grad, *[p.grad for p in model.parameters()]]
    if z is not None:
        grads.append(z.grad)
    if not torch.isfinite(value) or any(g is None or not torch.isfinite(g).all() for g in grads):
        raise RuntimeError("Non-finite loss or missing/non-finite gradients")
    operators = []
    kernels = []
    if profile:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities) as trace:
            step()
            sync()
        operators = sorted(
            {
                e.key
                for e in trace.key_averages()
                if any(k in e.key for k in ("scaled_dot_product", "flash", "efficient", "cudnn"))
            }
        )
        kernels = sorted({e.name for e in trace.events() if str(e.device_type).endswith("CUDA")})
    return dict(
        seconds_per_forward_backward=seconds / iterations,
        warmup_seconds=warmup_seconds,
        peak_allocated_bytes=peak,
        all_gradients_finite=True,
        pair_bias_gradient_checked=bool(pair_dim),
        attention_operators=operators,
        cuda_kernels=kernels,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--atoms", type=int, default=200)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--pair-dim", type=int, default=128)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument(
        "--backends",
        nargs="+",
        choices=["auto", "math", "efficient", "flash", "cudnn"],
        default=["auto", "math", "efficient", "flash", "cudnn"],
    )
    p.add_argument("--compile", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if (
        min(args.batch, args.atoms, args.dim, args.heads, args.warmup, args.iterations) < 1
        or args.dim % args.heads
        or args.pair_dim < 0
    ):
        p.error("Positive dimensions/iterations and divisible heads are required")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        p.error("CUDA unavailable; no GPU performance result can be produced")
    output = args.output or ROOT / "logs" / f"attention_{datetime.now():%Y%m%d_%H%M%S}.json"
    if output.exists():
        p.error("Output already exists")
    torch.set_float32_matmul_precision("high")
    records = []
    for amp in (False, True):
        for fused in (False, True):
            for backend in args.backends:
                label = dict(
                    precision="bf16" if amp else "fp32", fused_qkvg=fused, backend=backend
                )
                try:
                    result = measure_case(
                        device=device,
                        batch=args.batch,
                        atoms=args.atoms,
                        dim=args.dim,
                        heads=args.heads,
                        pair_dim=args.pair_dim,
                        amp=amp,
                        fused=fused,
                        backend=backend,
                        warmup=args.warmup,
                        iterations=args.iterations,
                        compile_backend="inductor" if args.compile else None,
                    )
                    label.update(status="ok", **result)
                except RuntimeError as error:
                    label.update(status="failed", reason=str(error))
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                records.append(label)
                print(
                    json.dumps({k: v for k, v in label.items() if k != "cuda_kernels"}), flush=True
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        json.dump(
            dict(
                scope="single-attention forward/backward; excludes optimizer, DDP and full model",
                torch_version=torch.__version__,
                device=str(device),
                gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                cases=records,
            ),
            f,
            indent=2,
        )
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
