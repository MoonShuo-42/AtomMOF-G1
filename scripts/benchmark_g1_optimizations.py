"""Compare the four G1 execution optimizations on synthetic inputs.

Run separately from training. CPU timings do not predict multi-GPU epoch time.
The dense condition path shares weights with the optimized one. Geometry uses
identical candidates; group-descriptor references compute the same formulas.
"""
import argparse
import json
import statistics
import sys
import time
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import torch
from reference_g1_condition import dense_condition_reference

from src.models.g1.conditioning import ConditionEncoder
from src.models.g1.geometry import PeriodicGeometry
from src.models.g1.geometry.groups import prepare_group_internal
from src.models.g1.group_attention import GroupGeometryFeatures
from src.models.g1.readback import ReadbackGeometryFeatures
from src.models.g1.routing.groups import AtomCountGroups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--atoms", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.batch_size, args.atoms, args.block_size, args.iterations) < 1 or args.warmup < 0:
        parser.error("sizes/iterations must be positive and warmup nonnegative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu only for a CPU microbenchmark")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    B, N, K = args.batch_size, args.atoms, min(32, args.atoms)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    blocks = (torch.arange(N, device=device) // args.block_size)[None].expand(B, -1)
    feats = dict(
        ref_pos=torch.randn(B, N, 3, device=device),
        ref_charge=torch.zeros(B, N, device=device),
        ref_element=torch.full((B, N), 6, dtype=torch.long, device=device),
        block_type=torch.ones(B, N, dtype=torch.long, device=device),
        block_index=blocks,
        atom_pad_mask=mask,
        bond_types=torch.zeros(B, N, N, dtype=torch.long, device=device),
    )

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def measure(fn, module=None, leaves=()):
        def step():
            if module is not None:
                module.zero_grad(set_to_none=True)
            for leaf in leaves:
                leaf.grad = None
            with torch.set_grad_enabled(args.backward):
                outputs = fn()
                if isinstance(outputs, torch.Tensor):
                    outputs = (outputs,)
                losses = [v.square().mean() for v in outputs if v.requires_grad]
                if args.backward and losses:
                    sum(losses).backward()

        for _ in range(args.warmup):
            step()
        synchronize()
        baseline = 0
        if device.type == "cuda":
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        samples = []
        for _ in range(args.iterations):
            synchronize()
            begin = time.perf_counter()
            step()
            synchronize()
            samples.append(1000 * (time.perf_counter() - begin))
        result = {"median_ms": statistics.median(samples)}
        if device.type == "cuda":
            result["peak_extra_mib"] = (
                torch.cuda.max_memory_allocated(device) - baseline
            ) / 2**20
        return result

    records = {}

    def compare(name, reference, optimized, ref_module=None, opt_module=None, leaves=()):
        with torch.no_grad():
            expected, actual = reference(), optimized()
            if isinstance(actual, torch.Tensor):
                expected, actual = (expected,), (actual,)
            for a, b in zip(actual, expected):
                # CUDA 'high' matmul precision may choose different GEMM shapes.
                torch.testing.assert_close(a, b, rtol=3e-3, atol=3e-4)
        ref = measure(reference, ref_module, leaves)
        opt = measure(optimized, opt_module, leaves)
        records[name] = {
            "reference": ref,
            "optimized": opt,
            "speedup": ref["median_ms"] / opt["median_ms"],
        }
        print(name + ": " + json.dumps(records[name]), flush=True)

    local = ConditionEncoder().to(device)
    dense = deepcopy(local)
    compare(
        "condition_pairs",
        lambda: dense_condition_reference(dense, feats),
        lambda: local(feats),
        dense,
        local,
    )
    del dense, local

    x = torch.rand(B, N, 3, device=device).requires_grad_(args.backward)
    cell = (torch.eye(3, device=device)[None].repeat(B, 1, 1) * 1.4).requires_grad_(args.backward)
    old_geometry = PeriodicGeometry(pair_chunk_size=4096).to(device)
    new_geometry = PeriodicGeometry().to(device)

    def geometry(module):
        output = module.from_cell(x, cell, mask)
        return output.E, output.rho

    compare(
        "geometry_chunks",
        lambda: geometry(old_geometry),
        lambda: geometry(new_geometry),
        leaves=(x, cell),
    )

    Z = (torch.randn(B, K, K, 16, device=device) * 0.1).requires_grad_(args.backward)
    U = (torch.randn(B, N, K, 16, device=device) * 0.1).requires_grad_(args.backward)
    groups = torch.ones(B, K, dtype=torch.bool, device=device)
    group_features, readback_features = (
        GroupGeometryFeatures(),
        ReadbackGeometryFeatures(),
    )

    def descriptors(shared):
        internal = prepare_group_internal(Z, groups) if shared else None
        return (
            group_features(Z, groups, internal_geometry=internal),
            readback_features(U, Z, mask, groups, internal_geometry=internal),
        )

    compare(
        "group_internal_reuse",
        lambda: descriptors(False),
        lambda: descriptors(True),
        leaves=(Z, U),
    )

    selector = AtomCountGroups()

    def routing(shared):
        if shared:
            prepared = selector.prepare(mask)
            return (prepared.mask,) * 4
        return tuple(selector.prepare(mask).mask for _ in range(4))

    compare("four_bridge_layouts", lambda: routing(False), lambda: routing(True))
    metadata = {
        "torch": torch.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "batch_size": B,
        "atoms": N,
        "block_size": args.block_size,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "backward": args.backward,
        "note": "Component microbenchmarks, not complete training or DDP throughput.",
        "results": records,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
