# Local validation: 2026-10-19

Base commit: `8780c8b4bec007cd8822b1016dc86d0ee848c7e3`.

Environment: Windows, Python 3.12, PyTorch 2.6.0+cpu, Lightning 2.5.4,
NumPy 2.2.6. Validation used an isolated environment outside the repository.
No CUDA device or BWDB server dataset was available locally. The integration
fixture stubs only the optional MLIP dependency; the Flow, DiT, G1 and Lightning
training code are real.

## Completed checks

- Full regression suite: 317 passed, 6 warnings, 116.80 seconds.
- Benchmark comparison checks were subsequently expanded to reject unequal
  warmup lengths, measured batch counts and measured sample counts, in addition
  to unequal optimizer update counts. The benchmark-selected suite then passed
  all 8 checks (4 warnings, 15.38 seconds).
- Reference and algebraic profiles: full Flow outputs, all parameter gradients,
  one AdamW update, masks and multiplicity 1/2, both baseline and G1.
- Readback fold: non-normalized weights, learned output bias, empty groups,
  padding, input gradients and every learned parameter gradient in float64.
- Shared geometry layout: masked NaN values, invalid geometry, E/Pbar gradients.
- BF16 CPU forward/backward: finite outputs and gradients, including G1.
- Real Lightning training with accumulation, checkpoint save/strict reload,
  sampling, and the benchmark callback in FP32 and BF16.
- Two-process Gloo DDP: BF16, gradient bucket views, mixed valid/invalid geometry,
  all parameter gradients, and synchronized parameters after optimizer updates.
- Regional compile graph capture with the eager backend: forward/backward and
  checkpoint names. This does not validate CUDA Inductor compilation.
- Generated reference/math/optimized benchmark commands compose with Hydra.

Full small architecture, same initialization seed:

| Model | Trainable parameters | Initial weights / state_dict keys |
| --- | ---: | --- |
| Baseline small | 38,376,320 | Identical across execution profiles |
| G1 small | 40,525,216 | Identical across execution profiles |

An isolated eight-layer DiT saved fewer distinct pair activation storages with
shared standardization. This is a structural check, not a GPU memory measurement.

## Pending server validation

The targets remain **training time <= 50% and peak allocated CUDA memory <= 50%**
of the matching reference run. Neither target has been established locally.
Run the supplied benchmark for baseline/G1 and all three stages, including full
stage-3 coverage. Inspect actual attention backends and compilation costs on
PyTorch 2.6 / L40S before enabling regional compilation in production.

The diagnostic launcher measures training windows with matched workloads;
validation, checkpointing and full process overhead require separate end-to-end
measurements. BF16 changes numerical rounding. Full training convergence and
MR/RMSD comparisons must use a controlled evaluation protocol; finite short-run
losses do not establish equivalent generation quality.
