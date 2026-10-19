# Local validation: 2026-10-19

Base commit for this refinement: `f75a6e3fe4f62a8de7b36cc931d5267a767dfb2d`.

Environment: Windows, Python 3.12, PyTorch 2.6.0+cpu, Lightning 2.5.4,
NumPy 2.2.6. Validation used an isolated environment outside the repository.
No CUDA device or BWDB server dataset was available locally. The integration
fixture stubs only the optional MLIP dependency; the Flow, DiT, G1 and Lightning
training code are real.

## Completed checks

- Full regression invocation: 351 passed, one failed, 6 warnings, 164.65 seconds.
  The failure was the independent-condition test oracle not accepting the new
  optional `c_cond` keyword. Its signature was updated and explicitly asserts
  `c_cond is None` during training. The failed test then passed in a targeted
  rerun (1 passed, 12.22 seconds), including all shared-encoder gradients.
  All 352 collected checks therefore have a passing result; there are no
  outstanding failures. The original failure and rerun are recorded rather than
  reported as a clean single invocation.
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
- Shared AMP pair cast: 24 independent projection branches preserve every input,
  LayerNorm-affine and projection gradient, with exact CPU equality to separate
  casts. The real 24-layer DiT saves only one distinct low-precision pair storage.
- Decoder broadcast mask: exact CPU outputs and gradients in FP32 and BF16.
- Current versus previous math paths: exact CPU full Flow outputs and all parameter
  gradients, baseline/G1, FP32/BF16, multiplicity 2. Reference versus math still
  uses tolerances because algebraic folding changes floating-point operation order.
- CPU-prepared layouts: nonprefix masks, all-padding batches, group capacities;
  compact featurizer/collator retains the same used features and RNG state.
- Deferred checks still reject invalid values and restore scope after exceptions.
- Sampling cache: baseline/G1, Euler/SDE, FP32/BF16, multiplicity 2 and final
  refinement. Static encoding runs once instead of four times for a three-step
  solve plus refinement; geometry still runs each step. The next sample call
  recomputes its cache. Training and cross-batch cache reuse are rejected.
- Shared AMP cast also works with activation checkpointing and eager compile
  capture. Disabling regional compilation removes wrappers without changing keys.
- Checkpoint prediction entry covers reference/math/optimized profiles with all
  three G1 override choices; saved model weights stay identical.
- Isolated attention diagnostic exercises real CPU profiling and pair-bias input /
  parameter gradients. CPU results have no GPU memory value.
- Optional low-precision learned pair storage has finite BF16 CPU gradients;
  it remains disabled in both shipped execution profiles.

Full small architecture, same initialization seed:

| Model | Trainable parameters | Initial weights / state_dict keys |
| --- | ---: | --- |
| Baseline small | 38,376,320 | Identical across execution profiles |
| G1 small | 40,525,216 | Identical across execution profiles |

Shared standardization reduces distinct pair activation storages. The refinement
also reduces the real 24-layer DiT's distinct BF16 pair cast storages from per-layer
copies to one. These are structural checks, not GPU peak-memory measurements.

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
