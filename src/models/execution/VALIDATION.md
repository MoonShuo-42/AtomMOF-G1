# Validation: full-precision execution optimization

Current local implementation based on `30995e5caabdccf7db86870266fe674e568a5497`,
validated on 2026-10-20. Environment: Windows, Python 3.12, PyTorch 2.6.0+cpu,
Lightning 2.5.4. Dependencies and the isolated validation environment are outside
the repository. No CUDA device or server BWDB dataset is available locally.
Only the unused optional MLIP backend is stubbed in integration fixtures; Flow,
DiT, G1, Lightning, data workers and Gloo DDP are real.

## Executed checks

- Entire test suite: **561 passed, 4 warnings in 225.23 seconds**. No failures or
  skips. Warnings concern intentionally absent validation loaders and small
  CPU-test worker counts.
- This revision adds 127 cases across backbone fusion, G1 storage/readback,
  training I/O, and whole-model integration.
- Whole-model baseline/G1 comparisons cover outputs, coordinate/lattice/time
  input gradients, all parameter gradients, one AdamW update, multiplicity 1/2,
  strict state_dict reload and resetting the execution profile.
- Real two-process Gloo DDP executes the current optimized profile with mixed
  geometry availability across ranks, complete finite FP32 gradients, gradient
  bucket views, and synchronized parameter updates.
- Existing CPU Lightning checkpoint save/load, baseline-without-G1 import,
  four-bridge integration, sampling and optional data preparation regressions
  remain passing.
- Continuous compiled paths use CPU `aot_eager` forward/backward. Packed condition
  attention, image bucketing, chunked readback, pair-feature compilation, token
  pair compilation and geometry moment compilation are tested in combination
  with the new full-model path. DiT block compilation is tested with precomputed
  head-first biases and time modulation, including profile reset.
- `git diff --check` passed. New tests and edited policy/benchmark files were
  formatted with Black (99 columns). Legacy formatting outside edited source
  regions is retained.

## Architecture, precision and data contracts

Full configured small-model parameter budgets remain:

| Model | Trainable parameters |
| --- | ---: |
| Baseline small | 38,376,320 |
| G1 small | 40,525,216 |

No learned parameter is renamed, shared, removed, or replaced. Projection fusion
concatenates original independent Parameters for a single forward; gradients
return to their original optimizer state. Four G1 bridge positions, their
independent parameters, shared condition encoder and shared current geometry
remain unchanged. Runtime options add no persistent checkpoint tensors.

Both math and optimized profiles retain `32-true`, the baseline `high` matmul
setting and existing TF32 policy. Saved floating activations/gradients remain
FP32/FP64. No BF16/FP16 path or new precision policy is enabled. Periodic search,
cutoff and cell guards retain their existing precision and rejection rules.
GEMM shapes/reduction order can change rounding; mathematical equivalence is
validated with explicit tolerances, not assumed to mean bitwise-identical full
training trajectories or identical MR/RMSD.

Data tests cover compact/noncompact fields, mixed dtype promotion, noncontiguous
inputs, absent optional fields, empty samples, bond write order, input gradients,
G1 metadata, worker spawn and the old datamodule positional constructor contract.
Contiguous block augmentation preserves original features and RNG states exactly.
Sample ordering, dynamic budgets, accumulation and optimizer scheduling are not
changed. Four loss means are reused; all 23 training log keys/calls, empty-bin
NaNs and callback/accumulation visibility are retained.

## Operator-specific evidence

- Head-first pair projection agrees with independent Linear oracles and all
  derivatives. Continuous masks, nonprefix/all-padding masks, differing LayerNorm
  epsilon and mask penalties, grouped tail sizes, layer hooks and activation
  checkpointing are covered. A separate CPU allocation trace confirmed the bias
  assembly's former contiguous/clone copy is absent on the new path; the SDPA
  contiguity guard remains in place.
- Time modulation shares only SiLU(time) inside one stack. Its independent
  per-layer affine parameters, time derivatives, callback order and keys remain.
- Readback fusion/chunking covers FP32/FP64, all input/parameter derivatives,
  unavailable geometry, masked NaNs, empty groups, all padding, non-unit attention
  mass, mixed-dtype fallback, invalid membership checks and compiled backward.
  Fresh-buffer reuse is verified not to mutate inputs or parameters. A CPU
  profiler regression confirms two full hidden-sized `aten::add` allocations
  are removed, with locally identical outputs and gradients.
- E(B,N,N,16) and prepared aggregation storage are verified to alias a single
  allocation. Values and coordinate/cell/Pbar gradients agree under skew/singular
  cells, image-cap rejection, padding, image bucketing and checkpointing. Generic
  external E still follows the public mask/NaN protection path.
- Same-work benchmark commands compose with identical precision, batch cap,
  accumulation, epochs and square budget. `optimized_30995` disables only the
  new runtime switches and preallocation in current code; unconditional loss
  reuse and feature slicing remain present on both sides. It is an operator
  ablation, not an exact historical source checkout.

## Remaining server validation

No GPU speed or peak-memory improvement is claimed for this revision. CPU tests
and allocation counts do not establish L40S performance. Use the README's
current-vs-previous-switch benchmark for baseline and G1 in all three stages,
including large structures. Compare complete equal-work reports, batch/shape
hashes, optimizer updates, precision settings, slowest-rank training time and
largest-rank allocated/reserved peaks.

Compilation, packed condition attention, image bucketing and atom chunking
remain opt-in. Measure each separately, including compilation/recompilation cost,
SDPA backend eligibility and backward memory. Chunking bounds temporaries but
can retain similar total saved activations and add launches. GPU profiling must
be separate from acceptance timing because tracing changes timing/lifetimes.

For final model-quality claims, use the same evaluation protocol and compare
full training convergence and test MR/RMSD. Numerical regression tests alone do
not establish those outcomes.
