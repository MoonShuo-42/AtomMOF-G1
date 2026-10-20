# Validation: full-precision operator execution

Working-tree changes based on `e509a4a78cfa3c9398ed0c09d9f524383eda398f`,
validated on 2026-10-20. Local environment: Windows, Python 3.12,
PyTorch 2.6.0+cpu, Lightning 2.5.4. The isolated environment is outside the
repository; server dependencies were not modified. No CUDA device or BWDB
server dataset is available locally. Only the optional unused MLIP backend is
stubbed in integration fixtures; Flow, DiT, G1, Lightning and DDP are real.

## Executed checks

- Full suite: **382 passed, 4 warnings, 133.99 seconds**. No failures or skips.
  Warnings concern deliberately absent validation loaders and small CPU-test
  worker counts.
- Existing execution/regression suite with the new runtime options: 53 passed.
- 29 additional cases cover independent operator formulas, derivatives, runtime
  resets, full-size parameter budgets, benchmark CLI composition and profiler
  construction. These are included in the full-suite result.
- Black checks passed for 11 changed execution/G1/script/test files.
  Legacy source formatting outside edited blocks was retained. Diff whitespace
  checks passed.

## Architecture and precision

Full configured small models were instantiated, not extrapolated from a toy
width. Parameter totals before and after runtime configuration are:

| Model | Trainable parameters |
| --- | ---: |
| Baseline small | 38,376,320 |
| G1 small | 40,525,216 |

Checkpoint keys, shapes and dtypes remain unchanged. Runtime configuration does
not modify initialization weights. Existing strict checkpoint reload checks and
actual Lightning checkpoint save/load pass. The four bridge insertion points,
independent bridge parameters, shared condition encoder, shared current geometry,
losses and training budgets are retained. Grouped pair projection is an execution
batching strategy, not weight tying.

Both profiles retain `32-true`, original FP32 attention inputs and baseline TF32
policy. Saved floating activations, outputs, parameters and gradients are checked
as FP32/FP64. Float64 search/cutoff guards remain. No half/bfloat16 or new TF32
settings were introduced. Floating-point operation ordering changes can alter
rounding; output/gradient/optimizer comparisons use explicit tolerances rather
than claiming bitwise-identical training trajectories.

## Semantic coverage

- Whole Flow baseline/G1 outputs, all parameter gradients, one AdamW update,
  multiplicity 1/2, initialization and strict checkpoint interfaces.
- Fused geometric pair projection against separate float64 projections,
  including every learned projection gradient, coordinate gradients and empty
  compact layouts. Unmasked bond/single terms retain the original behavior.
- Grouped pair projections with independent affine offsets, bias/no-bias
  projections, unequal head counts, and parameter/input gradients. Whole DiT
  tests cover chunk sizes 2/4/6/32, mixed normalization epsilons, layer hooks,
  all-padding/nonprefix masks and activation checkpointing.
- Packed symmetric descriptors versus expanded matrices, including factor-two
  off-diagonal terms, explicit matrix overrides, group/readback descriptors,
  masked NaNs and all input gradients.
- Periodic image buckets preserve the complete selected image list, pair order
  on output, availability/rejection and live coordinate/cell gradients. Tests
  use FP32/FP64, skew cells, singular/over-cap cases, masked atoms, empty batches,
  small chunk budgets and activation checkpointing. A heterogeneous case
  verifies that padded pair-image entries actually decrease.
- Joint time metrics preserve labels, boundary assignments, empty-bin NaNs,
  mixed loss dtypes and detached logging tensors.
- Real two-process Gloo DDP, bucket views, complete gradients and synchronized
  optimizer updates; actual CPU Lightning training and checkpointed sampling.
- Sampling-only static caches, original Euler/SDE inference behavior and
  regional compilation eager capture remain covered. CUDA NCCL and Inductor
  execution still require server checks.
- Same-work benchmark commands compose for previous/current/bucketed execution,
  including stage-1 batch 64/accumulation 1 and separate pair-feature compilation.
  Reference omits the execution group override; no `execution=null` CLI is used.

## User-reported server evidence before these operator changes

The following are the user's two completed stage-1 epochs on 8 L40S GPUs, with
G1 architecture and baseline precision. They were not reproduced by the local
CPU run and do not measure the current new operators.

| Execution | Batch/card | Accumulation | Batches/card | Epoch 1 / 2 training seconds | Peak allocated/card GiB |
| --- | ---: | ---: | ---: | --- | ---: |
| Unoptimized G1 | 32 | 2 | 651 | 209.90 / 210.33 | 31.41 |
| Previous optimized G1 | 32 | 2 | 651 | 157.27 / 154.68 | 16.00 |
| Previous optimized G1 | 64 | 1 | 326 | 131.18 / 129.51 | 30.93 |

Using epoch 2, previous B32 optimization reduced training-segment time by 26.46%
and allocated memory by 49.06%. B64/accumulation 1 reduced time by 38.43% relative
to unoptimized B32, but its allocated peak is 30.93 GiB. The B64 speed result
cannot be combined with the B32 memory result to claim a joint same-work target.
These are training segments, not training+validation+checkpoint wall time.

## Remaining CUDA measurements

There is no measured CUDA speed or memory claim for the new operators yet.
Use the previous/current/bucketed launcher in README.md under identical workloads,
then evaluate regional compilation separately. Image sorting can cost more than
its saved padding; it remains disabled by default. Grouped projections trade a
bounded head-bias intermediate for fewer GEMMs. GPU operator/kernel traces must
be measured separately from acceptance runs because profiling adds overhead.

Acceptance still requires training time <= 50% and peak allocated memory <= 50%
of the reference under the agreed same-work protocol. Stage 2/3 and large
structures require separate measurements. Numerical regression tests do not
establish identical full-run convergence or MR/RMSD.
