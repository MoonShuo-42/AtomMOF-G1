# Local validation: baseline-precision execution

Base commit: `b56484df25de2d68a323d4badbe305bed9125c54`.
Validated on 2026-10-19 using Windows, Python 3.12, PyTorch 2.6.0+cpu,
Lightning 2.5.4 and NumPy 2.2.6 in an isolated environment outside the repository.
No CUDA device or BWDB server dataset was available. The integration fixture
stubs only the optional MLIP dependency; Flow, DiT, G1 and Lightning are real.

## Results

- Targeted execution/regression suite: 53 passed, 2 warnings, 40.86 seconds.
- Full suite: **353 passed, 4 warnings, 126.22 seconds**. No failures or skips.
- Attention diagnostic CLI: separate and fused Q/K/V/gate projection cases both
  passed with 32-true, real CPU profiling and finite pair-bias/input/parameter
  gradients. GPU peak memory is null on CPU; no speed claim is made from these
  small CPU cases.
- Formatting checks passed for the changed execution utilities, benchmark scripts
  and regression tests. `git diff --check` passed.

## Precision and model checks

Both execution profiles compose with `trainer.precision=32-true` for training and
prediction. The original train.py matmul policy is unchanged. Runtime policy
configuration does not modify matmul precision, initialization weights or
state_dict keys. Baseline/G1 full Flow saved activations contain only FP32/FP64;
model parameters, outputs and parameter gradients remain FP32 in these tests.
All original high-precision geometry and G1 guards are retained.

Lower-precision content/pair/cast/native-SDPA execution options and their code
paths are removed. Tests explicitly reject those obsolete flags. The attention
benchmark has no mixed-precision branch. The training benchmark generates the
same FP32 precision for reference, math and optimized runs, and refuses speed
comparisons when precision, matmul policy, TF32 flags or workload evidence differ.

Full small architecture parameter budgets remain:

| Model | Trainable parameters |
| --- | ---: |
| Baseline small | 38,376,320 |
| G1 small | 40,525,216 |

## Retained optimization coverage

- Full Flow reference/math outputs, all parameter gradients, one optimizer update,
  checkpoint strict reload, masks and multiplicity 1/2, baseline and G1.
- Independent pair normalization affine/projection gradients; shared normalized
  pair storage without any low-precision copy.
- Folded readback with non-unit weights, output bias, empty groups, padding and
  all input/parameter gradients against a float64 mathematical reference.
- Shared geometry contraction layout with masked NaNs and E/Pbar gradients.
- FP32 decoder broadcast masks preserve outputs and gradients.
- Compact featurizer/collator preserves used features and RNG state; CPU G1 layouts
  handle nonprefix masks and all-padding batches without changing group counts.
- Deferred validity checks retain errors and restore scope after exceptions.
- Sampling cache: baseline/G1, Euler/SDE, multiplicity 2 and final refinement.
  Static conditions are computed once per solve, dynamic geometry every step;
  the next sample call constructs a fresh cache. Training reuse is rejected.
- Real FP32 Lightning training with accumulation, checkpoint save/reload and
  sampling. Two-process Gloo DDP with gradient bucket views, mixed geometry
  validity, all gradients and synchronized parameters after optimizer updates.
- Regional compile eager capture and activation checkpointing preserve gradients
  and checkpoint names. CUDA Inductor execution has not been measured locally.
- Prediction entry preserves checkpoint architecture/weights and explicit G1
  overrides for both execution profiles.

## Remaining server validation

The targets are still training time <= 50% and peak allocated CUDA memory <= 50%
of the reference, now strictly under the same baseline precision policy. Neither
has been established locally. Run the supplied benchmarks on L40S for both models
and all three stages, including large structures in stage 3.

Operator fusion/reordering may change floating-point rounding despite unchanged
precision; tests use explicit tolerances where needed. Short-run correctness
cannot establish full convergence or equivalent MR/RMSD. Training-window timing
excludes validation/checkpointing; a full end-to-end timing comparison is also
required before claiming halved training time.
