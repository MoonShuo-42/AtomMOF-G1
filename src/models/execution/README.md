# Whole-model execution optimization

The encoder, 24-layer DiT, decoder, G1 bridge positions, routing policy, losses,
learned parameter shapes and state_dict keys are preserved. Training uses the
baseline precision policy: `32-true`, with the existing FP32/FP64 geometry and
normalization safeguards. No training data, epochs or accumulation are reduced.

## Profiles and precision

- No execution override (`execution: null`): reference implementation.
- `execution=math`: shared pair standardization, fused Q/K/V/gate projections,
  folded G1 readback, shared masked geometry layout, vectorized time-bin logging,
  broadcast decoder masks, reduced redundant copies, CPU-prepared static G1
  layouts, batched device checks and sampling-only static condition caching.
  Unused token maps and dense bond-index matrices are omitted from BWDB batches;
  the bond-type matrix is retained.
- `execution=optimized`: the math profile plus fused AdamW, DDP gradient bucket
  views and persistent DataLoader workers. All learned computations use the same
  precision policy as the baseline; this profile does not enable mixed precision.

Both profiles explicitly set `trainer.precision: 32-true`. Append either profile
to the original full training command. If that command has
`+trainer.precision=32-true`, replace it with `++trainer.precision=32-true` (or omit
it), because the profile now defines that key. CLI overrides still take precedence;
keep `32-true` for these comparisons. Stage data/batch/epoch settings and
initialization checkpoints remain those of the original command.

Baseline and G1 use the same profiles. Keep the G1 enabled flag explicit. The
existing `torch.set_float32_matmul_precision("high")` in `src/train.py` is
unchanged; CUDA may use TF32 internally, just as in the original baseline.
`32-true` therefore does not mean TF32 has been disabled. All comparison runs must
use the same matmul and TF32 flags. No new precision reduction is introduced.

The previously added mixed-precision content path, low-precision pair storage,
shared AMP cast and native-low-precision SDPA switch have been removed from code
and configuration. Obsolete execution keys are rejected instead of silently kept.
G1's original explicit FP32/FP64 guards remain in place.

`model.flow_model_args.execution.compile_blocks=true` optionally compiles DiT
blocks with dynamic shapes. `compile_pair_features=true` optionally compiles the
original dense and G1 compact pair feature construction. Both are OFF by default
until CUDA timings and numerical checks establish a benefit. Compilation does not
rename checkpoint parameters. Geometry search stays eager. The benchmark
`--compile` enables both regional options, and compilation/recompilation cost must
be counted. The old whole-Flow `compile_model` option is not needed.

## Algebra and cache lifetimes

The same pair tensor z enters every layer in a DiT stack. Only its parameter-free
standardization is shared, grouped by epsilon and normalized shape. Each layer
retains its own gamma, beta and projection W. Effective W*gamma and W*beta are
computed differentiably in each forward; no learned parameter or gradient is
removed. Autograd saves one standardized pair tensor instead of per-layer
128-channel affine outputs. There is no low-precision cached copy.

G1 readback pools SiLU hidden values per attention head before the final linear
projection. Its output bias is multiplied by the actual attention mass, including
empty/padded rows. This is algebraically equivalent for arbitrary weights, and
all original value parameters remain. Calculation stays at least FP32.

G1 E is masked and arranged once as (B,N*16,N) in the current forward. Four bridges
reuse this layout, but recompute their routing, U and Z. Dynamic geometry contexts
are not reused across diffusion steps or optimizer updates. Coordinate and cell
gradients remain connected. Geometry cutoffs, search precision, the 125-image cap
and whole-sample fallback rules are unchanged.

CPU workers prepare same-block pair indices, the allowed-pair mask, atom counts
and the batch maximum count. These depend only on static features. This removes
GPU nonzero compaction and the host read for group storage size, without changing
the per-sample group count. Custom callers without metadata use the original
on-device layout calculation. Readback validity predicates are still evaluated
for every bridge; the successful forward reads their combined result once.
Invalid inputs still raise an error.

In eval/no-grad mode, static original atom c/p and the shared G1 condition encoding
are reused for one sample() call, including final refinement. The cache is an
explicit local object, rejected in training or for another model/batch/G1 setting,
and never retained on the module. No current coordinates, lattice, periodic
geometry, routing or group aggregates are cached across timesteps.

The reference implementation remains selectable for execution regression tests;
this does not restore obsolete G1 architectures. Floating-point operation order
can still change rounding, so mathematical equivalence is not bitwise identity.

## Attention diagnosis

`sdpa_backend=auto` lets PyTorch select the implementation. Other choices are
`math`, `efficient`, `flash`, `cudnn`. Attention retains the original FP32 Q/K/V and
mask input conversion. A forced unsupported backend raises an error; the code
never casts to lower precision or drops pair bias to make a backend work.
The original model already uses SDPA, so calling it is not itself an optimization.

Decoder padding masks use a broadcast (B,1,1,N) representation instead of a
materialized (B,H,N,N) tensor. Pair-biased attention retains its learned bias;
xformers retains its required mask layout.

The isolated diagnostic checks all gradients, including pair bias, and records
actual attention operators and CUDA kernel names. It compares fused/separate
Q/K/V/gate projections across backends using FP32 only. Unsupported cases are
reported as failures. Add `--compile` to measure Inductor compilation; its warmup
cost is reported separately.

```bash
python scripts/benchmark_attention.py --batch 32 --atoms 200 --pair-dim 128
python scripts/benchmark_attention.py --batch 32 --atoms 200 --pair-dim 0 --heads 4
```

Output is a new logs/attention_*.json file. This operator diagnostic does not
establish a whole-model or end-to-end training speedup.

## Server benchmark

```bash
cd /data/ai4science/AtomMOF-G1
python scripts/benchmark_execution.py \
  --lmdb-dir /data/ai4science/AtomMOF/AtomMOF-main/data/bwdb \
  --model g1 --stage 1 --batches 200 --warmup 20 --repeats 3
```

Repeat for stages 2 and 3 and `--model baseline`. Defaults use 8 visible L40S GPUs
and the actual stage atom/batch/accumulation budgets. No dataset sample limit is
applied. `--batches 0` measures full epochs; `--checkpoint PATH` loads the same
architecture's weights for every compared profile. Output uses a new directory
and never modifies training checkpoints.

All reference/math/optimized runs use 32-true, the same seed and rotated execution
order. Worker persistence is disabled equally in this diagnostic launcher so that
worker RNG lifecycles match across profiles, including multiple epochs. Optimized
training enables it; measure its end-to-end effect separately, and note that
persistent workers can change the sequence of random augmentations across epochs.

Reports record batch IDs, shapes, update counts, forward/backward/optimizer/DDP
wall time, peak allocated/reserved CUDA memory, precision and matmul/TF32 settings.
Speed comparisons are refused if workloads or precision settings differ. Missing
precision evidence is also rejected. CPU memory cannot satisfy the GPU target.

The joint target remains speedup >= 2.0 and peak allocated ratio <= 0.5 under the
same workload and precision. Peak includes optimizer warmup. Process wall time
includes startup/compilation. Diagnostic runs disable validation, checkpointing
and W&B equally, so they do not establish full end-to-end training speedup. Full
training/validation/save runs require separate measurement. Stage 3 must cover
large structures; 200 batches do not establish the largest dataset-wide peak.

## Inference and validation

src/predict.py accepts both execution profiles with 32-true. Runtime options are
applied to the loaded model while architecture and weights come from its
checkpoint. `predict_args.g1_enabled=null` keeps the checkpoint G1 setting. For
existing G1 checkpoints, add `data.prepare_g1_metadata=true` for CPU layouts:
the ordinary model config defaults to G1 disabled even when the checkpoint has
G1. Without metadata, the on-device fallback remains correct.

Prediction directory names are unchanged; use a new run directory for profile
comparisons to avoid mixing raw files. Reapplying a profile removes disabled
regional compilation wrappers without changing checkpoint parameter names.

Run `python -m pytest tests -q`. Checks cover full Flow/G1 outputs, gradients and
optimizer updates, checkpoint keys, data/RNG equivalence, layout/caching scope,
FP32 activations and gradients, sampling, two-process DDP and precision-matched
benchmark commands. See [VALIDATION.md](VALIDATION.md) for actual local results.

CPU tests and saved-tensor counts do not establish CUDA performance. No halving of
training time or GPU memory has been demonstrated locally. Baseline/G1 comparisons
must retain a controlled precision, training and evaluation protocol. Short-run
correctness does not establish full convergence or MR/RMSD equivalence.
