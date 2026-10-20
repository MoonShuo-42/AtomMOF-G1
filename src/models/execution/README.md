# Whole-model execution optimization

The encoder, 24-layer DiT, decoder, G1 bridge positions, routing policy, losses,
learned parameter shapes and state_dict keys are preserved. Training uses the
baseline precision policy: `32-true`, with the existing FP32/FP64 geometry and
normalization safeguards. No training data, epochs or accumulation are reduced.

## Profiles and precision

- Omit the execution CLI override: reference implementation. The YAML default is
  `execution: null`; do not pass `execution=null` on the command line, because
  Hydra parses that as an invalid config-group override.
- `execution=math`: shared pair standardization, fused Q/K/V/gate projections,
  folded G1 readback, shared masked geometry layout, vectorized time-bin logging,
  broadcast decoder masks, reduced redundant copies, CPU-prepared static G1
  layouts, batched device checks and sampling-only static condition caching.
  It also enables fused geometric pair projections, four-layer pair-bias
  projection groups, packed symmetric-moment contractions and joint loss bins.
  Token pair addition reuses a fresh temporary, fixed time features are reused
  within one forward, and prepared U storage is contracted directly into Z.
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

## Operator controls

Runtime options live under `model.flow_model_args.execution`. They change no
learned parameter, checkpoint key or architecture configuration. New defaults in
both math and optimized profiles are:

| Option | Profile value | Work reduced |
| --- | --- | --- |
| `reuse_pair_buffer` | `true` | Add the input pair tensor into the fresh single-to-pair sum, removing one full pair allocation without mutating inputs. |
| `cache_time_features` | `true` | Cache fixed frequencies and reuse raw time features within one forward. The three original time MLPs and four routers remain independent. |
| `aggregation_layout_reuse` | `true` | Compute Z directly from the prepared U storage before exposing the public U layout, avoiding the atom/group/channel copy. Requires the prepared geometry path for benefit. |
| `packed_block_attention` | `false` | Run ConditionEncoder QK/softmax/value attention in same-component length buckets; restore original atom order. |
| `compile_geometry_moments` | `false` | Compile only continuous periodic-moment reductions; FP64 cutoff decisions remain eager. |
| `compile_readback` | `false` | Compile the folded relation-value and pooling core, with validation and masks outside. |
| `compile_token_pairs` | `false` | Compile token pair construction and its original normalization/projection. |
| `fused_pair_features` | `true` | Concatenate the existing position/distance/mask projection weights; one five-channel input projection replaces three wide projections in original AtomMOF and compact G1 conditioning. |
| `pair_bias_chunk_size` | `4` | Several independent per-layer pair projections share one GEMM; each layer retains its own LayerNorm affine and head projection weights. Set to 1 for the previous per-layer path. Requires `pair_norm_reuse=true`. |
| `packed_geometry` | `true` | Compute chi and pair/internal trace contractions directly from six symmetric Q entries, avoiding large pairwise 3x3 expansions. Small group-internal matrices and their existing directional contractions remain. |
| `joint_time_metrics` | `true` | Assign time bins/counts once for all logged losses; preserve labels, empty-bin NaNs and loss dtypes. |
| `bucket_geometry_images` | `false` | Optional grouping of already accepted pairs by image counts, with power-of-two storage caps and restoration of original pair order. |

Pair projection groups respect differences in normalization epsilon. Grouping
never shares learned weights or moves/removes G1 hooks. Projected biases are
computed within the current forward and bounded to one group at a time; this
trades additional head-sized intermediate storage for fewer GEMMs. Every bias
and normalization parameter remains differentiable. Bond and single-conditioning
terms are not restricted to same-block pairs in the original AtomMOF encoder.

Packed contractions include both copies of off-diagonal entries. Explicitly
supplied matrix overrides retain their semantics. Existing FP32/FP64 protection,
padding and unavailable-geometry behavior are unchanged.

Image buckets do not truncate, approximate or drop periodic images. Float64
search, cutoff tests, the image cap, whole-sample rejection and coordinate/cell
gradients are preserved. Sorting and a histogram host read add overhead, so
bucketed execution needs its own CUDA measurement before adoption. In particular,
a small-image-count batch may gain nothing. The existing compile flags remain
false by default; there is no forced SDPA backend or precision reduction.

## Time features, block packing and continuous-core compilation

Time caches contain no learned projection outputs. A context-local cache lives
for exactly one Flow forward, preserves time gradients and is cleared even on
exceptions. G1 bridge checkpoint regions compute their own time features so
backward recomputation does not depend on another bridge's graph. Fixed-frequency
caches contain only constants, are keyed by device/calculation dtype/config,
and never add state_dict entries. They remain safe after inference followed by
training or module dtype/device changes.

The packed attention option changes only the G1 condition encoder's attention
execution. It keeps all same-component pairs and the original pair bias, feature
gate, FFN, masks and parameters. Components from different samples are separate,
even when block IDs match. CPU workers group components by power-of-two padded
length (capped at N), with one batched attention call per bucket. Outputs scatter
back into the original atom order. All-padding samples retain zero parameter
gradient connections for DDP. `data.prepare_g1_blocks` follows the execution
switch in both profiles. Direct callers without prepared metadata use a CPU
fallback, which can synchronize CUDA; worker preparation is preferred for speed.
The metadata must be regenerated if atom order, mask or block membership changes.

New compilation switches are independent of `compile_blocks` and
`compile_pair_features`. Each wraps a bound continuous-core method without
renaming parameters; disabling it removes the wrapper. The geometry search and
FP64 inside-sphere decisions are never compiled. No SDPA backend, precision,
cutoff, loss or model dimensions are changed. CPU `aot_eager` tests validate graph
capture and compiled backward; they are not CUDA Inductor benchmarks.

Packing and compilation stay opt-in pending GPU measurements. The baseline
execution and all five new paths can be compared in one launcher run:

```bash
cd /data/ai4science/AtomMOF-G1
export PROJECT_ROOT=/data/ai4science/AtomMOF-G1
python scripts/benchmark_execution.py \
  --lmdb-dir /data/ai4science/AtomMOF/AtomMOF-main/data/bwdb \
  --model g1 --stage 1 --batch-size 64 --accumulate-grad-batches 1 \
  --profiles optimized_40ea optimized packed compiled_operators all_operators \
  --compare-to optimized_40ea \
  --batches 0 --epochs 3 --warmup 20 --repeats 2
```

`optimized_40ea` reproduces the execution options of commit 40ea1df;
`optimized` enables the three new default options; `packed` adds block attention;
`compiled_operators` adds the three new compilation regions; `all_operators`
combines both. These are execution variants of the same weights and architecture.
`optimized_previous` still means the earlier e509a4a options, not 40ea1df.
The user's 129.51-second result predates 40ea1df and is not a measurement of it.
The launcher uses independent directories, records each command, and refuses
comparisons with different workload hashes or precision flags. Compilation can
increase startup/recompilation time or memory; include process_seconds and later
epochs rather than reporting only a favorable steady window. No automatic
fallback silently disables a failing compiled core.

For an existing training command, `execution=optimized` enables the three default
options. To attempt all five changes, additionally set:

```bash
model.flow_model_args.execution.packed_block_attention=true \
model.flow_model_args.execution.compile_geometry_moments=true \
model.flow_model_args.execution.compile_readback=true \
model.flow_model_args.execution.compile_token_pairs=true
```

The isolated pair-addition test eliminates one B*N*N*token_z allocation. At
B=64, N=200 and token_z=128 in FP32 this buffer is about 1.22 GiB. This is a local
allocation reduction, not a promised 1.22 GiB reduction in whole-training peak.
Packing savings depend on the actual component sizes; a single large component
may add overhead without reducing attention work.

## Compare against the earlier e509a4a implementation

The launcher supports `optimized_previous` (operator settings used at e509a4a),
`optimized` (current operators), and `bucketed` (current plus image buckets).
This is an execution comparison; it does not load an older model architecture.

For the user's stage-1 per-GPU batch 64, accumulation 1 experiment:

```bash
cd /data/ai4science/AtomMOF-G1
export PROJECT_ROOT=/data/ai4science/AtomMOF-G1
python scripts/benchmark_execution.py \
  --lmdb-dir /data/ai4science/AtomMOF/AtomMOF-main/data/bwdb \
  --model g1 --stage 1 --batch-size 64 --accumulate-grad-batches 1 \
  --profiles optimized_previous optimized bucketed \
  --compare-to optimized_previous \
  --batches 0 --epochs 3 --warmup 20 --repeats 2
```

Run without the batch/accumulation overrides for the original stage budget.
All compared profiles use the same batch cap, square budget, accumulation, data,
precision and worker lifecycle. Reports retain per-rank full training segment
time and peaks; comparison ratios use the measured post-warmup window. The
launcher disables validation, checkpointing and W&B in all profiles. These
results must not be presented as full training/validation/save wall time.

To test regional compilation independently, use
`--profiles optimized_previous optimized --compare-to optimized_previous` and add either
`--compile-pair-features` or `--compile-blocks`. These flags affect only the
optimized variant. `--compile` enables both. Account for compilation startup and
shape recompilation; successful eager capture on CPU is not validation of CUDA
Inductor performance or numerical behavior.

For diagnosis separately, add `profiler=execution` to a short training command in
a new output directory. It records module names, operator shapes and memory over
three active profiler steps after 20 wait and 2 warmup steps, exporting per-rank
traces under `profiler/`. Run enough batches to reach that window, and disable
validation/checkpointing equally when comparing traces. Do not use instrumented
timing or memory values for acceptance: profiling changes allocation lifetimes
and adds overhead. Inspect actual CUDA kernels, device copies/synchronization,
GEMM shapes, DataLoader gaps and DDP overlap before tuning those paths further.

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
