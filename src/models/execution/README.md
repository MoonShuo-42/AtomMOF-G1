# Whole-model execution optimization

This is an execution change, not a new model architecture. The encoder, 24-layer
DiT, decoder, G1 bridge positions, routing policy, losses, learned parameter
shapes and state_dict keys are preserved. The default configuration retains the
reference execution. No training data, epochs or accumulation are reduced.

## Profiles

- No execution profile override (default `execution: null`): reference execution.
- `execution=math`: shared pair standardization with independent affine and
  projection weights; fused Q/K/V/gate projections; folded G1 readback; shared
  masked geometry contraction layout; vectorized time-bin logging; removal of
  redundant copies when sample multiplicity is one. Existing
  precision is retained.
- `execution=optimized`: the math profile plus BF16 mixed precision for learned
  content/MLPs, native-dtype SDPA, fused AdamW, DDP gradient bucket views and
  persistent DataLoader workers. Sensitive geometry/search/normalization retain
  FP32/FP64. This changes floating-point rounding, not the model definition.

Append a profile to the original full training command. For the optimized profile,
replace an explicit `+trainer.precision=32-true` with
`trainer.precision=bf16-mixed`, or remove that explicit precision override.
CLI overrides have higher priority than the profile. All stage data/batch/epoch
settings and initialization checkpoints remain those of the original command.
Baseline and G1 both use these same profiles. Keep the G1 enabled flag explicit.

`model.flow_model_args.execution.compile_blocks=true` optionally compiles DiT
blocks with dynamic shapes. It is OFF by default until CUDA timings establish a
benefit. Compilation does not wrap model parameters in an `_orig_mod` prefix.
`model.flow_model_args.execution.compile_pair_features=true` optionally compiles
the original dense and G1 compact pair feature construction, fusing its tensor
operations without dropping any pair. The benchmark `--compile` enables both
regional options. Geometry search stays eager. Compilation cost and recompilation
must be counted.
The old whole-Flow `compile_model` option is not needed for this profile.

## Algebra and lifetime

For pair bias, the same z enters every layer in a DiT stack. Only its parameter-free
standardization is shared (grouped by epsilon and normalized shape). Each layer
keeps its own gamma, beta and projection W. The effective W*gamma and W*beta are
computed differentiably in each forward; they are not new parameters or detached
cached weights. Autograd saves one standardized pair tensor instead of a separate
128-channel affine output per layer. No learned pair bias or gradient is dropped.

G1 readback pools the SiLU hidden values per attention head BEFORE the final linear
projection. The output bias is multiplied by the actual attention mass, including
empty/padded rows. This is algebraically equivalent for arbitrary weights, not only
weights summing to one. Parameters stay under the same value module names.

G1 E is masked and arranged once as (B,N*16,N) in the current forward. Four bridges
reuse this layout, but recompute their own routing, U and Z. No context is reused
across diffusion steps or optimizer updates. Coordinates and cell gradients remain
connected. Geometry cutoffs, search precision, 125-image cap and whole-sample
fallback rules are unchanged.

The reference implementation remains selectable for execution regression tests;
this does not restore obsolete G1 architectures.

## Attention backend

`sdpa_backend=auto` lets PyTorch select the implementation. Other choices are
`math`, `efficient`, `flash`, `cudnn`. A forced unsupported backend raises an error;
there is no silent dropping of pair bias, mask or bias gradients. Native SDPA is
already used by the old code, so merely calling SDPA is not an optimization claim.
Profile the actual CUDA kernels with PyTorch 2.6 on the server before choosing a
backend. The optional xformers attention implementation is left available.

## Benchmark on the server

Activate the existing server environment, synchronize the repository, then run:

```bash
cd /data/ai4science/AtomMOF-G1
python scripts/benchmark_execution.py \
  --lmdb-dir /data/ai4science/AtomMOF/AtomMOF-main/data/bwdb \
  --model g1 --stage 1 --batches 200 --warmup 20 --repeats 3
```

Repeat for stages 2 and 3 and `--model baseline`. The defaults use all 8 visible
L40S GPUs and the actual stage-specific atom/batch/accumulation budgets. No data
sample limit is applied. `--batches 0` measures full epochs; `--checkpoint PATH`
loads the same architecture's weights for every compared profile. Benchmark output
is always in a new directory and never modifies training checkpoints.

Each comparison runs reference/math/optimized from the same seed and rotates their
order across repeats. Worker persistence is disabled equally in this diagnostic
launcher to keep worker RNG lifecycle comparable, including for multiple epochs.
The optimized training profile enables it; measure its end-to-end benefit separately.
The callback records actual per-rank batch indices and shapes, step counts,
forward/backward/optimizer/DDP wall time and CUDA peak allocated/reserved memory.
Reports refuse speed comparisons when the workload manifests differ. CPU memory
measurements never satisfy the GPU target.

`comparison.json` reports training-window speedup and memory ratio. The joint target
is speedup >= 2.0 AND peak allocated ratio <= 0.5 on identical workloads. Peak
includes warmup (optimizer initialization) as well as the measured window. Process
wall time includes startup/compilation. Diagnostic runs disable validation,
checkpointing and W&B equally and therefore DO NOT prove full end-to-end training
speedup. Complete training/validation/save runs must be measured separately before
claiming a halved total training time. Large structures in stage 3 must be covered;
200 batches are not evidence about the largest batch in the dataset.

## Correctness and reporting

Run `python -m pytest tests -q`. New checks cover independent affine pair gradients,
full Flow/G1 outputs and parameter gradients, an AdamW update, checkpoint keys,
masked/non-unit readback, geometry NaN masking, BF16 finite backward, dynamic
execution profiles and compile checkpoint names. Reference and math outputs are
compared within explicit floating-point tolerances, not claimed bit-identical.

Do not claim GPU speed/memory goals from CPU tests or saved-tensor counts. For
scientific comparisons, train baseline and G1 with the same selected precision and
runtime profile; keep the same test protocol. A short finite-loss test does not
prove MR/RMSD equivalence or convergence. Existing FP32 results remain historical
references. Current CUDA performance and full training accuracy are pending server
measurement.

See [VALIDATION.md](VALIDATION.md) for the recorded local checks and pending GPU work.
