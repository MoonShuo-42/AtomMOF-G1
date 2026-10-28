# G1-New v0

Independent dynamic local-environment extension of AtomMOF-small. Select
`model=g1new` or `experiment=g1new_small_stage1`. Baseline (`model=simple`)
and the existing G1 configs remain independent. Enabling G1 and G1-New together
is rejected. No template-based G1 bridge is constructed for G1-New.

## Architecture and files

* `bridge.py`: one insertion after DiT layer 12; multiplicity expansion and
  per-forward geometry. Conditions alone may be cached during sampling.
* `conditioning/`: an **independent copy of the existing G1 ConditionEncoder**,
  with its own code and weights. Original c/p 256/128 are reduced to 128/64,
  then one block-local Transformer (4 heads). Existing normalization, GELU,
  attention feature gates and compact same-block pairs are preserved.
* `geometry/`: heavy-atom anchors and a sparse `(batch, atom, image, environment)`
  edge table. Every valid non-H atom is an environment. Hydrogen participates
  in messages/readback. Different images are never merged before the nonlinear
  message. Parameter-free cell conversion and time encoding have independent
  copies of the established formulas; no legacy G1 imports, geometry moments,
  routing, contexts or template bank. The condition encoder retains common
  AtomMOF attention/execution operators, just as the existing G1 encoder does.
* `relations.py`: 16 RBF + direction3 + same-block1 + bond-embedding8 → 32.
* `environment.py`: initial state from anchor feature; independent
  `P=omega*sigmoid(score)`; `u=sum(P*value)/(1+sum(P))`; residual MLP update
  with `mass/(1+mass)`. Exactly the same weights are used for both rounds.
* `readback.py`: independent receiver-dependent score/content heads, globally
  normalized over all environment/image edges and one zero-valued null item.
* `module.py`: F64 from H256/condition128/time32; two local rounds followed by
  a bias-free, zero-initialized 64→256 residual. Original decoder/loss unchanged.
* `checkpoint.py`: explicit architecture-transfer loader.

The local module has 556,340 parameters at the default small dimensions,
including the 374,016-parameter condition encoder. Total small-model parameters:
38,932,660 (baseline 38,376,320). Environment count M affects activations/work,
not parameter count. The width 64 is **not** K=64.

## Numerical and resource decisions

Coordinates and cell lengths are nm; angles are radians, with row lattice
vectors in the original canonical frame. Use only current noisy coordinates /
cell, not target structure fields. The radius is 0.8 nm with cosine support
between 0.6 and 0.8 nm. Known input bond types 1..5 force their exact nearest
periodic image into the candidate set. Class 0 remains unknown/no forced edge.
This does not infer new inter-block coordination bonds.

Nearest chemical images use **hard selection**, as in v0. Exact ties select the
lexicographically first integer translation. This is reproducible but not
continuous at image switches; the regression test demonstrates this limitation.
No multi-image chemical smoothing or other architecture change is introduced.

Cell checks occur BEFORE unsafe divisions/square roots/inverses. Invalid
lengths/angles, nonfinite valid-atom coordinates, singular cells or condition
number >1e6 disable only the new residual for that sample. All parameters remain
connected even when an entire rank has no edges/anchors. Float64 discrete search
and safety checks preserve the existing geometry policy; learned computations
and reductions use FP32, with FP64 available for numerical tests. This does not
change the baseline FP32/TF32 trainer policy.

Exact enumeration is streamed, without a 27-image assumption or neighbour cap.
The configurable per-rank/per-forward guards default to 32,000,000 enumerated
search candidates (including nearest-image search) and 2,000,000 retained edges.
Exceeding either raises `SearchBudgetExceeded` **before** the corresponding
large allocation. It never truncates candidates or silently disables a valid
sample. These guards limit work/resources; they do not alter accepted results.
Inspect noisy cells and batch sizes before raising them. DDP will abort the run
on a rank exception rather than silently train a different geometry.

## Implementation optimizations

* One condition encoder and one set of environment-update weights; no weights
  allocated per environment or iteration.
* Shared per-atom/per-environment contributions to the first edge MLP projection
  are computed before gathering, preserving the full dense linear operator.
  This avoids materializing the 224-channel edge input and repeating its atom
  projections for every edge/image. No rank reduction or removed bias.
* Independent score/value heads are separate rows of one linear operation.
* Candidate indices, support, relation encoding and F are reused within one
  forward, then discarded. Learned states/geometry are never cached across
  Flow sampling steps. Static conditions can use the existing inference cache.
* Sparse packed anchors/edges, bounded pair/image/edge chunks. Edge MLPs use
  non-reentrant activation checkpointing by default to avoid retaining every
  chunk's hidden activations. No stochastic operations occur in these regions.
* Readback uses online segmented log-sum-exp with rescaling of previous chunks;
  the null item is counted once per atom. There is no per-chunk softmax.

Split GEMMs and reduction order can produce ordinary floating-point differences;
they are formula/gradient equivalent, not a promise of bitwise CUDA identity.
The full backbone still contains dense pair tensors. Lower branch parameter
count does not establish lower end-to-end runtime/memory; benchmark on the GPU.

Zero output initialization reproduces the retained backbone on valid atoms and
lattice predictions. The new module explicitly zeros padded hidden states.
When upstream loss gradients are nonzero, the output projection learns first;
internal message weights get gradients once the output projection is nonzero.

## Loading weights

Same-architecture G1-New checkpoints: use ordinary `pretrained_ckpt_path` for
weights-only stage transfer, or `resume_ckpt_path` for full training resume.
Prediction reconstructs G1-New from checkpoint hyperparameters. Independent
evaluation ablation: `predict_args.g1new_enabled=false`; null follows checkpoint.

Baseline or old G1 → G1-New is explicitly opted into using:

```bash
model=g1new \
pretrained_ckpt_path=/absolute/path/to/source.ckpt \
pretrained_g1new_transfer=true \
pretrained_g1_expand_groups=false \
resume_ckpt_path=null
```

Every backbone key/shape/dtype must match. An old G1 encoder is copied only when
its full state matches and its shared aliases agree. Old router, templates,
aggregation/group/readback weights are deliberately omitted. New local weights
retain initialization. The loader reports what transferred/discarded; optimizer,
scheduler and epochs start fresh. Removing old G1 bridges changes that model's
function even though its backbone weights are kept. K-expansion is unrelated.

## Validation and server trial

`tests/test_g1new.py` covers brute-force periodic enumeration, chemical-image
switches, explicit budget failure, direct-v0 vs optimized output/input/parameter
gradients, chunk sizes, empty paths, initial identity, cache/multiplicity,
independent imports/configs, condition-encoder parity, checkpoint transfers,
real training/resume/sampling, and two-rank DDP with an empty rank.
Existing G1/baseline integration tests must also pass.

Local validation (2026-10-26): 28 G1-New tests and 160 existing baseline/G1/
execution regression cases passed on CPU. The distributed check uses two Gloo
processes, including one rank with no environments. This is a correctness result,
not an 8-GPU performance or peak-memory measurement.

For a first server training trial (run in the existing AtomMOF environment):

```bash
cd /data/ai4science/AtomMOF-G1 || exit 1
export PROJECT_ROOT="$PWD"
RUN_DIR="$PROJECT_ROOT/logs/bench_bwdb_g1new_small_stage1_2epoch_8gpu_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR" || exit 1
git rev-parse HEAD > "$RUN_DIR/git_commit.txt"
git diff > "$RUN_DIR/tracked_changes.patch"
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u src/train.py \
  experiment=g1new_small_stage1 execution=optimized \
  data.lmdb_dir=/data/ai4science/AtomMOF/AtomMOF-main/data/bwdb \
  trainer.max_epochs=2 logger=null extras.enforce_tags=false \
  task_name=bench_bwdb_g1new_small_stage1_2epoch \
  hydra.run.dir="$RUN_DIR" \
  +callbacks.execution_benchmark._target_=src.callbacks.execution_benchmark.ExecutionBenchmark \
  +callbacks.execution_benchmark.warmup_batches=20 \
  2>&1 | tee "$RUN_DIR/train.log"
```

This stage-1 trial uses batch cap 64, accumulation 1, max atoms 200, square budget
3e6 and FP32, matching the optimized baseline training settings. GPU memory fit
and performance have to be measured. Stage-2/3 configs use 32/2, 100 epochs,
max atoms 300/unlimited and square budgets 3e6/5e5; set the intended prior
checkpoint explicitly. No training data or server jobs are launched by tests.
