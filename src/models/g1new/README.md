# G1-New: v0 local update with configurable insertion sites

Independent dynamic local-environment extension of AtomMOF-small. Select
`model=g1new` or `experiment=g1new_small_stage1`. Baseline (`model=simple`)
and the existing G1 configs remain independent. Enabling G1 and G1-New together
is rejected. No template-based G1 bridge is constructed for G1-New.

## Architecture and files

* `bridge.py`: a module per selected DiT layer, default `[6,12,18,24]`.
  All sites share one condition encoder and one current-geometry computation.
  Multiplicity expansion is done once. Fixed Fourier time features are computed
  once per network forward. Conditions alone may be cached during sampling.
* `conditioning/`: an **independent copy of the existing G1 ConditionEncoder**,
  with its own code and weights, shared across G1-New sites. Original c/p
  256/128 are reduced to 128/64,
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

A standalone local module has 556,340 parameters at the small dimensions,
including a 374,016-parameter condition encoder. The encoder is shared by object
identity, so it is counted only once by parameters()/the optimizer. Other learned
weights (including relation encoders) are independent across insertion sites.

For L sites, additional parameters = 374,016 + L * 182,324:

| Model | Added parameters | Total parameters |
| --- | ---: | ---: |
| optimized baseline small | 0 | 38,376,320 |
| G1-New, 1 site | 556,340 | 38,932,660 |
| G1-New, default 4 sites | 1,103,312 | 39,479,632 |

Compared with four independent complete modules (2,225,360 extra parameters),
sharing the encoder saves 1,122,048 parameters (50.42% of the added branch).
Environment count M affects activations/work, not parameter count. The width 64
is **not** K=64. Two internal rounds reuse each site's own update weights; rounds
are not the number of insertions.

Configure number and position together (one-based, sorted, unique, within the
trunk depth). These overrides apply when constructing a training model:

```bash
# Default: four sites, after every six layers
model=g1new model.flow_model_args.g1new.insert_after='[6,12,18,24]'
# Two sites
model=g1new model.flow_model_args.g1new.insert_after='[8,16]'
# One site (the scalar 12 is also accepted)
model=g1new model.flow_model_args.g1new.insert_after='[12]'
```

Disable the extension at construction with `model.flow_model_args.g1new.enabled=false`.
An empty list is rejected. Runtime prediction ablation bypasses modules without
removing their stored weights. The optimized AtomMOF backbone, execution policy,
loss, optimizer and precision defaults are not modified by this multi-site change.

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

* One shared condition encoder is constructed directly, without initializing
  then discarding encoders for later sites. Shared state is stored by reference;
  checkpoint aliases are validated for agreement. Each site has its own learned
  update weights, reused over its internal rounds and all environments.
* The sparse periodic image search, selected differentiable distances/directions,
  support weights and multiplicity expansion run once for all insertion sites.
  All site gradients accumulate through the live shared geometry/condition graph.
* Fixed Fourier time features are computed once per forward. Caches live in an
  explicit forward context, never in a persistent model attribute.
* Shared per-atom/per-environment contributions to the first edge MLP projection
  are computed before gathering, preserving the full dense linear operator.
  This avoids materializing the 224-channel edge input and repeating its atom
  projections for every edge/image. No rank reduction or removed bias.
* Independent score/value heads are separate rows of one linear operation.
* Each site's relation encoding and F are reused across its internal rounds.
  The learned relation encoders differ across sites, so their outputs, H-dependent
  features and environment states are not cached across sites. Dynamic geometry
  is rebuilt for every Flow sampling step. Static conditions can use the existing
  inference cache.
* Unlearned RBF/direction/same-block edge features are shared across sites. Bond
  embeddings and relation MLPs still have independent parameters at every site.
  Geometry and bond embeddings pass through the original joint 28-channel
  linear. The first optimization's six-row bond preprojection is withdrawn
  while investigating CUDA gradient differences; its split GEMMs introduced
  different rounding points under the existing FP32/high/TF32 policy.
* Geometry compacts valid indices once, reuses integer layouts and validates bond
  classes once on the same valid support. Larger pair/image chunks reduce CUDA
  launches and host reads. Float64 search decisions, exact chemical nearest-image
  ties, cutoff, invalid-cell behavior and fail-fast budgets are unchanged.
* Compute atom/anchor node projections and the invariant relation projection
  once per site when checkpointing is disabled. Reuse them across rounds, while
  preserving the reference addition order: atom + (anchor + state) + relation.
  Anchor features are also reused by initialization/projection, and the cached
  projections are released before readback. Chunked edge maps use `split` rather than
  independent slices, avoiding repeated full-edge-size slice-gradient allocation.
  A single/empty chunk returns directly without a redundant concatenation/copy.
* The last message linear and following score/value linear run separately in
  their original order. Their algebraic weight composition is withdrawn pending
  CUDA diagnosis. Parameter tensors, biases and the SiLU are unchanged; no
  parameter replacement, weight tying, detach or optimizer-state migration.
* Sparse packed anchors/edges, bounded pair/image/edge chunks. Optional
  non-reentrant checkpointing covers edge projection, nonlinear heads, weighted
  reduction and readback normalization, returning small per-environment/atom
  tensors. Saving only the edge-MLP checkpoint inputs would still retain huge
  edge-sized hidden tensors; this regression in the first optimization has been
  fixed. Shared relations and small node projections remain inputs. No stochastic
  operations occur in these regions. Checkpointing recomputes work in backward;
  the speed benchmark below explicitly keeps it disabled.
* Environment/readback reduce all edge contributions once. Readback uses one
  global segmented log-sum-exp; the null item is counted once per atom. There is
  no per-chunk softmax. The previous streaming implementation remains available
  for reference comparisons.

The new `model.flow_model_args.g1new.module.optimized` boolean defaults to `true`
and changes only the local implementation. It is independent of the backbone's
`execution=optimized` profile. It does **not** change modules, rounds, dimensions,
precision, trainable tensors, parameter names, initialization or checkpoint
shapes. Same-architecture pre-optimization checkpoints still load strictly.
The default edge/pair/image chunks are now 65536/65536/262144. To reproduce the
previous G1-New execution path, set:

```bash
model.flow_model_args.g1new.module.optimized=false \
model.flow_model_args.g1new.module.edge_chunk_size=8192 \
model.flow_model_args.g1new.module.geometry.pair_chunk_size=4096 \
model.flow_model_args.g1new.module.geometry.image_chunk_size=65536
```

Chunks bound individual map/search workspace, not the total saved training
activations. Peak GPU memory still requires measurement, especially with
activation checkpointing disabled. Larger chunks do not change the edge set or
the existing total candidate/edge limits.

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
The position list must match when resuming. Original single-site checkpoints
with saved scalar `insert_after` are reconstructed as one site; their original
`g1new_bridge.module.*` keys map to `g1new_bridge.bridges.<layer>.*` during strict
loading. This is a naming migration, not automatic insertion expansion. Loading
an old single-site checkpoint into the new four-site default is rejected. Do not
use `strict=False` to silently initialize missing sites in a trained checkpoint.
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
retain initialization at every site. The loader copies the compatible condition
encoder to every registered alias of the single shared instance. It reports
what transferred/discarded; optimizer,
scheduler and epochs start fresh. Removing old G1 bridges changes that model's
function even though its backbone weights are kept. K-expansion is unrelated.

## Validation and server trial

`tests/test_g1new.py` and `tests/test_g1new_multisite.py` cover periodic enumeration, chemical-image
switches, explicit budget failure, direct-v0 vs optimized output/input/parameter
gradients, chunk sizes, empty paths, initial identity, cache/multiplicity,
independent imports/configs, condition-encoder parity, checkpoint transfers,
real training/resume/sampling, and multi-site two-rank DDP with an empty rank.
Multi-site tests compare cached shared work to independent recomputation at each
site, including input/parameter gradients; count the encoder/geometry/time calls;
exercise optimized backbone insertion order and exact baseline ablation; and
validate shared checkpoint aliases and original single-site checkpoint names.
Existing G1/baseline integration tests must also pass.

Local correctness validation uses CPU, including two Gloo processes with one
rank having no environments. This is not an 8-GPU performance or peak-memory
measurement. Sharing eliminates repeated work, but end-to-end GPU improvements
and memory fit still need measurement with the intended dynamic batches.

`tests/test_g1new_execution.py` additionally compares reference and optimized
outputs and all input/parameter gradients in FP32/FP64, rounds 1/2/3 and with/
without checkpointing. It checks shared four-site geometry-feature gradients,
edge identity across chunk sizes, invalid-sample bond validation and budget
failures. The implementation changes are confined to G1-New: no optimized
backbone, legacy G1, shared execution, loss, optimizer or entry-point edits.
The review additionally checks all six bond classes, the backing storage of
saved checkpoint tensors (not just the shapes of small views), and three AdamW
updates including optimizer moments. No activation caches survive into the next
forward/optimizer step. The server numerical gate exercises all bond classes.

The CUDA report for commit `11d7480` (two rounds) had identical edge sets and
forward maximum absolute error 3.67e-5, but 9 of 158 tensor comparisons failed
the elementwise gradient tolerance (rtol=0.003, atol=0.0002). No training epochs
ran after that failure. Small whole-tensor relative L2 errors do not override
an elementwise failure. This evidence does not isolate which rearrangement
caused the differences or establish that they are harmless to training.

The current implementation restores affine boundaries, the joint relation
linear, and the environment addition order. Search/index reuse, shared raw
geometry features, split-based chunking and global pooling remain. CUDA must
be checked again; CPU parity does not establish CUDA parity or GPU speed.

Run the single-GPU diagnosis first (no training epochs):

```bash
cd /data/ai4science/AtomMOF-G1 || exit 1
export PROJECT_ROOT="$PWD"
CUDA_VISIBLE_DEVICES=0 python -u scripts/benchmark_g1new_execution.py \
  --diagnose-only --rounds 2
```

It compares a reference repeat, the current optimized path, three individual
reassociations (affine heads, bond preprojection, environment addition), and
their combination reproducing the withdrawn implementation. These ablations
apply only to disposable diagnostic model copies. Precision policy and the
original tolerance are identical for every comparison. Reports include failed
element counts, worst-element values/allowed errors, edge order, and GPU/Torch
metadata. The reference repeat and current optimized path must pass; failures
of intentionally restored legacy ablations are recorded without preventing
the other comparisons. The script always packages completed reports, including
on numerical failure. Do not bypass the gate to collect training timings.

For the server performance target, use the B32/accumulation-2 control, four sites
and two internal rounds. All activation checkpointing is explicitly disabled.
The script first checks four-site outputs/input/parameter gradients under the
actual CUDA FP32/high/TF32 policy, including nonzero output projections; a
failure stops training and preserves the diagnostic JSON. This is a numerical
equivalence gate with stated tolerances, not a guarantee of identical training
trajectories. CPU double/float regression tests provide tighter checks.

```bash
cd /data/ai4science/AtomMOF-G1 || exit 1
export PROJECT_ROOT="$PWD"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python -u scripts/benchmark_g1new_execution.py --rounds 2 --epochs 2
# Optional matched old/new execution comparison, rounds 1 then 2:
# python -u scripts/benchmark_g1new_execution.py --rounds 1 2 --epochs 2 --include-reference
```

The trial overrides the stage-1 experiment's 64/1 default to **32/2**, keeps
max atoms 200, square budget 3e6, full training data, val=1000 and seed=42.
It records each full training epoch (including warmup batches, excluding
validation/checkpoint writing), maximum allocated/reserved memory across all
eight ranks, precision, parameter count and batch-order hashes. `summary.log`,
`summary.json` and `reports.tar.gz` are written automatically under the printed
run directory. Incomplete rank/epoch reports cannot pass as complete epochs.
`--offline` changes only W&B connectivity. `--dry-run` prints the commands.

The <120 seconds/epoch target is **not yet verified on L40S**. Local CPU tests
cannot establish end-to-end CUDA speed or peak memory. Stage-2/3 experiment
defaults remain 32/2, 100 epochs, max atoms 300/unlimited, square budgets 3e6/5e5.
