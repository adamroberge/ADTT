# Two-stage ADTT training in this fork

This implementation fills the public release's missing training infrastructure.
It follows arXiv:2503.07338v5, particularly Eqs. 6 and 20, using the released
`TripLane` and `DeltaTokenPredictorV21` networks. It is not an author-provided
recipe, and successful tests do not establish convergence or published scores.

## What is implemented

| Entry point | Purpose |
| --- | --- |
| `tools/prepare_manifest.py` | Read flat or scene-grouped NumPy metadata and write JSONL manifests outside the dataset |
| `train.py --config configs/stage1_static.yaml` | Train the representation encoder and decoder using CE + lambda Lovasz |
| `train.py --config configs/stage2_static.yaml` | Freeze the transferred encoder; train forecasting and fine-tune the decoder with latent L1/L2 + xi occupancy loss |
| `evaluate.py` | Dense, paired evaluation on recorded ego trajectories, including per-horizon/per-class confusion matrices |
| `rollout.py` | Encode only observed frames, forecast explicit future poses, decode in chunks, export predictions and actual GT |

The trainers are single-process CPU/CUDA implementations. There is no DDP launcher
and no automatic selection of another GPU. No training is automatically started.

## Environment

Python 3.11 is recommended. Use an existing compatible **private** environment or
create a separate environment. Never install into another researcher's environment.
The optional legacy camera/MMDetection3D/Transformers dataset stack is not required
by these entry points. The released models still use MMEngine and TorchVision's
real deformable-convolution operator; it is not replaced with a stub.

For a new local CPU development environment:

```bash
python3.11 -m venv .venv-training
.venv-training/bin/python -m pip install -r requirements-training.txt
CUDA_VISIBLE_DEVICES='' .venv-training/bin/python -B -m unittest discover -s tests -v
```

For a new, explicitly private RTX 5090 environment, install the matching official
PyTorch CUDA 12.8 wheels before the remaining requirements:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-training.txt
```

The existing Shannon `adtt` environment has already been qualified for the
representation core. It does not need to be reinstalled. Full-size forecasting
CUDA/deformable-convolution qualification remains a separate hardware check.

## Data contract

Input occupancy is a uint8 `semantics` array with storage order **X,Y,Z** and shape
`(200,200,16)`, not the Z,X,Y presentation order used in the paper. Input label
values must be inside the configured vocabulary. Value 255 is rejected as an
encoder input rather than silently mapped to free space.

`static10` is the LEAF adaptation:

```
0 others       1 barrier      2 traffic_cone   3 driveable   4 other_flat
5 sidewalk     6 terrain      7 manmade        8 vegetation  9 free
```

Set `--label-space occ3d18` for original Occ3D labels (free=17); this is a distinct
training vocabulary. A representation checkpoint must match the forecasting
checkpoint's vocabulary, geometry and representation architecture. An author
18-class checkpoint is not silently adapted to ten classes.

Every JSONL row contains:

```json
{"scene":"scene-0003","token":"sample-token","prev":"previous-token","timestamp":1533201470448696,"semantics_path":"/absolute/path/labels.npz","ego2global_translation":[249.9,917.5,0.0],"ego2global_rotation":[1.0,0.0,0.0,0.0]}
```

Timestamps are integer nuScenes microseconds; quaternions are **w,x,y,z**; positions
are metres in the global frame. Paths may be relative to their original manifest,
but absolute paths are recommended for relocatable run receipts. The preparer
always emits absolute paths. Scene/token uniqueness, increasing timestamps,
internal `prev` links, finite poses, quaternion norms and file coverage are checked.
Actual semantic values are validated on every frame load. Scene-contained windows
with excessive timestamp gaps are excluded. Geometric/time-reversal augmentation
is not enabled: transforming occupancy without poses would corrupt alignment.

The provided LEAF metadata has 700 training scenes and 150 validation scenes.
The paper mentions 100 test scenes; this implementation does not pretend those
are the same evaluation cohort. Use the agreed LEAF scene/start-token manifest.

Existing private `data/metadata-001/{train,val}.jsonl` manifests can be used directly.
To prepare new manifests from existing metadata, without copying occupancy data:

```bash
python tools/prepare_manifest.py \
  --metadata /datasets/nuscenes/world-nuscenes_infos_train.pkl \
  --data-root /user_data/tianranliu/preprocessed_dataset/nuscenes/gts \
  --output "$ADTT_ROOT/data/manifests/train.jsonl"
python tools/prepare_manifest.py \
  --metadata /datasets/nuscenes/world-nuscenes_infos_val.pkl \
  --data-root /user_data/tianranliu/preprocessed_dataset/nuscenes/gts \
  --output "$ADTT_ROOT/data/manifests/val.jsonl"
```

The preparer accepts only basic/NumPy pickle globals; unsupported object types fail
explicitly. It never executes arbitrary custom pickle classes or edits annotations.

## Configuration provenance

Both stages use complete YAML configurations. Stage 2 inherits stage 1 so that
every inherited `TENTATIVE` comment remains visible at its source. Save a modified
configuration under a new name when changing experiments.

| Setting | Provenance |
| --- | --- |
| 4 history / 6 future frames, eight-channel planes, 8x8x4 grid, 64/192 queries | Paper Sec. 4.1/Table 7 |
| nearest warping, zero padding, residual prediction | Paper Sec. 3.2/3.3 |
| token dimension 128 / depth 2 | **Tentative resolution**: Table 5 highlighted row; Sec. 4.1 instead says dimension 96 |
| V21 class selection | Closest released 3D token implementation; exact author model configuration absent |
| decoder width/layers, attention heads/dropout | Tentative released defaults; actual experiment choices unconfirmed |
| AdamW, LR, weight decay, clipping, batch size, steps, warmup/schedule, seed | **Tentative**; the paper does not provide the complete recipe |
| lambda=1 and xi=1 | **Tentative**; only symbolic weights appear in the paper |
| 200k balanced queries, class quotas/boundary threshold | Tentative sampler defaults, not a verified author experiment configuration |
| validation subset/frequency, checkpoint retention, query chunks | Tentative operational choices |

All computation is float32 with TF32 and CuDNN autotuning disabled. CUDA grid-sample
and other kernels can still be nondeterministic: RNG restoration does not promise
bit-for-bit GPU replay. The CPU tests check deterministic resume of both stages.

The released V21 class inherits some legacy modules that are not on its prediction
path. Total instantiated/trainable counts are logged; do not claim they exactly
match the paper's 2.30M parameter count without an additional architecture audit.

## Both training stages

All run directories must be **new**. Resumes use a new work directory and never
overwrite an earlier run. With `ADTT_ROOT` set, outputs must stay under it. On
Shannon the `/user_data/tianranliu/adam` ownership boundary is also enforced.

CPU example (use tiny test fixtures rather than full-resolution sweeps on a Mac):

```bash
python train.py --config configs/stage1_static.yaml \
  --train-manifest "$ADTT_ROOT/data/metadata-001/train.jsonl" \
  --val-manifest "$ADTT_ROOT/data/metadata-001/val.jsonl" \
  --work-dir "$ADTT_ROOT/runs/representation-001" --device cpu --stop-after 2
```

`--stop-after` bounds an invocation without altering the mathematical schedule.
`--max-steps` changes the configured total budget; a short block is not convergence.

After explicit GPU assignment, the equivalent CUDA command requires the assigned
UUID. Set `ASSIGNED_GPU_UUID` only after authorization and a fresh process check:

```bash
python train.py --config configs/stage1_static.yaml \
  --train-manifest "$ADTT_ROOT/data/metadata-001/train.jsonl" \
  --val-manifest "$ADTT_ROOT/data/metadata-001/val.jsonl" \
  --work-dir "$ADTT_ROOT/runs/representation-002" \
  --device cuda --gpu-uuid "$ASSIGNED_GPU_UUID"
```

Stage 2 requires a checkpoint produced by stage 1 (or a complete stage-2 resume):

```bash
python train.py --config configs/stage2_static.yaml \
  --train-manifest "$ADTT_ROOT/data/metadata-001/train.jsonl" \
  --val-manifest "$ADTT_ROOT/data/metadata-001/val.jsonl" \
  --representation-checkpoint "$ADTT_ROOT/runs/representation-002/best.pt" \
  --work-dir "$ADTT_ROOT/runs/forecast-001" \
  --device cuda --gpu-uuid "$ASSIGNED_GPU_UUID"
```

Transfer is strict for encoder, embeddings, normalization, decoder and geometry
convolutions. Vocabulary/geometry/representation settings and manifest hashes must
match. The encoder remains frozen and in evaluation mode. Future occupancy is
encoded only for detached loss targets; it never enters the predictor's history.
The decoder remains trainable. CE/Lovasz are globally pooled over valid sampled
queries; latent L1/L2 are averaged within each plane and equally over the planes.
Decoder activation checkpointing preserves the full sampled Lovasz objective.

Resume example:

```bash
python train.py --config configs/stage2_static.yaml \
  --train-manifest "$ADTT_ROOT/data/metadata-001/train.jsonl" \
  --val-manifest "$ADTT_ROOT/data/metadata-001/val.jsonl" \
  --resume "$ADTT_ROOT/runs/forecast-001/best.pt" \
  --work-dir "$ADTT_ROOT/runs/forecast-resume-001" \
  --device cuda --gpu-uuid "$ASSIGNED_GPU_UUID"
```

Checkpoints contain model/optimizer/scheduler, epoch/batch cursor, all RNG state,
configuration and provenance. Mathematical configuration and dataset changes are
rejected on resume. Constant schedules may extend max_steps; cosine schedules may
not change their original total duration. Model/training source hashes must also
match; use the archived source version for exact resumes. Parent checkpoint hashes
and the original representation-initialization receipt are retained across resumes.
This supports one GPU, not world-size
changes or arbitrary optimizer changes. Sampling happens in the main process;
data workers perform deterministic reads. The final complete checkpoint is reloaded
to verify readability. Only checkpoints created by the current fresh run are pruned.

Each run keeps source/configuration snapshots, a source ZIP, dependency versions,
data manifest hashes, input-token/content-hash traces, per-class/horizon validation
JSONs, training events, numbered recent checkpoints, `best.pt`, and a summary.
`budget_complete` means the configured steps ran, not that quality was established.

## GPU safeguards

The CUDA UUID mask is set before Torch imports and must expose exactly one device.
Preflight checks memory/utilization and **all** reported compute/graphics processes.
Runtime checks run before every batch, validation frame and forecast step. Conflicts
fail the current job; other processes are never killed or signalled. Prior complete
checkpoints remain available. There is no wait-and-auto-launch behavior.

A persistent gdm/Xorg display service also blocks launch by default. Only with
explicit approval may `--allow-display-pid PID` pin that specific service's PID,
owner, executable name, type and process start time, with at most 8 MiB reported
GPU use. It cannot authorize another Python worker or arbitrary display processes.

Snapshots and environment masking do not reserve a GPU against another user's
later launch. Obtain the site's allocation/reservation before beginning a run.

## Dense evaluation and 100-frame export

```bash
python evaluate.py --checkpoint "$ADTT_ROOT/runs/forecast-001/best.pt" \
  --val-manifest "$ADTT_ROOT/data/metadata-001/val.jsonl" \
  --work-dir "$ADTT_ROOT/runs/evaluation-001" \
  --device cuda --gpu-uuid "$ASSIGNED_GPU_UUID"
```

Evaluation visits every voxel, never treats unqueried locations as free, rejects
training-scene overlap, and reports pooled/horizon-specific confusion matrices.
It reports all ten per-class IoUs, primary mIoU over nine non-free classes and an
explicitly named mean including free. Zero-union classes are undefined/excluded;
false-positive-only classes score zero. `visibility: camera/lidar` requires actual
masks; `all` is explicitly an all-voxel protocol. These are static-data metrics,
not directly the paper's 18-class scores. Set `--limit 16` for a diagnostic subset.

```bash
python rollout.py --checkpoint "$ADTT_ROOT/runs/forecast-001/best.pt" \
  --manifest "$ADTT_ROOT/data/metadata-001/val.jsonl" --scene scene-0003 \
  --trajectory straight --frames 100 --fps 2 --speed-from-history \
  --embedding-policy native \
  --work-dir "$ADTT_ROOT/runs/rollout-0003-001" \
  --device cuda --gpu-uuid "$ASSIGNED_GPU_UUID"
```

This is an example protocol, not a confirmed match to Cecilia's speed/start rules.
Freeze the agreed scene/start-token/trajectory manifest first. `--start-token`
selects the first future frame. Default history length is four; the 100 exported
frames exclude history. Straight motion begins immediately, with constant heading,
height and either explicit nonnegative `--speed-mps` or history-estimated speed.
Only historical occupancy is loaded before prediction. Future occupancy is read
after prediction solely for actual GT export. No GT is invented beyond the scene.

The native policy preserves the released future-step lookup and its clamp to the
last embedding. `repeat-training-horizon` changes only the embedding index to
`step % trained_future`, keeping predicted latent states and pose history intact.
Neither policy is established by the paper for 100 future frames (about 50 s at
2 Hz). Select it explicitly; compare policies with identical checkpoints, inputs
and trajectories on held-out recorded windows before claiming long-rollout quality.

Exports contain two dense uint8 files per scene under `occupancy/<scene>/`:
`pred.npy` has exactly the requested number of future frames; `gt.npy` contains
at most the requested real frames (35 by default). Static IDs map to Occ3D IDs
`[0,1,8,11,12,13,14,15,16,17]`, including free 9->17. `--export-label-space native`
keeps the ten-class IDs. Metadata/provenance live outside the scene folder.
Prediction files are re-opened to verify shape/dtype. Failed jobs never report
completion and their work directories must not be reused as complete results.

`--trajectory recorded` permits paired predictions only up to the available pose
horizon. Counterfactual straight predictions are marked **not paired with GT**;
do not report their disagreement with recorded-view GT as paired IoU. BEV/video
rendering and distributional metrics must use the same agreed lab evaluator.

## Tests and remaining qualification

The CPU suite uses real released network operators on tiny synthetic grids. It
checks source-forward equivalence, future-target isolation, strict transfer and
encoder freezing, gradients, query-chunk equivalence, geometry/control timing,
labels/masks, temporal chains, guarded process classification, both stages'
checkpoint/resume, dense evaluation, 100-frame export and unchanged input files.

It does not establish full-resolution CUDA memory/throughput, convergence, the
correct author recipe, quality at 50 seconds, or the final lab comparison protocol.
Those require the assigned GPU, actual data and the agreed experiment budget.
