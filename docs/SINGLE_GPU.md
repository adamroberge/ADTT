# Run both ADTT stages on one GPU

This runbook uses this fork's static10 configuration and an explicitly assigned,
otherwise idle GPU. The server needs Linux, Python 3.11, an NVIDIA driver compatible
with CUDA 12.8, and existing static occupancy/pose metadata. It installs no system
packages and does not edit any existing conda environment or dataset.

## 1. Get the committed training branch

These commits are local until you push or transfer them. A GitHub clone cannot see
unpublished local commits. To transfer without pushing, create a bundle on the Mac:

```bash
git -C '/Users/roberge/UofT/Leaf Lab/OccSim Project/ADTT' bundle create \
  '/Users/roberge/UofT/Leaf Lab/OccSim Project/ADTT-training.bundle' training
```

Copy that bundle to your server using your normal transfer method. On the server,
choose a **new private directory** (on Shannon, inside `/user_data/tianranliu/adam`):

```bash
git clone --branch training /path/to/ADTT-training.bundle /path/to/private/ADTT
cd /path/to/private/ADTT
export ADTT_ROOT="$(pwd -P)"
git status --short --branch
git log -2 --oneline
```

After a later authorized push, cloning the fork's `training` branch is equivalent.
Do not copy another researcher's environment or run `finish_on_shannon.sh`.

## 2. Create a private environment

Run from the new clone. `python3.11` may be an absolute path to an existing Python
3.11 interpreter; `venv` creates an isolated environment without changing it.
On the previously inspected RTX 3090 machine, that interpreter was
`/home/adam/miniforge3/envs/occsim-cuda3090/bin/python`.

```bash
set -euo pipefail
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
mkdir -p "$ADTT_ROOT/.cache/pip" "$ADTT_ROOT/.tmp"
export PIP_CACHE_DIR="$ADTT_ROOT/.cache/pip" TMPDIR="$ADTT_ROOT/.tmp"
test ! -e "$ADTT_ROOT/.venv-training"
python3.11 -m venv "$ADTT_ROOT/.venv-training"
export ADTT_PYTHON="$ADTT_ROOT/.venv-training/bin/python"
"$ADTT_PYTHON" -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
"$ADTT_PYTHON" -m pip install -r requirements-training.txt
"$ADTT_PYTHON" -m pip check
"$ADTT_PYTHON" -B -m unittest discover -s tests -v
```

The CPU suite keeps GPUs hidden. The wheel pairing comes from
[PyTorch's official instructions](https://pytorch.org/get-started/previous-versions/).
CUDA kernels, including TorchVision's deformable convolution, are exercised by
the guarded real-data qualification in step 4. No CUDA toolkit compilation is required.

For a later terminal session, restore `ADTT_ROOT`, `ADTT_PYTHON`, and
`CUDA_VISIBLE_DEVICES=''`; no conda activation is necessary.

## 3. Point at existing compact static10 data

Set these three paths to existing files/directories. `STATIC_GTS` must contain
`scene-*/<token>/labels.npz`, with compact labels **0..9 and free=9**. Cecilia's
remapped Occ3D-ID copy (free=17) is not interchangeable with this input vocabulary.
Training/validation metadata must contain temporal links, timestamps and ego poses.

```bash
export STATIC_GTS='/absolute/path/to/preprocessed_dataset/nuscenes/gts'
export TRAIN_METADATA='/absolute/path/to/world-nuscenes_infos_train.pkl'
export VAL_METADATA='/absolute/path/to/world-nuscenes_infos_val.pkl'
"$ADTT_PYTHON" -B tools/prepare_manifest.py \
  --metadata "$TRAIN_METADATA" --data-root "$STATIC_GTS" \
  --output "$ADTT_ROOT/data/manifests/train.jsonl"
"$ADTT_PYTHON" -B tools/prepare_manifest.py \
  --metadata "$VAL_METADATA" --data-root "$STATIC_GTS" \
  --output "$ADTT_ROOT/data/manifests/val.jsonl"
```

Use existing valid manifests instead of rerunning conversion if they already exist.
No occupancy is copied or rewritten. Paths must be valid on this server; regenerate
manifests from the existing metadata when moving from a different machine.
Scene and token overlap is rejected; every loaded occupancy frame is checked for
shape and label range. Both stages must use the same train/validation manifests.

## 4. Qualify the full configured workload before a long run

Obtain the GPU assignment first. Inspect current processes, then set the assigned
UUID explicitly; the code never chooses a GPU based on free memory.

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free --format=csv
nvidia-smi
export ASSIGNED_GPU_UUID='GPU-REPLACE_WITH_EXPLICITLY_ASSIGNED_UUID'
"$ADTT_PYTHON" -B tools/qualify_training.py \
  --train-manifest "$ADTT_ROOT/data/manifests/train.jsonl" \
  --val-manifest "$ADTT_ROOT/data/manifests/val.jsonl" \
  --work-dir "$ADTT_ROOT/runs/qualification-001" \
  --device cuda --gpu-uuid "$ASSIGNED_GPU_UUID"
```

This sequential command runs six checks on real data:

1. Stage 1: two optimizer updates with the supplied full-resolution config.
2. Resume Stage 1 for one additional update and verify checkpoint read-back/digest.
3. Stage 2: two updates, initialized from that representation checkpoint.
4. Resume Stage 2 for one additional update.
5. Dense evaluation of one validation window.
6. A 100-frame straight rollout with the final trained embedding reused.

The full 200k query count, 200x200x16 geometry, four/six frame context and 96-dimensional
tokens are retained. Only the invocation length and diagnostic validation subset
are reduced; `max_steps` and the cosine schedule are unchanged. The rollout uses
an explicit **1 m/s diagnostic speed**, not a finalized comparison protocol.
The coordinator writes per-command logs and `qualification.json` with PASS/FAIL,
checkpoint hashes, execution times and peak CUDA memory for each training block.
If any check fails, it stops. It does not silently lower resolution or switch GPUs.

The default per-process allocator cap is 80% of GPU VRAM. If a run fails for memory,
inspect the log and explicitly adjust `training.decode_chunk` in a new config (both
stages), then qualify that config using `--stage1-config`/`--stage2-config`. Do not
assume a different query count, batch size or architecture has passed qualification.
Batch size defaults to one, with no DDP or multi-GPU process launcher.

The smoke run establishes execution on sampled inputs, not convergence or the
validity of every possible configuration. Its checkpoints are diagnostic artifacts.
Start the full training runs below in fresh directories.

## 5. Train representation, then forecasting

```bash
"$ADTT_PYTHON" -B train.py --config configs/stage1_static.yaml \
  --train-manifest "$ADTT_ROOT/data/manifests/train.jsonl" \
  --val-manifest "$ADTT_ROOT/data/manifests/val.jsonl" \
  --work-dir "$ADTT_ROOT/runs/representation-001" \
  --device cuda --gpu-uuid "$ASSIGNED_GPU_UUID"
```

After that run completes successfully, use its selected representation checkpoint:

```bash
"$ADTT_PYTHON" -B train.py --config configs/stage2_static.yaml \
  --train-manifest "$ADTT_ROOT/data/manifests/train.jsonl" \
  --val-manifest "$ADTT_ROOT/data/manifests/val.jsonl" \
  --representation-checkpoint "$ADTT_ROOT/runs/representation-001/best.pt" \
  --work-dir "$ADTT_ROOT/runs/forecast-001" \
  --device cuda --gpu-uuid "$ASSIGNED_GPU_UUID"
```

Both defaults use 30,000 optimizer updates, batch one, AdamW peak LR 1e-3,
500 warmup updates, cosine floor 1e-6, and tentative lambda=xi=1. These are an
explicit reconstructed recipe, not evidence that 30,000 steps suffice for convergence.
Select the total budget **before** starting; an exact cosine resume cannot change it.
Validation during training uses 16 items for checkpoint selection. Use
`--validation-frames 0` for full validation, at a substantially higher runtime cost.

To resume after an interruption, inspect the run's `latest.json`, use its exact
`step-XXXXXXXX.pt` file, and choose a new work directory:

```bash
"$ADTT_PYTHON" -B train.py --config configs/stage2_static.yaml \
  --train-manifest "$ADTT_ROOT/data/manifests/train.jsonl" \
  --val-manifest "$ADTT_ROOT/data/manifests/val.jsonl" \
  --resume "$ADTT_ROOT/runs/forecast-001/step-XXXXXXXX.pt" \
  --work-dir "$ADTT_ROOT/runs/forecast-resume-001" \
  --device cuda --gpu-uuid "$ASSIGNED_GPU_UUID"
```

The source, dataset and mathematical config must match the checkpoint. Monitor
`events.jsonl` for actual/next LR, loss terms, gradient norm, validation scores and
checkpoint paths. `summary.json` records completion status and peak memory.

## 6. Evaluate and export

```bash
"$ADTT_PYTHON" -B evaluate.py \
  --checkpoint "$ADTT_ROOT/runs/forecast-001/best.pt" \
  --val-manifest "$ADTT_ROOT/data/manifests/val.jsonl" \
  --work-dir "$ADTT_ROOT/runs/evaluation-001" \
  --device cuda --gpu-uuid "$ASSIGNED_GPU_UUID"
"$ADTT_PYTHON" -B rollout.py \
  --checkpoint "$ADTT_ROOT/runs/forecast-001/best.pt" \
  --manifest "$ADTT_ROOT/data/manifests/val.jsonl" --scene scene-0003 \
  --trajectory straight --frames 100 --fps 2 --speed-from-history \
  --embedding-policy clamp-trained \
  --work-dir "$ADTT_ROOT/runs/rollout-0003-001" \
  --device cuda --gpu-uuid "$ASSIGNED_GPU_UUID"
```

Evaluation reports pooled/per-horizon IoU, all ten per-class scores, and non-free
mIoU with a fixed denominator of nine (absent classes contribute zero).
Rollout exports 100 predictions and available real GT separately. Export defaults
to Occ3D IDs for downstream lab compatibility; it does not invent dynamic labels.
Agree the scene list, first future token (`--start-token`) and straight speed policy
with the comparison pipeline. A counterfactual straight path is not paired with
recorded-view GT; use recorded trajectories for paired IoU.
