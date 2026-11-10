"""Standalone paired evaluation and trajectory-conditioned occupancy export."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .checkpoint import atomic_json, load_checkpoint, source_receipt
from .config import STATIC_TO_OCC3D
from .data import OccupancyDataset, load_frame
from .engine import dense_prediction, evaluate, initialize_runtime
from .manifest import read_manifest, sha256
from .models import build_model, forecast


def runtime_arguments(parser):
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--allow-display-pid", type=int)
    parser.add_argument("--memory-fraction", type=float, default=0.8)


def load_trained(args, guard):
    device = initialize_runtime(args, guard)
    saved = load_checkpoint(args.checkpoint)
    config = saved["config"]
    guard.step()
    model = build_model(config).to(device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    return model, config, saved, device


def evaluation_parser():
    parser = argparse.ArgumentParser(description="Dense paired evaluation on recorded ego trajectories")
    runtime_arguments(parser)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates all items; otherwise a diagnostic subset")
    return parser


def evaluation_main(args, run, guard):
    model, config, saved, device = load_trained(args, guard)
    if args.limit < 0:
        raise ValueError("limit must be nonnegative")
    dataset = OccupancyDataset(args.val_manifest, config)
    train_scenes = set(saved["provenance"]["training_scenes"])
    if train_scenes & {r["scene"] for r in dataset.rows}:
        raise ValueError("Evaluation manifest overlaps training scenes")
    result = evaluate(model, dataset, config, device, guard, args.limit)
    result.update(checkpoint_sha256=sha256(args.checkpoint), checkpoint_step=saved["step"],
                  manifest_sha256=sha256(args.val_manifest))
    atomic_json(run / "metrics.json", result)
    atomic_json(run / "provenance.json", source_receipt(run))
    print(json.dumps(result["aggregate"], indent=2))


def rollout_parser():
    parser = argparse.ArgumentParser(description="History-only ADTT rollout; outputs are predictions, not synthetic GT")
    runtime_arguments(parser)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--start-token", help="First future GT token to align the export start")
    parser.add_argument("--start-index", type=int, help="First future frame index; default is history length")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--trajectory", choices=("straight", "recorded"), required=True)
    parser.add_argument("--fps", type=float, default=2.0)
    speed = parser.add_mutually_exclusive_group()
    speed.add_argument("--speed-mps", type=float)
    speed.add_argument("--speed-from-history", action="store_true")
    parser.add_argument("--embedding-policy", choices=("native", "repeat-training-horizon"),
                        help="Required for horizons exceeding the trained six-step setting")
    parser.add_argument("--gt-frames", type=int, default=35)
    parser.add_argument("--export-label-space", choices=("native", "occ3d18"), default="occ3d18")
    parser.add_argument("--allow-training-scene", action="store_true", help="Debug only; recorded in metadata")
    return parser


def straight_plan(history_rows, horizon, fps, speed_mps=None):
    """Constant heading, speed and height, beginning at the first future step.

    The speed policy is a lab-specific choice, not an ADTT paper hyperparameter.
    Only historical poses are consulted when estimating speed.
    """
    if fps <= 0 or not history_rows:
        raise ValueError("Positive FPS and historical poses are required")
    last = history_rows[-1]
    rotation = np.asarray(last["ego2global_rotation"], dtype=float)
    w, x, y, z = rotation
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    if speed_mps is None:
        if len(history_rows) < 2:
            raise ValueError("Speed estimation needs two observed poses")
        a, b = history_rows[-2:]
        dt = (b["timestamp"] - a["timestamp"]) / 1e6
        if dt <= 0:
            raise ValueError("Invalid history timestamps")
        delta = np.asarray(b["ego2global_translation"]) - np.asarray(a["ego2global_translation"])
        speed_mps = float(delta[:2] @ np.asarray([np.cos(yaw), np.sin(yaw)]) / dt)
    if not np.isfinite(speed_mps) or speed_mps < 0:
        raise ValueError("Straight forward rollout requires finite nonnegative speed")
    origin = np.asarray(last["ego2global_translation"], dtype=float)
    direction = np.asarray([np.cos(yaw), np.sin(yaw), 0.0])
    future_translation = [origin + direction * speed_mps * (i + 1) / fps for i in range(horizon)]
    translations = [r["ego2global_translation"] for r in history_rows] + [v.tolist() for v in future_translation]
    rotations = [r["ego2global_rotation"] for r in history_rows] + [rotation.tolist() for _ in range(horizon)]
    return dict(ego2global_translation_seq=translations, ego2global_rotation_seq=rotations), speed_mps


@torch.no_grad()
def rollout_main(args, run, guard):
    model, config, saved, device = load_trained(args, guard)
    if config["stage"] != "forecast":
        raise ValueError("Rollout requires a trained forecasting checkpoint")
    if args.frames < 1 or args.gt_frames < 0 or args.fps <= 0:
        raise ValueError("Invalid frame count/FPS")
    if args.start_token and args.start_index is not None:
        raise ValueError("Choose start-token or start-index")
    if args.frames > config["forecast"]["future"] and args.embedding_policy is None:
        raise ValueError("Long-horizon embedding policy must be selected explicitly and validated")
    rows = [r for r in read_manifest(args.manifest) if r["scene"] == args.scene]
    if not rows:
        raise ValueError("Scene not found")
    is_training_scene = args.scene in saved["provenance"]["training_scenes"]
    if is_training_scene and not args.allow_training_scene:
        raise ValueError("Training-scene export requires explicit --allow-training-scene (debug only)")
    history = config["forecast"]["history"]
    if args.start_token:
        start = next((i for i, r in enumerate(rows) if r["token"] == args.start_token), None)
        if start is None:
            raise ValueError("Start token not found")
    else:
        start = history if args.start_index is None else args.start_index
    if start < history or start >= len(rows):
        raise ValueError("Not enough history or start is outside scene")
    observed_rows = rows[start - history:start]
    if args.trajectory == "straight":
        if args.speed_mps is None and not args.speed_from_history:
            raise ValueError("Select --speed-mps or --speed-from-history explicitly")
        meta, speed = straight_plan(observed_rows, args.frames, args.fps, args.speed_mps)
    else:
        if args.speed_mps is not None or args.speed_from_history:
            raise ValueError("Speed overrides do not apply to a recorded trajectory")
        planned = rows[start - history:start + args.frames]
        if len(planned) != history + args.frames:
            raise ValueError("Recorded trajectory does not cover requested horizon; no synthetic GT is invented")
        meta = dict(ego2global_translation_seq=[r["ego2global_translation"] for r in planned],
                    ego2global_rotation_seq=[r["ego2global_rotation"] for r in planned])
        speed = None
    # Only observed occupancy is read before prediction. Future rows supply poses
    # (recorded mode) or later GT export, never predictor input.
    observed = torch.stack([load_frame(row, config["data"])[0] for row in observed_rows])[None].to(device)
    period = config["forecast"]["future"] if args.embedding_policy == "repeat-training-horizon" else None
    planes = forecast(model, observed, [meta], args.frames, guard.step, period)
    output = run / "occupancy" / args.scene
    output.mkdir(parents=True)
    mapping = np.asarray(STATIC_TO_OCC3D, dtype=np.uint8) if (
        config["data"]["label_space"] == "static10" and args.export_label_space == "occ3d18") else None
    shape = (args.frames, *config["data"]["grid_size"])
    predictions = np.lib.format.open_memmap(output / "pred.npy", mode="w+", dtype=np.uint8, shape=shape)
    empty_percent = []
    for step in range(args.frames):
        guard.step()
        prediction = dense_prediction(model, [p[:, step] for p in planes],
            config["data"]["grid_size"], config["training"]["decode_chunk"], device).numpy()
        free = model.num_cls - 1
        empty_percent.append(float(100 * (prediction == free).mean()))
        predictions[step] = mapping[prediction] if mapping is not None else prediction
    predictions.flush()
    del predictions
    gt_rows = rows[start:min(start + args.gt_frames, len(rows))]
    gt = np.stack([load_frame(row, config["data"])[0].numpy().astype(np.uint8) for row in gt_rows]) if gt_rows else np.empty((0, *shape[1:]), dtype=np.uint8)
    if mapping is not None:
        gt = mapping[gt]
    with (output / "gt.npy").open("xb") as stream:
        np.save(stream, gt, allow_pickle=False)
    atomic_json(run / "metadata.json", dict(
        scene=args.scene, first_future_index=start, history_tokens=[r["token"] for r in observed_rows],
        split="train (debug)" if is_training_scene else "held out from training",
        recorded_GT_timestamps_us=[r["timestamp"] for r in gt_rows],
        gt_tokens=[r["token"] for r in gt_rows], GT_frames=len(gt_rows), predicted_frames=args.frames,
        trajectory=args.trajectory, fps=args.fps, speed_mps=speed,
        embedding_policy=args.embedding_policy or "native", paired_with_GT=args.trajectory == "recorded",
        GT_source="provided semantic grid; static10 remains static after namespace mapping",
        checkpoint_sha256=sha256(args.checkpoint), checkpoint_step=saved["step"],
        source_label_space=config["data"]["label_space"], export_label_space=args.export_label_space,
        poses=meta, empty_percent=empty_percent))
    reloaded = np.load(output / "pred.npy", mmap_mode="r", allow_pickle=False)
    if reloaded.shape != shape or reloaded.dtype != np.uint8:
        raise RuntimeError("Prediction export read-back failed")
    atomic_json(run / "provenance.json", source_receipt(run))
    print(json.dumps(dict(status="rollout_export_complete", predictions=str(output / "pred.npy"),
                          GT_frames=len(gt_rows), frames=args.frames), indent=2))
