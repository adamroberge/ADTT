#!/usr/bin/env python3
"""Bounded qualification of both stages, resume, evaluation and 100-frame export.

Uses real manifest data and the supplied model/query configuration. No packages
are installed, GPU is never auto-selected, and no full training job is launched.
The coordinator imports no Torch; each child owns and releases the one GPU.
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adtt_training.cli import bootstrap
from adtt_training.config import load_config, validate_config
from adtt_training.manifest import read_manifest, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-config", default=str(ROOT / "configs/stage1_static.yaml"))
    parser.add_argument("--stage2-config", default=str(ROOT / "configs/stage2_static.yaml"))
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--allow-display-pid", type=int)
    parser.add_argument("--memory-fraction", type=float, default=0.8)
    parser.add_argument("--steps", type=int, default=2, help="Initial updates per stage; each is then resumed for one update")
    parser.add_argument("--scene", help="Validation scene for rollout; default first scene with sufficient frames")
    parser.add_argument("--rollout-frames", type=int, default=100)
    args = parser.parse_args()
    args.stage1_config = str(Path(args.stage1_config).resolve(strict=True))
    args.stage2_config = str(Path(args.stage2_config).resolve(strict=True))
    configs = [load_config(args.stage1_config), load_config(args.stage2_config)]
    for config, stage in zip(configs, ("representation", "forecast")):
        validate_config(config)
        if config["stage"] != stage or not 1 <= args.steps < config["training"]["max_steps"]:
            raise ValueError("Wrong stage or steps leave no room for the one-update resume")
    if args.rollout_frames < 1:
        raise ValueError("rollout-frames must be positive")
    args.train_manifest = str(Path(args.train_manifest).resolve(strict=True))
    args.val_manifest = str(Path(args.val_manifest).resolve(strict=True))
    train_rows, val_rows = read_manifest(args.train_manifest), read_manifest(args.val_manifest)
    if {r["scene"] for r in train_rows} & {r["scene"] for r in val_rows}:
        raise ValueError("Training and validation scenes overlap")
    sizes = {}
    for row in val_rows:
        sizes[row["scene"]] = sizes.get(row["scene"], 0) + 1
    needed = configs[1]["forecast"]["history"] + configs[1]["forecast"]["future"]
    candidates = sorted(s for s, count in sizes.items() if count >= needed)
    scene = args.scene or (candidates[0] if candidates else None)
    if scene not in candidates:
        raise ValueError("Selected validation scene does not have enough frames")
    run, guard = bootstrap(args)
    report = dict(status="RUNNING", device=args.device, gpu_uuid=args.gpu_uuid, checks=[],
                  configurations=configs, steps_per_initial_stage=args.steps,
                  train_manifest_sha256=sha256(args.train_manifest),
                  val_manifest_sha256=sha256(args.val_manifest),
                  qualification="Execution on sampled real data, not convergence or full-dataset quality")
    started = time.perf_counter()
    runtime = ["--device", args.device, "--memory-fraction", str(args.memory_fraction)]
    if args.gpu_uuid:
        runtime += ["--gpu-uuid", args.gpu_uuid]
    if args.allow_display_pid is not None:
        runtime += ["--allow-display-pid", str(args.allow_display_pid)]
    data = ["--train-manifest", args.train_manifest, "--val-manifest", args.val_manifest,
            "--validation-frames", "1"]

    def command(label, script, *arguments):
        guard.check()  # Every child must start on an idle, explicitly assigned device.
        argv = [sys.executable, "-B", str(ROOT / script), *map(str, arguments), *runtime,
                "--work-dir", str(run / label)]
        tick = time.perf_counter()
        print(f"[{label}] starting; log: {run / (label + '.log')}", flush=True)
        entry = dict(check=label, argv=argv, status="RUNNING")
        report["checks"].append(entry)
        with (run / (label + ".log")).open("x") as log:
            process = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        entry.update(returncode=process.returncode, seconds=time.perf_counter() - tick,
                     status="PASS" if process.returncode == 0 else "FAIL")
        if process.returncode:
            raise RuntimeError(f"{label} failed: inspect {run / (label + '.log')}")
        print(f"[{label}] PASS ({entry['seconds']:.1f}s)", flush=True)
        return run / label

    def checked_checkpoint(directory, expected_step):
        summary = json.loads((directory / "summary.json").read_text())
        if summary["status"] not in {"bounded_stop", "budget_complete"} or summary["step"] != expected_step:
            raise RuntimeError(f"Unexpected training status/step in {directory}")
        checkpoint = Path(summary["checkpoint"])
        latest = json.loads((directory / "latest.json").read_text())
        if summary["checkpoint_readback"] != "PASS" or sha256(checkpoint) != latest["sha256"]:
            raise RuntimeError("Checkpoint read-back or digest mismatch")
        events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
        updates = [e for e in events if e["event"] == "training"]
        if not updates or any(not math.isfinite(e[k]) for e in updates for k in ("total", "grad_norm", "lr")):
            raise RuntimeError("No finite optimizer updates recorded")
        report["checks"][-1].update(checkpoint_sha256=latest["sha256"],
            final_step=expected_step, peak_allocated_bytes=summary.get("peak_allocated_bytes"),
            peak_reserved_bytes=summary.get("peak_reserved_bytes"))
        return checkpoint

    try:
        first = command("stage1", "train.py", "--config", args.stage1_config, *data, "--stop-after", args.steps)
        ae = checked_checkpoint(first, args.steps)
        resumed = command("stage1-resume", "train.py", "--config", args.stage1_config, *data,
                          "--resume", ae, "--stop-after", 1)
        ae = checked_checkpoint(resumed, args.steps + 1)
        second = command("stage2", "train.py", "--config", args.stage2_config, *data,
                         "--representation-checkpoint", ae, "--stop-after", args.steps)
        wm = checked_checkpoint(second, args.steps)
        resumed = command("stage2-resume", "train.py", "--config", args.stage2_config, *data,
                          "--resume", wm, "--stop-after", 1)
        wm = checked_checkpoint(resumed, args.steps + 1)
        evaluation = command("evaluation", "evaluate.py", "--checkpoint", wm,
                             "--val-manifest", args.val_manifest, "--limit", 1)
        metrics = json.loads((evaluation / "metrics.json").read_text())["aggregate"]
        if metrics["voxels"] <= 0 or not math.isfinite(metrics["semantic_mIoU_nonfree"]):
            raise RuntimeError("Dense evaluation did not produce finite metrics")
        export = command("rollout", "rollout.py", "--checkpoint", wm, "--manifest", args.val_manifest,
                         "--scene", scene, "--trajectory", "straight", "--speed-mps", "1",
                         "--frames", args.rollout_frames, "--embedding-policy", "clamp-trained")
        metadata = json.loads((export / "metadata.json").read_text())
        if metadata["predicted_frames"] != args.rollout_frames or metadata["embedding_policy"] != "clamp-trained":
            raise RuntimeError("Rollout did not export the requested horizon/policy")
        report.update(status="PASS", rollout_scene=scene, rollout_frames=args.rollout_frames,
                      note="Diagnostic 1 m/s straight rollout; checkpoints are smoke artifacts, not trained baselines")
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        with (run / "qualification.json").open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(f"Qualification {report['status']}: {run / 'qualification.json'}", flush=True)


if __name__ == "__main__":
    main()
