"""CLI bootstrap: establish output/cache paths and CUDA mask before Torch."""
import argparse
import json
import os
import tempfile
from pathlib import Path

from .config import load_config, validate_config
from .device import select_device


def parser(mode="train"):
    argument = argparse.ArgumentParser(description=f"Reconstructed two-stage ADTT {mode}")
    argument.add_argument("--config", required=mode == "train")
    argument.add_argument("--work-dir", required=True)
    argument.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    argument.add_argument("--gpu-uuid", help="One explicitly authorized GPU UUID; never auto-select")
    argument.add_argument("--allow-display-pid", type=int, help="Only if authorized: pin gdm/Xorg identity")
    argument.add_argument("--memory-fraction", type=float, default=0.8)
    argument.add_argument("--train-manifest")
    argument.add_argument("--val-manifest")
    argument.add_argument("--max-steps", type=int)
    argument.add_argument("--validation-frames", type=int, help="0 evaluates all validation items")
    argument.add_argument("--label-space", choices=("static10", "occ3d18"))
    argument.add_argument("--resume", help="Resume into a NEW output directory")
    argument.add_argument("--representation-checkpoint")
    argument.add_argument("--stop-after", type=int, help="Bound this invocation without changing the training schedule")
    return argument


def bootstrap(args):
    if not 0 < args.memory_fraction <= 1:
        raise ValueError("memory-fraction must be in (0,1]")
    run = Path(args.work_dir).expanduser().absolute()
    if run.resolve() != run:
        raise ValueError("Output path must be canonical, not a symlink")
    workspace = os.environ.get("ADTT_ROOT")
    if workspace and not run.is_relative_to(Path(workspace).resolve(strict=True)):
        raise ValueError("With ADTT_ROOT set, all outputs must remain inside that private workspace")
    if os.uname().nodename.split(".")[0] == "shannon" and not run.is_relative_to(Path("/user_data/tianranliu/adam")):
        raise ValueError("Shannon writes must remain inside /user_data/tianranliu/adam")
    if run.exists() or run.is_symlink():
        raise FileExistsError("Use a fresh work-dir; existing runs are never overwritten")
    run.mkdir(parents=True, mode=0o700)
    for variable, directory in {
        "TMPDIR": "tmp", "XDG_CACHE_HOME": "cache", "XDG_CONFIG_HOME": "config-cache",
        "XDG_DATA_HOME": "data-cache", "XDG_STATE_HOME": "state-cache",
        "TORCH_HOME": "cache/torch", "TORCH_EXTENSIONS_DIR": "cache/torch-extensions",
        "CUDA_CACHE_PATH": "cache/cuda", "TRITON_CACHE_DIR": "cache/triton",
        "NUMBA_CACHE_DIR": "cache/numba", "MPLCONFIGDIR": "cache/matplotlib",
        "HF_HOME": "cache/huggingface", "MMENGINE_HOME": "cache/mmengine",
        "TORCHINDUCTOR_CACHE_DIR": "cache/torchinductor",
    }.items():
        path = run / directory
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    tempfile.tempdir = None  # Respect the private TMPDIR even if stdlib cached an earlier location.
    try:
        guard = select_device(args.device, args.gpu_uuid, args.allow_display_pid)
    except Exception as error:
        with (run / "launch-failure.json").open("x") as stream:
            json.dump(dict(status="launch_refused", error=repr(error)), stream, indent=2)
        raise
    return run, guard


def train_main():
    args = parser().parse_args()
    config = load_config(args.config)
    for option in ("train_manifest", "val_manifest", "label_space"):
        if getattr(args, option) is not None:
            config["data"][option] = getattr(args, option)
    for option in ("max_steps", "validation_frames"):
        if getattr(args, option) is not None:
            config["training"][option] = getattr(args, option)
    validate_config(config)
    if args.stop_after is not None and args.stop_after < 1:
        raise ValueError("stop-after must be positive")
    if config["stage"] == "forecast" and not (args.resume or args.representation_checkpoint):
        raise ValueError("Forecast training requires --representation-checkpoint (or a full --resume)")
    run, guard = bootstrap(args)
    # No CUDA-aware imports above this point.
    from .engine import train
    try:
        train(config, run, guard, args)
    except Exception as error:
        print(json.dumps({"status": "FAILED", "error": repr(error), "run": str(run)}), flush=True)
        raise
