"""Configuration loading and label contracts; no CUDA-aware imports."""
import copy
import hashlib
import json
import os
from pathlib import Path

import yaml

LABELS = {
    "static10": ["others", "barrier", "traffic_cone", "driveable", "other_flat",
                 "sidewalk", "terrain", "manmade", "vegetation", "free"],
    "occ3d18": ["others", "barrier", "bicycle", "bus", "car", "construction_vehicle",
                "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck",
                "driveable", "other_flat", "sidewalk", "terrain", "manmade",
                "vegetation", "free"],
}
STATIC_TO_OCC3D = [0, 1, 8, 11, 12, 13, 14, 15, 16, 17]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_config(path, _seen=None):
    path = Path(path).resolve(strict=True)
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError("Circular base configuration")
    seen.add(path)
    config = yaml.safe_load(path.read_text())
    base = config.pop("base", None)
    if base:
        inherited = load_config(path.parent / base, seen)
        def merge(left, right):
            for key, value in right.items():
                if isinstance(value, dict) and isinstance(left.get(key), dict):
                    merge(left[key], value)
                else:
                    left[key] = value
        merge(inherited, config)
        config = inherited
    validate_config(config)
    return config


def validate_config(config):
    if config.get("stage") not in {"representation", "forecast"}:
        raise ValueError("stage must be representation or forecast")
    data = config["data"]
    if data["label_space"] not in LABELS:
        raise ValueError("Unknown label_space")
    if len(data["grid_size"]) != 3 or any(int(n) < 4 for n in data["grid_size"]):
        raise ValueError("grid_size must contain three dimensions >= 4")
    if data.get("visibility", "all") not in {"all", "camera", "lidar"}:
        raise ValueError("visibility must be all, camera or lidar")
    for key in ("batch_size", "max_steps", "validate_every", "checkpoint_every",
                "decode_chunk", "keep_checkpoints"):
        if int(config["training"][key]) < 1:
            raise ValueError(f"training.{key} must be positive")
    if int(config["training"]["validation_frames"]) < 0:
        raise ValueError("validation_frames must be >= 0 (0 means all)")
    if int(config["sampling"]["points"]) < 1:
        raise ValueError("sampling.points must be positive")
    volume = 1
    for size in data["grid_size"]:
        volume *= size
    if config["sampling"]["points"] > volume:
        raise ValueError("sampling.points must not exceed the grid volume")
    if config["sampling"]["strategy"] not in {"balanced", "random"}:
        raise ValueError("Invalid sampling strategy")
    if not 0 <= float(config["sampling"]["occupied_ratio"]) <= 1:
        raise ValueError("occupied_ratio must lie in [0,1]")
    if int(config["training"].get("num_workers", 0)) < 0:
        raise ValueError("num_workers must be nonnegative")
    if config["training"]["schedule"] not in {"constant", "cosine"}:
        raise ValueError("schedule must be constant or cosine")
    if config["training"]["optimizer"] not in {"adam", "adamw"}:
        raise ValueError("optimizer must be adam or adamw")
    if not 0 <= config["training"]["warmup_steps"] < config["training"]["max_steps"]:
        raise ValueError("warmup_steps must be in [0,max_steps)")
    if config["training"]["lr"] <= 0 or config["training"]["clip_norm"] <= 0:
        raise ValueError("lr and clip_norm must be positive")
    if any(config["loss"][name] < 0 for name in ("ce", "lovasz", "latent_l1", "latent_l2", "occupancy")):
        raise ValueError("Loss weights must be nonnegative")
    if config["loss"]["ce"] + config["loss"]["lovasz"] <= 0:
        raise ValueError("At least one occupancy loss must have a positive weight")
    forecast = config["forecast"]
    if int(forecast["history"]) < 1 or int(forecast["future"]) < 1:
        raise ValueError("history/future must be positive")
    if forecast["token_dim"] % forecast["heads"]:
        raise ValueError("token_dim must be divisible by heads")
    tokens = 1
    for size in forecast["token_grid"]:
        tokens *= int(size)
    if tokens != forecast["ego_queries"] + forecast["external_queries"]:
        raise ValueError("ego_queries + external_queries must equal token grid volume")


def resume_contract(config):
    """Operational logging frequency may change; mathematical settings may not."""
    result = copy.deepcopy(config)
    for key in ("validate_every", "checkpoint_every", "validation_frames",
                "decode_chunk", "keep_checkpoints", "num_workers"):
        result["training"].pop(key, None)
    result["data"].pop("train_manifest", None)
    result["data"].pop("val_manifest", None)
    if result["training"]["schedule"] == "constant":
        result["training"].pop("max_steps", None)
    return result


def manifest_path(value):
    expanded = os.path.expandvars(value)
    if "$" in expanded:
        raise ValueError(f"Unresolved environment variable in {value!r}")
    return Path(expanded).expanduser().resolve(strict=True)
