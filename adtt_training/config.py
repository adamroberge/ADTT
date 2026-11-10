"""Configuration loading and label contracts; no CUDA-aware imports."""
import copy
import hashlib
import json
import math
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
    def finite(value, key):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number")

    for section in ("training", "loss", "sampling", "representation", "forecast", "data"):
        for key, value in config[section].items():
            if isinstance(value, float):
                finite(value, f"{section}.{key}")
    integer_fields = {
        "training": ("batch_size", "max_steps", "validate_every", "checkpoint_every", "decode_chunk",
                     "keep_checkpoints", "validation_frames", "num_workers", "seed", "warmup_steps"),
        "forecast": ("history", "future", "token_dim", "heads", "depth", "ego_queries", "external_queries"),
        "representation": ("channels", "decoder_hidden", "decoder_layers"),
        "sampling": ("points", "boundary_distance"),
    }
    for section, keys in integer_fields.items():
        for key in keys:
            value = config[section][key]
            if type(value) is not int or value < 0:
                raise ValueError(f"{section}.{key} must be a nonnegative integer")
    if config.get("stage") not in {"representation", "forecast"}:
        raise ValueError("stage must be representation or forecast")
    data = config["data"]
    if data["label_space"] not in LABELS:
        raise ValueError("Unknown label_space")
    if len(data["grid_size"]) != 3 or any(type(n) is not int or n < 4 for n in data["grid_size"]):
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
    if config["training"]["weight_decay"] < 0:
        raise ValueError("weight_decay must be nonnegative")
    for key in ("min_lr_ratio", "warmup_start_ratio"):
        if not 0 < config["training"].get(key, 0.001) <= 1:
            raise ValueError(f"{key} must lie in (0,1]")
    if data["max_gap_seconds"] <= 0:
        raise ValueError("max_gap_seconds must be positive")
    if any(config["representation"][key] < 1 for key in ("channels", "decoder_hidden", "decoder_layers")):
        raise ValueError("Representation dimensions must be positive")
    if any(config["loss"][name] < 0 for name in ("ce", "lovasz", "latent_l1", "latent_l2", "occupancy")):
        raise ValueError("Loss weights must be nonnegative")
    if config["loss"]["ce"] + config["loss"]["lovasz"] <= 0:
        raise ValueError("At least one occupancy loss must have a positive weight")
    forecast = config["forecast"]
    if int(forecast["history"]) < 1 or int(forecast["future"]) < 1:
        raise ValueError("history/future must be positive")
    if forecast["future"] > 32:
        raise ValueError("Training horizon exceeds the released 32-entry future-step table")
    if any(forecast[key] < 1 for key in ("token_dim", "heads", "depth")):
        raise ValueError("token_dim, heads and depth must be positive")
    if not 0 <= forecast["dropout"] < 1:
        raise ValueError("dropout must lie in [0,1)")
    if len(forecast["token_grid"]) != 3 or any(type(n) is not int or n < 1 for n in forecast["token_grid"]):
        raise ValueError("token_grid must contain three positive integers")
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
