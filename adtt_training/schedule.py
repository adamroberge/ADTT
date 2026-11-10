"""Explicit, resumable learning-rate factors indexed by completed updates."""
import math


def learning_rate_factor(step, training):
    """LR for update step+1: linear warmup, then cosine through the last update.

    Warmup updates include the start and peak. Cosine begins at the peak on the
    next update and reaches min_lr_ratio on the final configured update. The
    interpolation convention is reconstructed; no v5 author schedule is released.
    """
    warmup = training["warmup_steps"]
    if step < warmup:
        start = training.get("warmup_start_ratio", 0.001)
        return start + (1 - start) * step / (warmup - 1) if warmup > 1 else 1.0
    if training["schedule"] == "constant":
        return 1.0
    duration = training["max_steps"] - warmup
    if duration == 1:
        return training["min_lr_ratio"]
    progress = min(max(step - warmup, 0) / max(duration - 1, 1), 1.0)
    floor = training["min_lr_ratio"]
    return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2
