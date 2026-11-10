"""Atomic checkpoints, complete RNG state and exact source/data receipts."""
import importlib.metadata
import json
import os
import random
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch

from .manifest import sha256


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def rng_state():
    state = np.random.get_state()
    return dict(python=random.getstate(), torch=torch.get_rng_state(),
                numpy=(state[0], state[1].tolist(), state[2], state[3], state[4]),
                cuda=torch.cuda.get_rng_state(0) if torch.cuda.is_initialized() else None)


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    value = state["numpy"]
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), *value[2:]))
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("A CUDA training resume requires its CUDA RNG state/device")
        torch.cuda.set_rng_state(state["cuda"], 0)


def load_checkpoint(path):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("format") != "adtt-two-stage-v1":
        raise ValueError("Expected an ADTT two-stage checkpoint; convert author checkpoints explicitly")
    return value


def save_checkpoint(path, payload):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    # Atomic publication without overwriting any pre-existing artifact.
    os.link(temporary, path)
    temporary.unlink()
    return sha256(path)


def source_receipt(run):
    root = Path(__file__).resolve().parents[1]
    files = []
    for directory in ("model", "loss", "dataset", "utils", "adtt_training", "configs", "tools"):
        if (root / directory).exists():
            files.extend(p for p in (root / directory).rglob("*") if p.is_file() and p.suffix in {".py", ".yaml"})
    files.extend(p for name in ("README.md", "requirements-training.txt", "LICENSE", "NOTICE",
                               "THIRD_PARTY_NOTICES.md", "citation.bib", "train.py", "evaluate.py", "rollout.py")
                 if (p := root / name).exists())
    files.extend((root / "docs").glob("*.md"))
    files.extend((root / "licenses").glob("*.txt"))
    hashes = {str(path.relative_to(root)): sha256(path) for path in sorted(files)}
    with zipfile.ZipFile(run / "source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, str(path.relative_to(root)))
    try:
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    packages = {}
    for name in ("torch", "torchvision", "mmengine", "numpy", "scipy", "PyYAML"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    training_source = {name: value for name, value in hashes.items() if name.endswith(".py") and
                       (name.startswith(("model/", "adtt_training/")) or name == "train.py")}
    return dict(git_revision=revision, source_sha256=hashes, training_source_sha256=training_source, packages=packages,
                source_archive_sha256=sha256(run / "source.zip"))
