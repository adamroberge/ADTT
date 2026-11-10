"""Read-only metadata conversion; no Torch import or dataset writes."""
import codecs
import collections
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MetadataReader(pickle.Unpickler):
    """Accept basic containers/NumPy metadata, never arbitrary pickle globals."""
    def find_class(self, module, name):
        module = module.replace("numpy._core", "numpy.core").replace("__builtin__", "builtins")
        allowed = {
            ("numpy", "ndarray"): np.ndarray, ("numpy", "dtype"): np.dtype,
            ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
            ("numpy.core.multiarray", "scalar"): np.core.multiarray.scalar,
            ("numpy.core.numeric", "_frombuffer"): np.core.numeric._frombuffer,
            ("collections", "OrderedDict"): collections.OrderedDict,
            ("_codecs", "encode"): codecs.encode,
            ("builtins", "set"): set, ("builtins", "frozenset"): frozenset,
            ("builtins", "slice"): slice,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unsupported metadata global: {module}.{name}")
        return allowed[module, name]


def validate_rows(rows):
    if not rows:
        raise ValueError("Empty manifest")
    tokens, groups = set(), collections.defaultdict(list)
    for row in rows:
        if not isinstance(row["scene"], str) or not row["scene"] or not isinstance(row["token"], str):
            raise ValueError("Scene/token must be strings")
        if row["token"] in tokens:
            raise ValueError(f"Duplicate token: {row['token']}")
        tokens.add(row["token"])
        translation = np.asarray(row["ego2global_translation"], dtype=float)
        rotation = np.asarray(row["ego2global_rotation"], dtype=float)
        if translation.shape != (3,) or rotation.shape != (4,):
            raise ValueError("Expected xyz translation and wxyz quaternion")
        if not np.isfinite(translation).all() or not np.isfinite(rotation).all() or abs(np.linalg.norm(rotation) - 1) > 1e-3:
            raise ValueError(f"Non-finite/non-unit pose: {row['token']}")
        if not isinstance(row["timestamp"], int):
            raise ValueError("timestamp must be integer nuScenes microseconds")
        if not Path(row["semantics_path"]).is_file():
            raise FileNotFoundError(row["semantics_path"])
        groups[row["scene"]].append(row)
    ordered = []
    for scene in sorted(groups):
        sequence = sorted(groups[scene], key=lambda row: row["timestamp"])
        for previous, current in zip(sequence, sequence[1:]):
            if current.get("prev") != previous["token"] or current["timestamp"] <= previous["timestamp"]:
                raise ValueError(f"Broken temporal chain in {scene}: {current['token']}")
        ordered.extend(sequence)
    return ordered


def read_manifest(path):
    path = Path(path).resolve(strict=True)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        location = Path(row["semantics_path"])
        row["semantics_path"] = str((path.parent / location).resolve(strict=True))
    return validate_rows(rows)


def prepare_manifest(metadata_path, data_root, output):
    metadata_path, data_root = Path(metadata_path).resolve(strict=True), Path(data_root).resolve(strict=True)
    output = Path(output).absolute()
    if output.resolve().is_relative_to(data_root):
        raise ValueError("Manifest output must be outside the read-only occupancy dataset")
    with metadata_path.open("rb") as stream:
        document = MetadataReader(stream).load()
    infos = document["infos"]
    entries = [(scene, item) for scene, values in infos.items() for item in values] if isinstance(infos, dict) else [(None, item) for item in infos]
    rows = []
    for group_scene, info in entries:
        scene = info.get("scene_name", group_scene)
        if scene is None:
            pieces = Path(info["occ_path"]).parts
            scene = next((part for part in pieces if part.startswith("scene-")), None)
        if scene is None:
            raise ValueError("Cannot infer scene name from metadata")
        token = str(info["token"])
        if Path(scene).name != scene or Path(token).name != token or token in {".", ".."}:
            raise ValueError("Unsafe scene/token path")
        labels = (data_root / scene / token / "labels.npz").resolve(strict=True)
        if not labels.is_relative_to(data_root):
            raise ValueError("Occupancy path escapes data root")
        rows.append(dict(scene=scene, token=token, prev=str(info.get("prev", "")),
                         timestamp=int(info["timestamp"]), semantics_path=str(labels),
                         ego2global_translation=np.asarray(info["ego2global_translation"], dtype=float).tolist(),
                         ego2global_rotation=np.asarray(info["ego2global_rotation"], dtype=float).tolist()))
    rows = validate_rows(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    return dict(frames=len(rows), scenes=len({row["scene"] for row in rows}),
                input_sha256=sha256(metadata_path), manifest_sha256=sha256(output))
