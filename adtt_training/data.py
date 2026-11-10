"""Strict semantic grids and scene-contained temporal windows."""
import hashlib

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .config import LABELS
from .manifest import read_manifest


def load_frame(row, data):
    with np.load(row["semantics_path"], allow_pickle=False) as archive:
        labels = archive["semantics"]
        if labels.dtype != np.uint8 or tuple(labels.shape) != tuple(data["grid_size"]):
            raise ValueError(f"Invalid semantic grid: {row['semantics_path']}")
        if int(labels.min()) < 0 or int(labels.max()) >= len(LABELS[data["label_space"]]):
            raise ValueError(f"Invalid input label (including unsupported 255): {row['semantics_path']}")
        visibility = data.get("visibility", "all")
        if visibility == "all":
            mask = np.ones(labels.shape, dtype=bool)
        else:
            key = "mask_" + visibility
            if key not in archive:
                raise ValueError(f"Requested {key} is absent: {row['semantics_path']}")
            raw = archive[key]
            if raw.shape != labels.shape or not np.isin(raw, [0, 1]).all():
                raise ValueError("Invalid visibility mask")
            mask = raw.astype(bool)
    return torch.from_numpy(labels.astype(np.int64)), torch.from_numpy(mask.copy())


class OccupancyDataset(Dataset):
    def __init__(self, manifest, config):
        self.rows = read_manifest(manifest)
        self.data = config["data"]
        self.forecast = config["stage"] == "forecast"
        length = config["forecast"]["history"] + config["forecast"]["future"] if self.forecast else 1
        self.windows = []
        for begin in range(len(self.rows) - length + 1):
            sequence = self.rows[begin:begin + length]
            if sequence[0]["scene"] != sequence[-1]["scene"]:
                continue
            if any((b["timestamp"] - a["timestamp"]) / 1e6 > self.data["max_gap_seconds"]
                   for a, b in zip(sequence, sequence[1:])):
                continue
            self.windows.append(list(range(begin, begin + length)))
        if not self.windows:
            raise ValueError("No valid windows; check scene lengths and timestamp gap policy")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        rows = [self.rows[i] for i in self.windows[index]]
        frames = [load_frame(row, self.data) for row in rows]
        return dict(labels=torch.stack([frame[0] for frame in frames]),
                    mask=torch.stack([frame[1] for frame in frames]), rows=rows, index=index,
                    frame_content_sha256=[hashlib.sha256(frame[0].numpy().astype(np.uint8).tobytes()
                                           + frame[1].numpy().tobytes()).hexdigest() for frame in frames])

    def validation_indices(self, limit):
        if limit == 0 or limit >= len(self):
            return list(range(len(self)))
        by_scene = {}
        for index, window in enumerate(self.windows):
            by_scene.setdefault(self.rows[window[0]]["scene"], []).append(index)
        scenes = sorted(by_scene)
        if limit <= len(scenes):
            return [by_scene[scenes[int(i)]][len(by_scene[scenes[int(i)]]) // 2]
                    for i in np.linspace(0, len(scenes) - 1, limit, dtype=int)]
        return np.linspace(0, len(self) - 1, limit, dtype=int).tolist()


def collate(items):
    return dict(labels=torch.stack([item["labels"] for item in items]),
                mask=torch.stack([item["mask"] for item in items]),
                rows=[item["rows"] for item in items], indices=[item["index"] for item in items],
                frame_content_sha256=[item["frame_content_sha256"] for item in items])


class ResumableBatches(Sampler):
    def __init__(self, size, batch_size, seed, epoch, start_batch=0):
        self.size, self.batch_size = size, batch_size
        self.seed, self.epoch, self.start_batch = seed, epoch, start_batch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.size, generator=generator).tolist()
        for offset in range(self.start_batch * self.batch_size, self.size, self.batch_size):
            yield indices[offset:offset + self.batch_size]

    def __len__(self):
        return max(0, (self.size + self.batch_size - 1) // self.batch_size - self.start_batch)
