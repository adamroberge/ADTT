"""Single-GPU launch checks using only stdlib/NVIDIA's read-only process report."""
import json
import os
import pwd
import re
import subprocess
import xml.etree.ElementTree as ET


class GPUConflict(RuntimeError):
    pass


class DeviceGuard:
    def __init__(self, uuid=None, allowed_display_pid=None):
        self.uuid = uuid
        # No display exception by default. Supply only after explicit authorization.
        self.display = None
        if allowed_display_pid is not None:
            entry = self._process(allowed_display_pid, "G", "/usr/lib/xorg/Xorg", 0)
            if entry["owner"] != "gdm":
                raise GPUConflict("Display exception requires verified gdm/Xorg")
            self.display = entry

    @staticmethod
    def _process(pid, kind, name, memory):
        pid = int(pid)
        root = f"/proc/{pid}"
        owner = pwd.getpwuid(os.stat(root).st_uid).pw_name
        with open(root + "/stat") as stream:
            start_ticks = stream.read().rsplit(")", 1)[1].split()[19]
        return dict(pid=pid, owner=owner, type=kind, name=name,
                    used_MiB=memory, start_ticks=start_ticks)

    def check(self, active=False):
        if self.uuid is None:
            return {"device": "cpu"}
        document = ET.fromstring(subprocess.check_output(
            ["nvidia-smi", "-i", self.uuid, "-q", "-x"], text=True, timeout=10))
        devices = document.findall("gpu")
        if len(devices) != 1 or devices[0].findtext("uuid") != self.uuid:
            raise GPUConflict("Assigned GPU UUID could not be verified")
        gpu = devices[0]
        section = gpu.find("processes")
        if section is None or (section.text or "").strip():
            raise GPUConflict("GPU process inventory unavailable; refusing use")
        def number(path, unit):
            match = re.fullmatch(r"([0-9]+)\s*" + unit, gpu.findtext(path) or "")
            if not match:
                raise GPUConflict(f"Unavailable GPU observation: {path}")
            return int(match.group(1))
        observation = dict(UUID=self.uuid, used_MiB=number("fb_memory_usage/used", "MiB"),
                           free_MiB=number("fb_memory_usage/free", "MiB"),
                           utilization=number("utilization/gpu_util", "%"), processes=[])
        conflicts = []
        for item in section.findall("process_info"):
            memory = re.fullmatch(r"([0-9]+)\s+MiB", item.findtext("used_memory") or "")
            if memory is None:
                raise GPUConflict("Unavailable per-process memory; refusing use")
            entry = self._process(item.findtext("pid"), item.findtext("type"),
                                  item.findtext("process_name"), int(memory.group(1)))
            observation["processes"].append(entry)
            if active and entry["pid"] == os.getpid():
                continue
            if self.display is not None and all(entry[key] == self.display[key] for key in
                    ("pid", "owner", "type", "name", "start_ticks")) and entry["used_MiB"] <= 8:
                continue
            conflicts.append(entry)
        if conflicts or (not active and (observation["used_MiB"] > 1024 or
                observation["free_MiB"] < 4096 or observation["utilization"] > 5)):
            raise GPUConflict("Assigned GPU is occupied: " + json.dumps(observation))
        return observation

    def step(self, _index=None):
        return self.check(active=True)


def select_device(device, uuid=None, allowed_display_pid=None):
    """Call BEFORE importing Torch. This is a visibility mask, not a reservation."""
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return DeviceGuard()
    if device != "cuda" or not uuid or not uuid.startswith("GPU-") or "," in uuid:
        raise ValueError("CUDA requires one explicitly assigned --gpu-uuid GPU-...")
    existing = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing not in (None, "", uuid):
        raise GPUConflict("Existing CUDA visibility conflicts with the requested UUID")
    guard = DeviceGuard(uuid, allowed_display_pid)
    guard.check()
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    return guard
