#!/usr/bin/env python3
"""Convert existing flat/scene-grouped nuScenes metadata without modifying data."""
import argparse
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adtt_training.manifest import prepare_manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--data-root", required=True, help="Directory containing scene-*/token/labels.npz")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).absolute()
    workspace = os.environ.get("ADTT_ROOT")
    if workspace and not output.resolve().is_relative_to(Path(workspace).resolve(strict=True)):
        raise ValueError("Manifest output must remain inside ADTT_ROOT")
    if os.uname().nodename.split(".")[0] == "shannon" and not output.resolve().is_relative_to(Path("/user_data/tianranliu/adam")):
        raise ValueError("Shannon manifest writes must remain inside the approved private workspace")
    print(json.dumps(prepare_manifest(args.metadata, args.data_root, output), indent=2))
