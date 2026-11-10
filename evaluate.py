#!/usr/bin/env python3
"""Standalone evaluation, defaulting to CPU with dense per-class metrics."""
import sys
sys.dont_write_bytecode = True

if __name__ == "__main__":
    # Parse/bootstrap without importing Torch or torchvision.
    import argparse
    from adtt_training.cli import bootstrap
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--allow-display-pid", type=int)
    parser.add_argument("--memory-fraction", type=float, default=0.8)
    args = parser.parse_args()
    run, guard = bootstrap(args)
    from adtt_training.inference import evaluation_main
    evaluation_main(args, run, guard)
