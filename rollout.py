#!/usr/bin/env python3
"""Trajectory-conditioned rollout; explicit policy selection for long horizons."""
import argparse
import sys
sys.dont_write_bytecode = True

if __name__ == "__main__":
    # Use a lightweight parser before any Torch import. The complete parser lives
    # in inference.py; CPU visibility is hidden by default during bootstrap.
    from adtt_training.cli import bootstrap
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--allow-display-pid", type=int)
    parser.add_argument("--memory-fraction", type=float, default=0.8)
    if "--help" in sys.argv or "-h" in sys.argv:
        # Help must never need an allocation or output directory.
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        from adtt_training.inference import rollout_parser
        rollout_parser().print_help()
        raise SystemExit(0)
    initial, _ = parser.parse_known_args()
    run, guard = bootstrap(initial)
    from adtt_training.inference import rollout_main, rollout_parser
    args = rollout_parser().parse_args()
    rollout_main(args, run, guard)
