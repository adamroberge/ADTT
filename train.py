#!/usr/bin/env python3
"""Train either ADTT stage; default device is CPU, CUDA requires an assigned UUID."""
import sys
sys.dont_write_bytecode = True

from adtt_training.cli import train_main

if __name__ == "__main__":
    train_main()
