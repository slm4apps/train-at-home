#!/usr/bin/env python3
"""Run `mlx_lm lora` with a hard memory cap, and print the peak memory at the end.

    python3 mlx_capped.py [--cap-gb 4.5] -- <any mlx_lm lora arguments>

On an 8 GB Mac the model, the optimizer state, the activations and every open app share one pool. Capping MLX makes a
run slow down or fail early instead of pushing the machine into swap.
"""
import atexit, sys

args = sys.argv[1:]
cap = 4.5
if args and args[0] == "--cap-gb":
    cap = float(args[1])
    args = args[2:]
if args and args[0] == "--":
    args = args[1:]

import mlx.core as mx

mx.set_memory_limit(int(cap * 1024**3))
mx.set_cache_limit(512 * 1024**2)
atexit.register(lambda: print("PEAK_MEMORY_GB %.2f" % (mx.get_peak_memory() / 1024**3), flush=True))
from mlx_lm import lora

sys.argv = ["mlx_lm.lora"] + args
lora.main()
