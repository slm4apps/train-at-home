#!/usr/bin/env python3
"""Weight-average full-model checkpoints of ONE training run (for models trained from scratch).

    python3 average_checkpoints.py OUT_DIR  CKPT_DIR_A:0.5 CKPT_DIR_B:0.5 [CKPT_DIR_C:...]

Each CKPT_DIR is a Hugging Face model folder (model.safetensors + config). Weights after the colon are normalised to
sum to 1. The first folder's config and tokenizer files are copied to OUT_DIR. Score the average like any checkpoint:
it often matches or beats the best single save, without choosing one by loss.
"""
import os, shutil, sys
from safetensors.torch import load_file, save_file

if len(sys.argv) < 3:
    sys.exit(__doc__)
out = sys.argv[1]
pairs = []
for s in sys.argv[2:]:
    d, w = s.rsplit(":", 1)
    pairs.append((d, float(w)))
tot = sum(w for _, w in pairs)
avg = None
for d, w in pairs:
    sd = load_file(os.path.join(d, "model.safetensors"))
    if avg is None:
        avg = {k: v.float() * (w / tot) for k, v in sd.items()}
        dtypes = {k: v.dtype for k, v in sd.items()}
    else:
        assert sd.keys() == avg.keys(), f"{d} has different tensors"
        for k, v in sd.items():
            avg[k] += v.float() * (w / tot)
os.makedirs(out, exist_ok=True)
save_file({k: v.to(dtypes[k]).contiguous() for k, v in avg.items()}, os.path.join(out, "model.safetensors"),
          metadata={"format": "pt"})
for f in os.listdir(pairs[0][0]):
    if f != "model.safetensors" and os.path.isfile(os.path.join(pairs[0][0], f)):
        shutil.copy(os.path.join(pairs[0][0], f), out)
print("averaged", len(pairs), "checkpoints ->", out)
