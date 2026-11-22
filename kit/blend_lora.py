#!/usr/bin/env python3
"""Blend two LoRA adapters from the same run exactly: delta = w1*(up1 @ down1) + w2*(up2 @ down2)  (w2 = 1 - w1).

    python3 blend_lora.py ADAPTER_DIR_1 ADAPTER_DIR_2 W1 OUT_DIR

Done by stacking ranks (down = [down1; down2], up = [w1*up1, w2*up2]), so the result is one adapter of rank 2r with alpha doubled
(same scale alpha/r). It checks the blended delta against the weighted sum on a few layers and prints the error.
Use it to blend two checkpoints of one run, then score the blend like any other checkpoint.
"""
import json, os, sys
import torch
from safetensors.torch import load_file, save_file

if len(sys.argv) != 5:
    sys.exit(__doc__)
a1, a2, w1, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
w2 = 1.0 - w1
s1 = load_file(os.path.join(a1, "adapter_model.safetensors"))
s2 = load_file(os.path.join(a2, "adapter_model.safetensors"))
assert s1.keys() == s2.keys(), "the two adapters do not have the same tensors"
cfg = json.load(open(os.path.join(a1, "adapter_config.json")))
cfg2 = json.load(open(os.path.join(a2, "adapter_config.json")))
norm = lambda c: {k: (sorted(v) if k == "target_modules" else v) for k, v in c.items()}
assert norm(cfg) == norm(cfg2), "the two adapter configs differ"
r = cfg["r"]
o = {}
for k in s1:
    if ".lora_A." in k:
        o[k] = torch.cat([s1[k], s2[k]], 0)
    elif ".lora_B." in k:
        o[k] = torch.cat([w1 * s1[k].float(), w2 * s2[k].float()], 1).to(s1[k].dtype)
    else:
        raise SystemExit("unexpected tensor " + k)
os.makedirs(out, exist_ok=True)
save_file(o, os.path.join(out, "adapter_model.safetensors"), metadata={"format": "pt"})
cfg["r"], cfg["lora_alpha"] = 2 * r, 2 * cfg["lora_alpha"]
json.dump(cfg, open(os.path.join(out, "adapter_config.json"), "w"), indent=2)
json.dump({"mix": [[a1, w1], [a2, w2]]}, open(os.path.join(out, "MIX.json"), "w"), indent=1)

sc = cfg["lora_alpha"] / cfg["r"]
worst = 0.0
for k in [k for k in s1 if ".lora_A." in k][:6]:
    kb = k.replace(".lora_A.", ".lora_B.")
    d1 = s1[kb].float() @ s1[k].float()
    d2 = s2[kb].float() @ s2[k].float()
    dm = o[kb].float() @ o[k].float()
    want = (w1 * d1 + w2 * d2) * (cfg["lora_alpha"] / 2 / r)
    worst = max(worst, ((dm * sc - want).abs().max() / want.abs().max()).item())
print(f"blended {len(o)} tensors, rank {cfg['r']}, worst relative error {worst:.2e} -> {out}")
