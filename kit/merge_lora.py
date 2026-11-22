#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into its base model and save a full-precision Hugging Face folder.

    python3 merge_lora.py --base BASE_DIR --adapter out/run1/final --out out/run1-merged [--scale 1.0]

--scale multiplies the adapter before merging (0.7 = 70% strength). Merging is ONE point on the strength curve:
measure at least three scales (score_checkpoints.py --scales) before you pick. The merged folder is then converted
with convert_gguf.sh. Always merge into the FULL-PRECISION base, never into something rebuilt from a GGUF.
"""
import argparse, json, os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--scale", type=float, default=1.0)
ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
a = ap.parse_args()

dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.dtype]
if a.scale != 1.0:
    # scale the adapter by scaling every lora_B matrix; written to a temp copy, the input is left alone
    from safetensors.torch import load_file, save_file
    import shutil, tempfile
    tmp = tempfile.mkdtemp()
    for f in os.listdir(a.adapter):
        if os.path.isfile(os.path.join(a.adapter, f)):
            shutil.copy(os.path.join(a.adapter, f), tmp)
    sd = load_file(os.path.join(tmp, "adapter_model.safetensors"))
    sd = {k: (v * a.scale if ".lora_B." in k else v) for k, v in sd.items()}
    save_file(sd, os.path.join(tmp, "adapter_model.safetensors"), metadata={"format": "pt"})
    adapter_dir = tmp
else:
    adapter_dir = a.adapter

model = AutoModelForCausalLM.from_pretrained(a.base, dtype=dt)
model = PeftModel.from_pretrained(model, adapter_dir)
model = model.merge_and_unload()
os.makedirs(a.out, exist_ok=True)
model.save_pretrained(a.out, safe_serialization=True)
AutoTokenizer.from_pretrained(a.base).save_pretrained(a.out)
json.dump({"base": a.base, "adapter": a.adapter, "scale": a.scale, "dtype": a.dtype},
          open(os.path.join(a.out, "MERGE.json"), "w"), indent=1)
print("merged ->", a.out)
