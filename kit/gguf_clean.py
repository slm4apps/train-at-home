#!/usr/bin/env python3
"""Copy a GGUF with release metadata: your name, nothing from the machine that converted it.

    python3 gguf_clean.py IN.gguf OUT.gguf --name "My Model 0.5B" [--chat-template-file FILE]
    python3 gguf_clean.py IN.gguf --scan                      # only list what would leak, change nothing

Converters write what they find on the converting machine: general.name and basename guessed from the training
folder, general.base_model.* and general.tags made from a local path, the base model's stock chat template with its
stock identity line. This copy sets general.name, removes those keys, optionally replaces the chat template, keeps every
tensor and every other key, then reads the copy back and fails (deleting it) if a string still looks like a home path,
a private address or a user name. Needs: pip install gguf==0.17.1
"""
import argparse, os, re, subprocess, sys
from gguf import GGUFReader

DROP_PREFIX = ("general.base_model.", "general.source.", "general.dataset.")
DROP = {"general.basename", "general.finetune", "general.tags", "general.url", "general.repo_url",
        "general.organization", "general.author", "general.base_model.count"}
LEAK = re.compile(r"(/Users/|/home/[a-z]|C:\\Users\\|\b(?:10|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d+\.\d+(?:\.\d+)?\b|\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d+\.\d+)")


def strings(path):
    """(key, text) for every string-valued metadata field."""
    out = []
    r = GGUFReader(path)
    for k, f in r.fields.items():
        try:
            v = f.contents()
        except Exception:
            continue
        if isinstance(v, str):
            out.append((k, v))
        elif isinstance(v, list) and v and isinstance(v[0], str) and len(v) < 64:
            out.extend((k, x) for x in v)
    return r, out


def scan(path):
    r, items = strings(path)
    found = [(k, LEAK.search(v).group(0)[:12] + "...") for k, v in items if LEAK.search(v)]
    for k, v in items:
        if k in DROP or k.startswith(DROP_PREFIX):
            found.append((k, "(key that usually leaks the build machine)"))
    return r, found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--name")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--chat-template-file")
    a = ap.parse_args()
    reader, found = scan(a.src)
    if a.scan or not a.dst:
        for k, why in found:
            print(f"{k}: {why}")
        print("clean" if not found else f"{len(found)} finding(s)")
        return 1 if found else 0
    if not a.name:
        sys.exit("--name is required")
    cmd = [sys.executable, "-m", "gguf.scripts.gguf_new_metadata", a.src, a.dst, "--general-name", a.name, "--force"]
    for k in reader.fields:
        if k in DROP or k.startswith(DROP_PREFIX):
            cmd += ["--remove-metadata", k]
    if a.chat_template_file:
        cmd += ["--chat-template", open(a.chat_template_file).read()]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("gguf_new_metadata failed:\n" + r.stderr[-2000:])
    _, left = scan(a.dst)
    left = [x for x in left if not x[1].startswith("(key")]
    if left:
        os.remove(a.dst)
        sys.exit("still not clean, copy removed: " + ", ".join(k for k, _ in left))
    print("clean:", a.dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
