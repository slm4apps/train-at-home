#!/bin/sh
# Prove a setup works in minutes, before any long run.
#
#   kit/smoke.sh --dry-run                      # no training, no model needed: checks the kit itself and your tools
#   kit/smoke.sh --path mlx  --base BASE        # 20 real training iterations on the sample data (Apple silicon)
#   kit/smoke.sh --path cuda --base BASE        # GPU check + 10 real updates on the sample data (NVIDIA)
#
# BASE = a small instruct model as a Hugging Face folder or id (a 0.5B model is plenty).
# Prints the speed it measured: compare it with the speed you planned the long run on.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python3}"
PATHNAME=""; BASE=""; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --path) PATHNAME="$2"; shift ;;
    --base) BASE="$2"; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
  shift
done
[ "$DRY" = 1 ] || [ -n "$PATHNAME" ] || { echo "give --dry-run, or --path mlx|cuda --base BASE" >&2; exit 2; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/train-at-home-smoke.XXXXXX")"
echo "work folder: $WORK"
ok() { echo "OK   $1"; }

"$PY" "$HERE/make_sample_data.py" "$WORK/data" >/dev/null && ok "sample data written"
"$PY" "$HERE/prepare_data.py" check "$WORK/data" >/dev/null && ok "dataset check (format, no train/test leak)"
"$PY" "$HERE/prepare_data.py" per-turn "$WORK/data" "$WORK/data-turns" >/dev/null && ok "per-turn rows written (for MLX)"
for f in "$HERE"/*.py; do "$PY" -m py_compile "$f"; done && ok "every kit script compiles"
for f in "$HERE"/*.sh; do sh -n "$f"; done && ok "every kit shell script parses"
"$PY" "$HERE/train_lora_cuda.py" --help >/dev/null && ok "train_lora_cuda.py --help"
DRY_RUN=1 sh "$HERE/train_lora_mlx.sh" some/base "$WORK/data-turns" "$WORK/mlx-out" >/dev/null </dev/null && ok "train_lora_mlx.sh dry run"
for t in llama-server; do command -v "$t" >/dev/null 2>&1 && ok "$t found" || echo "MISS $t (needed for scoring and serving checks: install llama.cpp)"; done
"$PY" -c "import mlx_lm" 2>/dev/null && ok "mlx_lm importable" || echo "MISS mlx_lm (only needed for the MLX path)"
"$PY" -c "import torch, peft, transformers" 2>/dev/null && ok "torch + peft + transformers importable" || echo "MISS torch/peft/transformers (only needed for the CUDA path and for merging)"

if [ "$DRY" = 1 ]; then echo "dry run done: nothing was trained."; exit 0; fi
[ -n "$BASE" ] || { echo "--base is required for a real smoke run" >&2; exit 2; }

case "$PATHNAME" in
  mlx)
    START=$(date +%s)
    PY="$PY" sh "$HERE/train_lora_mlx.sh" "$BASE" "$WORK/data-turns" "$WORK/mlx-run" 20
    END=$(date +%s)
    echo "20 iterations took $((END - START)) s (includes model load). Look at 'Tokens/sec' in the log above." ;;
  cuda)
    "$PY" "$HERE/train_lora_cuda.py" --check-gpu
    "$PY" "$HERE/train_lora_cuda.py" --base "$BASE" --rows "$WORK/data/train.jsonl" --out "$WORK/cuda-run" \
      --updates 10 --save-every 5
    echo "see seconds per update in $WORK/cuda-run/train_log.jsonl" ;;
  *) echo "--path must be mlx or cuda" >&2; exit 2 ;;
esac
echo "smoke run done. Adapter in $WORK. Next: GUIDE.md chapter 'Measure'."
