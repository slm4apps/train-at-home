#!/bin/sh
# LoRA training on an Apple-silicon Mac with mlx_lm.
#
#   kit/train_lora_mlx.sh BASE DATA_DIR OUT_DIR [ITERS]
#     BASE      Hugging Face id or local folder of the base model (e.g. a small instruct model)
#     DATA_DIR  folder with train.jsonl + valid.jsonl, ALREADY per-turn: python3 prepare_data.py per-turn ...
#     OUT_DIR   adapters and logs go here
#     ITERS     default 600. In mlx_lm one iteration is ONE batch (batch size 1 here = one row).
#
# Settings: rank 8, scale 20 (= alpha 160), last 16 layers, batch 1, gradient accumulation 4, gradient checkpointing,
# max sequence 2048, lr 1e-4, loss on the last assistant message only (--mask-prompt), MLX capped at 4.5 GB.
# Environment: DRY_RUN=1 prints the command and the checks, trains nothing. EMBED=1 adds embed_tokens + lm_head to the
# LoRA keys (needed when special tokens or a tool-call tag matter). MIN_FREE=30 minimum free-memory percent.
set -e
BASE="${1:?base model}"; DATA="${2:?data dir}"; OUT="${3:?out dir}"; ITERS="${4:-600}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python3}"
MIN_FREE="${MIN_FREE:-30}"

[ -f "$DATA/train.jsonl" ] && [ -f "$DATA/valid.jsonl" ] || { echo "need $DATA/train.jsonl and valid.jsonl" >&2; exit 2; }
mkdir -p "$OUT"

# one training job at a time
if pgrep -f "mlx_capped.py|mlx_lm.lora" >/dev/null 2>&1 && [ "${DRY_RUN:-0}" != 1 ]; then
  echo "another mlx_lm training is running: one job at a time" >&2; exit 3
fi
# free memory (macOS)
if command -v memory_pressure >/dev/null 2>&1; then
  FREE="$(memory_pressure | sed -n 's/.*free percentage: *\([0-9]*\)%.*/\1/p')"
  echo "free memory: ${FREE:-unknown}% (minimum $MIN_FREE%)"
  if [ -n "$FREE" ] && [ "$FREE" -lt "$MIN_FREE" ] && [ "${DRY_RUN:-0}" != 1 ]; then
    echo "not starting: close other apps (browser, editors, a second model server) and try again" >&2; exit 4
  fi
fi

KEYS='["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]'
[ "${EMBED:-0}" = 1 ] && KEYS='["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj","embed_tokens","lm_head"]'
cat > "$OUT/lora_config.yaml" <<YAML
lora_parameters:
  rank: 8
  scale: 20.0
  dropout: 0.0
  keys: $KEYS
YAML

CMD="$PY $HERE/mlx_capped.py --cap-gb 4.5 -- --model $BASE --train --data $DATA --adapter-path $OUT/adapters \
--iters $ITERS --batch-size 1 --grad-accumulation-steps 4 --num-layers 16 --learning-rate 1e-4 \
--max-seq-length 2048 --grad-checkpoint --mask-prompt --save-every 100 --steps-per-eval 100 --steps-per-report 10 \
-c $OUT/lora_config.yaml"
echo "$CMD"
rows="$(wc -l < "$DATA/train.jsonl" | tr -d ' ')"
PASSES="$(awk -v i="$ITERS" -v r="$rows" 'BEGIN{printf "%.2f", i/r}')"
echo "length: $ITERS iterations. In mlx_lm 0.32.2 with batch size 1, one iteration was one row: $ITERS rows seen = $PASSES passes over $rows rows."
echo "Confirm with the 20-iteration trial (GUIDE.md, 'Counting honestly') and write the length as rows and passes."
[ "${DRY_RUN:-0}" = 1 ] && { echo "(dry run: nothing started)"; exit 0; }
# shellcheck disable=SC2086
exec $CMD 2>&1 | tee "$OUT/train.log"
