#!/bin/sh
# Convert a trained model to GGUF with llama.cpp. Full-precision folder in, GGUF out.
#
#   kit/convert_gguf.sh model  HF_DIR   OUT.gguf [q8_0|f16|bf16]     # a full model folder (merged, or trained from scratch)
#   kit/convert_gguf.sh lora   ADAPTER  BASE_HF_DIR OUT.gguf [f16]   # a PEFT adapter as its own small GGUF
#   kit/convert_gguf.sh fuse-mlx BASE ADAPTER_DIR OUT_HF_DIR         # MLX path: merge an mlx_lm adapter into a HF folder
#
# LLAMA_CPP = path of a llama.cpp checkout (default ./llama.cpp). I used the release tag b11899; get it with
#   git clone https://github.com/ggml-org/llama.cpp && git -C llama.cpp checkout b11899
# and the converter's Python packages: pip install -r kit/requirements-convert.txt
# q8_0 is the default (it is what I scored); measure q4_k_m before you ship it, it scored worse for me.
set -e
LLAMA_CPP="${LLAMA_CPP:-./llama.cpp}"
PY="${PY:-python3}"
MODE="${1:?model | lora | fuse-mlx}"; shift
case "$MODE" in
  model)
    IN="${1:?HF dir}"; OUT="${2:?out.gguf}"; T="${3:-q8_0}"
    "$PY" "$LLAMA_CPP/convert_hf_to_gguf.py" "$IN" --outtype "$T" --outfile "$OUT" ;;
  lora)
    AD="${1:?adapter dir}"; BASE="${2:?base HF dir}"; OUT="${3:?out.gguf}"; T="${4:-f16}"
    "$PY" "$LLAMA_CPP/convert_lora_to_gguf.py" --base "$BASE" --outtype "$T" --outfile "$OUT" "$AD" ;;
  fuse-mlx)
    BASE="${1:?base}"; AD="${2:?adapter dir}"; OUT="${3:?out dir}"
    "$PY" -m mlx_lm fuse --model "$BASE" --adapter-path "$AD" --save-path "$OUT" --dequantize ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
echo "done: $MODE -> ${OUT:-}"
echo "next: python3 kit/gguf_clean.py IN OUT --name \"Your Model Name\"; then kit/check_served.py"
