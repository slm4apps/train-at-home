# kit/

Generic, runnable scripts for the steps in [`../GUIDE.md`](../GUIDE.md). Standard library only unless noted.

| File | What it does | Needs |
|---|---|---|
| `smoke.sh` | proves a setup works; `--dry-run` trains nothing | shell, Python 3 |
| `make_sample_data.py` | writes the made-up sample dataset (`data/sample/`) | Python 3 |
| `prepare_data.py` | `check` (format, leaks between splits) and `per-turn` (MLX rows) | Python 3 |
| `inspect_rows.py` | decodes rows with the learned span marked | transformers |
| `train_lora_mlx.sh`, `mlx_capped.py` | LoRA on Apple silicon with a memory cap and checks | mlx-lm |
| `train_lora_cuda.py` | LoRA with PyTorch + PEFT; `--check-gpu`, `--dry-run` | torch, peft, transformers |
| `merge_lora.py` | merge a PEFT adapter into its base, optionally scaled | torch, peft |
| `blend_lora.py` | exact blend of two adapters from one run | torch, safetensors |
| `average_checkpoints.py` | weight-average full-model checkpoints (from scratch) | torch, safetensors |
| `convert_gguf.sh` | model, LoRA or MLX-fuse conversion with llama.cpp | llama.cpp checkout |
| `gguf_clean.py` | clean GGUF metadata, scan for leaks | gguf |
| `score.py` | strict and usable scores against a served model | llama-server or any /v1/chat/completions server |
| `score_checkpoints.py` | score every checkpoint and strength, one server at a time | llama-server |
| `check_served.py` | trained model vs served GGUF: same answers? | transformers, llama-server |
| `requirements-*.txt` | versions I used | |

## Status

The scripts are generic versions of the ones I trained with. They were checked with `py_compile`, `--help` and the dry runs in
`smoke.sh`. They were **not** run through a full training in this form, and a few details are not verified here: how your `mlx_lm`
version applies LoRA keys to `embed_tokens` and `lm_head` (look at the trainable-parameter line), the checkpoint file names
your `mlx_lm` writes, and that your `mlx_lm` version reads a `tools` field in chat rows (the per-turn rows carry one). Run `smoke.sh --path ...` first.

## The sample dataset

`data/sample/` is a made-up shopping-list assistant: 120 train, 20 valid and 20 test conversations, each a request, a tool
call, a tool reply and a short answer. `make_sample_data.py` writes it deterministically. It exists to prove the pipeline, not to teach a
useful model.
