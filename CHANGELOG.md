# Changelog

Format: newest first. Each entry says what changed in the guide or kit. A new lesson also names where it was measured.

## 0.1.0

First version.

- `GUIDE.md`: three hardware paths (8 GB Apple silicon with MLX, an old NVIDIA GPU with CUDA 11.8, a shared CPU server), data,
  install, train (MLX and PyTorch LoRA), measure, ship (merge, GGUF, metadata cleaning, served-answers check), training from
  scratch, the nine mistakes (symptom, cause, fix), the pre-flight checklist, and a table of every number with what it measures.
- `kit/`: sample dataset (made up), dataset check and per-turn converter, row inspector, MLX and PyTorch LoRA trainers, adapter
  merge and blend, checkpoint averaging, GGUF conversion and cleaning, scorer (strict and usable), checkpoint and strength
  scoring, served-answers check, and `smoke.sh` with a dry-run mode.
- Not yet run end to end in this form: the generic kit scripts were checked for syntax and dry runs only.
