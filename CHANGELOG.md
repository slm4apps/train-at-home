# Changelog

Format: newest first. Each entry says what changed in the guide or kit. A new lesson also names where it was measured.

## [0.2.0] - 2026-11-22

The Tiny inclusion.

### Added
- `kit/train_from_scratch.py`: the from-scratch trainer (own BPE tokenizer with whole format words, pretrain stage, answer-only
  fine-tune stage, pause and resume, time cap, best checkpoints kept). Smoke-tested with `--smoke`.
- `kit/make_scratch_sample.py` and the shipped `kit/data/scratch-sample/`: a made-up dataset for it.
- `GUIDE.md` chapter 7 names the model (Maris3D SLM Blender Tiny 0.02B, with the Hugging Face preview and the article) and has a new
  section 7.6 "Run it".
- `kit/smoke.sh` compiles the new scripts (it already compiled every kit script) and checks `train_from_scratch.py --help`.

## 0.1.0

First version.

- `GUIDE.md`: three hardware paths (8 GB Apple silicon with MLX, an old NVIDIA GPU with CUDA 11.8, a shared CPU server), data,
  install, train (MLX and PyTorch LoRA), measure, ship (merge, GGUF, metadata cleaning, served-answers check), training from
  scratch, the nine mistakes (symptom, cause, fix), the pre-flight checklist, and a table of every number with what it measures.
- `kit/`: sample dataset (made up), dataset check and per-turn converter, row inspector, MLX and PyTorch LoRA trainers, adapter
  merge and blend, checkpoint averaging, GGUF conversion and cleaning, scorer (strict and usable), checkpoint and strength
  scoring, served-answers check, and `smoke.sh` with a dry-run mode.
- Not yet run end to end in this form: the generic kit scripts were checked for syntax and dry runs only.
