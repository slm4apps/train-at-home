# Train at home

How to train small language models on home hardware, written by someone who did it on an 8 GB MacBook, a 2014 gaming
laptop with a GTX 980M, and a slice of a shared 4-core server. No cloud GPU, no cluster.

The guide is complete enough to hand to an AI coding assistant and end up with a trained, converted and served model.
It is kept up to date: every new lesson is added with its symptom, cause, fix and where it was measured.

**Who it is for:** you have a laptop or an old gaming PC and want to fine-tune (or train from scratch) a model of up to about
half a billion parameters, and you would rather not lose days to mistakes someone already made.

## Three hardware paths

| Path | Machine | Use |
|---|---|---|
| **A. MLX** | Apple silicon, 8 GB | LoRA training of a ~0.5B model; peak memory of my recipe was 4.8 to 4.9 GB |
| **B. CUDA** | an old NVIDIA GPU (a GTX 980M, 4 GB) | LoRA training; the only PyTorch build with kernels for it is CUDA 11.8 (`torch 2.6.0+cu118`) |
| **C. CPU only** | a shared 4-core server | scoring, GGUF conversion, benchmarks |

## Quick start

```sh
git clone <this repo> && cd train-at-home
kit/smoke.sh --dry-run                                  # 1. checks the kit and your tools, trains nothing
pip install -r kit/requirements-mlx.txt                 # 2. path A (path B: see GUIDE.md chapter 3)
kit/smoke.sh --path mlx --base Qwen/Qwen2.5-0.5B-Instruct   # 3. 20 real iterations on the sample data
python3 kit/prepare_data.py check YOUR_DATA             # 4. your own rows, same format as kit/data/sample
kit/train_lora_mlx.sh BASE YOUR_DATA out/run1 600       # 5. train (path B: kit/train_lora_cuda.py)
python3 kit/score_checkpoints.py --base base.gguf --test YOUR_DATA/test.jsonl --adapters out/*.gguf --scales 0.7 1.0
python3 kit/merge_lora.py --base BASE --adapter out/run1/final --out out/merged    # 6. ship
kit/convert_gguf.sh model out/merged out/model-q8_0.gguf q8_0
python3 kit/check_served.py compare --gguf out/model-q8_0.gguf --test YOUR_DATA/test.jsonl --reference out/reference.jsonl
```

Read [GUIDE.md](GUIDE.md) before step 5. It explains each step, what to measure, and the nine mistakes that each cost a run.

## What is here

- [`GUIDE.md`](GUIDE.md): pick your path, data, install, train, measure, ship, train from scratch, the nine mistakes, the pre-flight checklist.
- [`kit/`](kit/): runnable scripts and a tiny made-up sample dataset (see [`kit/README.md`](kit/README.md)).
- [`CHANGELOG.md`](CHANGELOG.md), [`CONTRIBUTING.md`](CONTRIBUTING.md): how it changes, and how to add a lesson.

## Honest limits

- Every number in the guide is something I measured, with what it measures next to it. The kit scripts were written from
  the scripts I trained with, but the generic versions here were checked only for syntax and dry runs; run
  `kit/smoke.sh` on your machine first, and tell me what breaks (see CONTRIBUTING).
- The guide covers models up to about 0.5B parameters (LoRA) and one 24.5M-parameter model from scratch. It says nothing about larger models.

Licence: Apache-2.0 (see [LICENSE](LICENSE)).

Zino
