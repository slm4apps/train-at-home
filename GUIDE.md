# Train a small language model at home: the guide

This is the whole path, from an empty folder to a model you can serve: pick your machine, build data, train, measure,
ship. Then how to train one from scratch. Then the mistakes that each cost me a run.

If you hand this file to an AI coding assistant, start with the box below. Every command here has a runnable script in
[`kit/`](kit/). Every number is something I measured, with what it measures next to it. Where I did not measure, I say so.

> **For an assistant working from this guide.** Do the chapters in order. Run `kit/smoke.sh --dry-run` first and fix
> what it reports. Do not start a long run until the 20-iteration speed trial (chapter 4) is done and the length is
> written as rows seen and passes. Before every run, do the pre-flight checklist (chapter 9). Do not report a model as
> working until chapter 6 is done: scored on the job, and the converted file checked against the trained model.

Contents: [1 Pick your path](#1-pick-your-path) · [2 Data](#2-data) · [3 Install and first check](#3-install-and-first-check) ·
[4 Train](#4-train) · [5 Measure](#5-measure-the-job-not-the-loss) · [6 Ship](#6-ship) ·
[7 From scratch](#7-training-from-scratch) · [8 The nine mistakes](#8-the-nine-mistakes) ·
[9 Pre-flight checklist](#9-pre-flight-checklist) · [Appendix: numbers](#appendix-every-number-and-what-it-measures)

---

## 1. Pick your path

Each home machine has one hard limit. You find it on day one, or you find it on day ten when something breaks.
Find it on day one.

| Path | Machine | Hard limit | Use it for |
|---|---|---|---|
| **A. MLX** | Apple silicon, 8 GB unified memory | memory shared by the model, optimizer, activations and every open app | LoRA training of a ~0.5B model |
| **B. CUDA** | an old NVIDIA GPU (mine: GTX 980M, compute capability 5.2 / sm_52, 4 GB video memory, 23 GB system RAM) | the software build: only one PyTorch build still has kernels for it | LoRA training of a ~0.5B model, running small local models |
| **C. CPU only** | a shared 4-core server | it is not yours alone, and it is slow | scoring, GGUF conversion, benchmarks. I did not train on it |

You can mix them. I trained on A and B, and scored and converted on C.

### Path A: 8 GB Apple silicon with MLX

MLX is Apple's array framework; `mlx_lm` gives LoRA training on unified memory.

What I measured with a rank-8 LoRA on the last 16 layers plus the embedding and output layers, maximum sequence length
2,048, batch size 1, gradient accumulation 4 and gradient checkpointing: **peak memory 4.8 to 4.9 GB** (the highest run
4.91 GB). That sounds comfortable on 8 GB. It is not, for two reasons.

1. **One model in memory at a time.** Anything else I loaded at the same time ran the training out of memory.
2. **Your open apps matter more than you think.** A memory test of mine was skipped at 28% free memory, because my gate
   was 30%. I closed a browser, two editors, a notes app and a local model server. Free memory went from 31% to 65%, and
   swap from 6.2 GB to 1.4 GB. The test then ran at 69 to 72% free. The laptop was the bottleneck, not the model.
   Before a long run: close everything, check free memory (`memory_pressure`) and swap. It takes a minute.

`kit/train_lora_mlx.sh` does the free-memory check, refuses to start a second job, and caps MLX at 4.5 GB
(`kit/mlx_capped.py`), so a run slows down or fails early instead of pushing the machine into swap.

Speed reference: a 0.5B LoRA run of 5,000 iterations over 2.06 million tokens took 53 minutes at 645 tokens/s of
training, on an Apple M3 with the 4.5 GB cap.

### Path B: an old NVIDIA GPU with CUDA

The trap here is software, not hardware.

- Only the PyTorch build for **CUDA 11.8 (`cu118`)** still shipped kernels for sm_52. Newer builds install fine, and
  `torch.cuda.is_available()` returns `True`, but the first real operation fails because there is no kernel for the card.
  **That flag lies.** Test with a real matrix multiply, which is what `python3 kit/train_lora_cuda.py --check-gpu` does.
- The combination I had installed together on that machine: **torch 2.6.0+cu118**, with transformers 5.22.1 and
  safetensors 0.8.0 in the same virtual environment. (A requirements file of mine pinned other, newer versions, torch 2.11.0
  and transformers 4.57.6; I did not test those on this card.) I did not pin `peft`.
- With 4 GB you live by **layer offload** when you run bigger models: some layers on the GPU, the rest on the CPU. The right
  number is found by crashing. Measured on this card, one 160-token answer each, served with llama.cpp:

| Model | GPU layers | Speed (tokens/s) |
|---|---:|---:|
| Qwen2.5-Coder 1.5B | all | 62.2 |
| my trained 1.5B (one narrow job) | all | 41.8 |
| Qwen3-VL 4B | all | 22.8 |
| Qwen2.5-Coder 7B | 16 | 7.6 |
| Qwen3.5 9B | 10 | 4.4 |
| Qwen2.5-Coder 14B | 10 | 2.5 |
| Devstral 24B | 6 | 1.5 |

The failures matter as much as the speeds. The 7B at 20 layers ran out of GPU memory, and so did the 9B at 13 layers.
They load fine and then crash on the first long prompt. **Write down the layer count that survives a long prompt,
not the one that loads.** Once a model does not fit on the card, speed falls off a cliff (22.8 tokens/s for a 4B that
fits, 4.4 for a 9B at 10 layers). For training small adapters, stay small.

Training reference on this card: a LoRA run of 925 updates of 4 rows each (3,697 rows, one epoch), fp16, gradient
checkpointing, took 9,873.6 s (about 2.7 hours) and peaked at 3,133 MB of video memory.

### Path C: a shared CPU server

Four cores that also run other work. The rules are about being a good neighbour:

- **One scoring run at a time, behind a lock file.** The day two runs start at once, both produce garbage timings.
  A lock is five lines of shell. `kit/score_checkpoints.py` takes `LOCK=/path/lockfile` and refuses to start while it exists.
- **Pin cores** so you do not steal CPU from the real services (`taskset -c 0-3 ...` on Linux).

---

## 2. Data

For a small model the data is most of the work. A model of half a billion parameters has little spare room: it learns
what you show it, and it forgets what you do not.

### Row format

One conversation per line, in the standard chat "messages" format (this is what `kit/prepare_data.py` and both
trainers read). A tool-calling row looks like this (the sample dataset in `kit/data/sample/` is made up for this
repo: a shopping-list assistant):

```json
{"messages": [
  {"role": "user", "content": "put two milk on the shopping list"},
  {"role": "assistant", "content": "", "tool_calls": [{"type": "function",
     "function": {"name": "add_item", "arguments": {"item": "milk", "qty": 2}}}]},
  {"role": "tool", "content": "ok: milk x2 added"},
  {"role": "assistant", "content": "Added 2 milk."}],
 "tools": [{"type": "function", "function": {"name": "add_item", "description": "Add an item to the shopping list.",
   "parameters": {"type": "object", "properties": {"item": {"type": "string"}, "qty": {"type": "integer"}},
   "required": ["item", "qty"]}}}]}
```

A plain question-and-answer row is the same without `tool_calls` and `tools`. An assistant message may carry
`"weight": 0`: it stays in the row as context and is not learned (`train_lora_cuda.py` honours it; `prepare_data.py per-turn`
leaves it out of the learned turns).

### Train only what the model should say

My first released model was trained with no loss mask, so the loss covered the whole conversation, tool replies
included. The model learned to write the replies itself. **In 8 of 30 edit requests it made no tool call at all and typed
`<tool_response> Done: added ...` on its own, as if the tool had run.** Runs with the loss restricted to assistant turns
wrote none.

The rule: the loss covers the model's own turns, and nothing else. Not the system prompt, not the user, not tool replies.

- **PyTorch path:** `kit/train_lora_cuda.py` finds each assistant turn by rendering message prefixes through the chat
  template, and puts the loss only there.
- **MLX path:** `mlx_lm` with `--mask-prompt` learns only the last assistant message of a row. So
  `python3 kit/prepare_data.py per-turn DATA OUT` turns a row with three assistant turns into three rows, each ending in
  the turn to learn, with everything before it as context.
- **Check it with your eyes, every time:** `python3 kit/inspect_rows.py --base BASE --rows train.jsonl --n 3` prints the
  rows decoded back to text with the learned part in `[[[ ]]]`. If a tool reply or a user sentence is inside the brackets, stop.

### Build and check the dataset

1. Write your rows however you like, but **make every answer one you have checked**: run the tool call, compare the result.
   A wrong row is worse than a missing row, because a small model copies it faithfully.
2. Put rows in `train.jsonl` and `valid.jsonl`, and a frozen `test.jsonl` with an `"expect"` field (the tool call a
   correct answer must contain). Hold the test set out **by phrasing and by name**, not only by row: the same request
   reworded still counts as the same request. `prepare_data.py check` fails when a valid or test request also appears in train.
3. `python3 kit/prepare_data.py check DATA_DIR` checks format, counts tool calls, and looks for leaks between splits.
4. Read ten rows yourself.
5. **Replay.** If a new dataset is narrow, mix the older data back in (mistake 4).
6. **One behaviour per adapter** (mistake 8). If you want two, build one mix for both from the start, or train two adapters.
7. If the model must write special tokens (a tool-call tag, your own markers), plan to train the embedding and
   output layers too (mistake 3).

How big? My first released model trained on 2,276 rows; a later adapter on 3,697. A narrow job fits in the low thousands.

---

## 3. Install and first check

Do this once. Use a virtual environment (`python3 -m venv .venv && . .venv/bin/activate`).

### llama.cpp (every path needs it, for conversion and serving)

```sh
git clone https://github.com/ggml-org/llama.cpp
git -C llama.cpp checkout b11899        # the tag I converted with
cmake -S llama.cpp -B llama.cpp/build && cmake --build llama.cpp/build --config Release -j
pip install -r kit/requirements-convert.txt   # gguf==0.17.1, safetensors==0.8.0, numpy
export PATH="$PWD/llama.cpp/build/bin:$PATH"  # llama-server on your PATH
```

(Those are the standard llama.cpp build steps; on a Mac, Metal is on by default. The tag is the one I used on the Mac; a newer
one will probably work, but then the "same answers" check in chapter 6 is the thing that tells you.)

### Path A (MLX)

```sh
pip install -r kit/requirements-mlx.txt       # mlx-lm==0.32.2
```

### Path B (CUDA, old card)

```sh
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r kit/requirements-cuda.txt      # transformers==5.22.1, safetensors==0.8.0, peft
python3 kit/train_lora_cuda.py --check-gpu    # must print "real fp16 matrix multiply on the GPU: OK"
```

If `--check-gpu` fails with `no kernel image`, your torch build does not support your card. Do not debug your own code; change the build.

### Get a base model

Any small instruct model with a chat template works; I used a 0.5B Qwen2.5-Coder. The trainers accept a Hugging Face id (they
download it) or a local folder. llama.cpp conversion needs a local folder, so download one and keep it: this is your
full-precision base, and its hashes go in your notes (mistake 5).

```sh
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir base
sha256sum base/*.safetensors > base.sha256        # on a Mac: shasum -a 256
```

### First check, no training

```sh
kit/smoke.sh --dry-run
```

It writes the sample data, checks it, converts it per turn, compiles every kit script, dry-runs the MLX launcher, and tells
you what is missing on your machine (`llama-server`, `mlx_lm`, `torch`). Fix every `MISS` that your path needs.

Then a real, short run on the sample data:

```sh
kit/smoke.sh --path mlx  --base Qwen/Qwen2.5-0.5B-Instruct   # 20 iterations
kit/smoke.sh --path cuda --base Qwen/Qwen2.5-0.5B-Instruct   # GPU check + 10 updates
```


---

## 4. Train

### Counting honestly

The word "step" means different things in different tools, and your training length depends on it.

- In `mlx_lm` 0.32.2, with batch size 1, **one iteration was one row of data**. Not a batch, not an epoch. My first released
  model trained for 2,600 iterations on 2,276 rows: 1.14 passes over the data, barely more than one epoch. If I had said
  "2,600 steps" and a reader assumed batches of 8, they would have believed I trained for nine epochs.
- In my PyTorch trainer, one update is one optimizer step over `--accum` rows. 925 updates of 4 rows is 3,700 row-visits.
  Same word, different meaning.

So always write the length as **rows seen, and passes over the data**. Both trainers print it. A step count without its
unit is like quoting a speed without units.

### The speed trial (do this before every long run)

Run 20 iterations on the real data, read the speed, and work out the time for the real run **from that, not from a
token count.** My own two reference points show why:

- A short-row run on the Mac averaged **709 tokens per second**: 1,800 iterations in 57 minutes, the mean of 180 progress
  reports. That became my smoke-test number: if a 20-iteration trial is far from your own reference, recompute the
  time before going on.
- A heavier run on 15,270 rows (21.7 million tokens) ran at **3.2 seconds per iteration**, so **one pass took 13.5 hours.**
  My first estimate, from the 709 tokens/s and the token count, was about 8.6 hours.

Short rows go fast, long rows go slow. When an estimate and a measurement on the real job disagree, believe the measurement.

### Path A: MLX LoRA

```sh
python3 kit/prepare_data.py per-turn kit/data/sample out/data-turns          # loss on the last assistant message only
python3 kit/inspect_rows.py --base Qwen/Qwen2.5-0.5B-Instruct --rows kit/data/sample/train.jsonl --n 3
kit/train_lora_mlx.sh Qwen/Qwen2.5-0.5B-Instruct out/data-turns out/run1 600  # EMBED=1 for special tokens
```

What the launcher runs, all in one place: `mlx_lm lora --train --batch-size 1 --grad-accumulation-steps 4 --num-layers 16
--learning-rate 1e-4 --max-seq-length 2048 --grad-checkpoint --mask-prompt --save-every 100`, with a YAML file giving rank 8,
scale 20 (the same as alpha 160 at rank 8) and the keys to adapt: the seven linear modules of each block
(`q,k,v,o,gate,up,down`). `EMBED=1` adds `embed_tokens` and `lm_head`. That is what I trained with for my first model. I have
not re-verified in this kit how your `mlx_lm` version applies keys to the embedding and output layers, so look at the
trainable-parameter line the trainer prints at the start. It should go up when you add them.

Adapters land in `out/run1/adapters/` (a checkpoint every 100 iterations).

### Path B: PyTorch LoRA (CUDA)

```sh
python3 kit/train_lora_cuda.py --base BASE_HF_DIR --rows kit/data/sample/train.jsonl --out out/run1 \
    --updates 120 --save-every 30 --accum 4 --lr 1e-4 --rank 8 --alpha 160 --first-layer 8 --device cuda --dtype fp16
```

The settings I used for a ~0.5B model: PEFT LoRA rank 8, alpha 160 (effective scale 20), all seven linear modules, layers 8 to
23 only (`--first-layer 8`), learning rate 1e-4, batch 1 with accumulation 4, fp16, gradient checkpointing, seed 367,
**updates = exactly one epoch** (3,697 rows, 925 updates of 4). Add `--embed` for special tokens. `--dry-run` encodes your
rows and prints tokens, trained-token share and the length in rows and passes without loading the model. It runs on a
CPU-only machine.

`--resume ADAPTER_DIR` continues from an adapter. Checkpoints are `out/run1/checkpoint-N/` and `out/run1/final/`.
`train_log.jsonl` has loss, seconds and peak memory per update.

### What can go wrong while it runs

- **First validation loss looks too high, or the model answers with a stray `!` after 20 iterations.** See mistake 2.
- **Out of memory:** close everything, then lower `--max-seq-length` / `--max-len`. A second model in memory is the usual cause.
- **Loss falls, answers are junk:** you probably trained the wrong span. Run `inspect_rows.py`.

---

## 5. Measure: the job, not the loss

The loss tells you the model fits the training text. It does not tell you the model does the job. **Score every
checkpoint on the job**, on the same frozen test file.

`kit/score.py` asks each test request of a served model at temperature 0 and reads the tool call. It prints **two numbers
together**, plus the failures that are not the model's:

| Number | Means |
|---|---|
| **strict** | the tool name and every argument equal the expected call |
| **usable** | a call to a known tool, arguments valid JSON, required keys present: a host could run it |
| no_call | plain text instead of a call, or the model wrote a tool reply itself |
| infra | HTTP errors, "loading model" 503s, timeouts, after 3 tries: **not counted against the model** |

Why two numbers: in one run of mine, between two checkpoints PASS rose from 11 to 14 while the usable rate fell from 50% to
34%. Tracking PASS alone I would have called the later checkpoint better. It was worse for every real user. When the two
move apart, stop and read the outputs.

Why separate infrastructure failures: in one scoring run, five rows failed because the llama server answered
`HTTP 503: Loading model` mid-run (zero calls, 0.1 seconds). A naive scorer counts those as wrong answers. Some generated
scripts also tried to allocate more than 32 GB; I wrapped each run in a 60-second wall-clock limit that records real
timeouts as failures with a reason. Before you compare two models, look at the failures and ask: did the model fail,
or did the harness?

### Score every checkpoint and several strengths

Convert each checkpoint to a LoRA GGUF, then:

```sh
kit/convert_gguf.sh lora out/run1/checkpoint-60  BASE_HF_DIR out/ck60.gguf
kit/convert_gguf.sh lora out/run1/final          BASE_HF_DIR out/final.gguf
python3 kit/score_checkpoints.py --base out/base-q8_0.gguf --test kit/data/sample/test.jsonl \
    --adapters out/ck60.gguf out/final.gguf --scales 0.7 1.0 1.3
```

You get one row per (checkpoint, strength), with the base alone as the first row. (`out/base-q8_0.gguf` is the base
model converted with `kit/convert_gguf.sh model BASE_HF_DIR out/base-q8_0.gguf q8_0`.)

**MLX path:** `mlx_lm` adapters are not PEFT adapters, so score each checkpoint as a whole model instead. `mlx_lm` saves
checkpoints in the adapter folder as `NNNNNNN_adapters.safetensors`; to fuse one, copy it into a new folder as
`adapters.safetensors` next to `adapter_config.json` (check the file names your version writes), run
`kit/convert_gguf.sh fuse-mlx BASE that-folder out/fused-N`, convert with `convert_gguf.sh model`, and give all the files to
`score_checkpoints.py --base out/ck-100.gguf out/ck-200.gguf ...` (with no `--adapters`, every base is scored alone).

**Pick by that table**, not by the loss. On my from-scratch model the held-out loss bottomed out at step 5,670 and then rose,
but the task scores did not fall (chapter 7).

**Adapter strength is a dial.** The same edit adapter, merged into the weights at strength 1.0, gave the best usable rate
(77.2%, 26 of 30 edit answers usable) and a lower full-pass rate (19.4%). Loaded on the base model at strength 0.7, it gave
the best full-pass rate (23.9%) and 24 of 30 edit answers, with a lower usable rate (68.9%). Same file, two different products
(180-request benchmark, one run each). Merging is one point on a curve; I measured two strengths and they already
disagreed. Measure at least three.

---

## 6. Ship

### 6.1 Merge (or keep the adapter separate)

You can ship an adapter as its own small file and load it at serve time, or merge it into the base.

```sh
python3 kit/merge_lora.py --base BASE_HF_DIR --adapter out/run1/final --out out/merged --scale 1.0   # PEFT adapters
kit/convert_gguf.sh fuse-mlx BASE_HF_DIR out/run1/adapters out/merged                               # MLX adapters
```

**Merge into the full-precision base, never into something rebuilt from a GGUF** (mistake 5). Pick `--scale` from your
strength table. To blend two checkpoints of one run into one adapter (I shipped a 30/70 blend of two checkpoints):

```sh
python3 kit/blend_lora.py out/run1/checkpoint-60 out/run1/checkpoint-90 0.3 out/blend
```

It stacks the two adapters' ranks, so the result is exact, with rank doubled, and it prints the error against the weighted sum.
Score the blend like any other checkpoint.

### 6.2 Convert to GGUF

```sh
kit/convert_gguf.sh model out/merged out/model-q8_0.gguf q8_0
```

**q8_0 is the default.** It is what I scored. I measured q4_k_m once: on the 180-request benchmark it scored 18.3% pass and
67.2% usable, against 19.4% and 77.2% for the merged q8_0 file. Never ship a smaller quantization without scoring it.

### 6.3 Clean the metadata

Converters write what they find on the converting machine: the model's name guessed from your training folder, tags and
base-model entries made from a local path, and the base model's stock chat template with its stock identity line. Before a file
goes to anyone:

```sh
python3 kit/gguf_clean.py out/model-q8_0.gguf --scan                                  # list what would leak
python3 kit/gguf_clean.py out/model-q8_0.gguf out/model-release.gguf --name "My Model 0.5B"
```

The copy keeps every tensor and every other key, and the script reads the copy back and deletes it if a string still looks
like a home path or a private address. Then run chapter 6.4 on the **cleaned** file: cleaning must not change answers. (I
checked 15 of 15 rows identical before and after cleaning on one release.)

### 6.4 Check that what you serve is what you trained

```sh
python3 kit/check_served.py reference --model out/merged --test kit/data/sample/test.jsonl --out out/reference.jsonl
python3 kit/check_served.py compare   --gguf out/model-release.gguf --test kit/data/sample/test.jsonl --reference out/reference.jsonl
```

The first step runs the trained (unconverted) model greedily. The second serves the converted file in a **stock**
`llama-server` with the chat template the file carries, and compares. It exits 0 only when every row matches. On my from-scratch
model, a stock server answered exactly as the trained model on 40 of 40 rows, and the same check had earlier caught one newline
that llama.cpp's template engine dropped after a block tag. Without the check that would have shipped.

### 6.5 Serve

```sh
llama-server -m out/model-release.gguf --jinja -c 4096 --temp 0
# an adapter kept separate, at a chosen strength:
llama-server -m base-q8_0.gguf --jinja -c 4096 --temp 0 --lora-scaled adapter.gguf 0.7
```

What worked for tool-calling models: `--jinja` (so the file's own chat template is used), temperature 0, a context of 4,096,
and **repeat penalty left at 1.0**: penalties suppress the tool-call tag that has to repeat on every call. Check the chat endpoint
with `python3 kit/score.py --url http://127.0.0.1:8080 --test kit/data/sample/test.jsonl`. With a separate adapter,
`llama-server` also exposes `/lora-adapters`, where you can read the adapter ids and set each one's scale at run time.

### 6.6 What to keep

Keep the full-precision base, every file you released, every checkpoint another run started from, and every parent of a
blended adapter. Everything else is a temporary copy. Do not delete a checkpoint before you have recorded how to rebuild
whatever came from it (once I deleted the parents of a blend; only their converted copies survived). Record sha256 of every
released file next to the recipe that built it.

---

## 7. Training from scratch

Everything above adapts a model that already exists. I also trained one from nothing: a 24.5-million-parameter model, no
pretrained weights. A Llama-shaped model (10 layers, width 384, 6 heads, context 2,048), made in two stages. That is small
enough to train on a laptop, and I learned four things only by doing it. `kit/average_checkpoints.py` covers the last of them.

The model is **Maris3D SLM Blender Tiny 0.02B** (24.5M parameters). The preview is on Hugging Face:
<https://huggingface.co/slm4apps/Maris3D-Blender/tree/tiny-0.02b-preview>. The story is in the article
[How small can a Blender model go?](https://www.linkedin.com/pulse/how-small-can-blender-model-go-maris3d-tiny-002b-zino-adidi-wt0xf/).
`kit/train_from_scratch.py` is the script it was trained with.

### 7.1 Size the tokenizer before the model

The embedding table is vocabulary size times model width. At width 384:

| Tokenizer | Vocabulary | Embedding parameters |
|---|---:|---:|
| a 151k-token tokenizer (the Qwen2.5 family's) | 151,665 | 58 million |
| Llama 3's | 128,256 | 49 million |
| GPT-2's | 50,257 | 19 million |
| **my own** | **17,665** | **6.8 million** |

A standard tokenizer would have cost more than twice the whole model. So I trained a byte-level BPE of 16,384 ordinary tokens and added
1,281 whole words for the task's own format (sizes, relations, colours, finishes): one token each. Answers get shorter, more of them
fit in the context, and a format word cannot be misspelt. Check that every format word really is one token (the trainer
I used logged any that were not).

**The tokenizer has a price.** llama.cpp's converter did not know it; the fix was one line registering it as the type its
pre-tokenizer is (`gpt-2`). And a model with its own tokens cannot share a token-level training signal with any other model.

### 7.2 Two stages

1. **Pretrain** on plain text so it learns the language: for me 103 million tokens (a small encyclopedia in simple English
   plus the task data's text), 3 passes.
2. **Task stage:** the full task mix, request to answer, loss on the answer only (same rule as chapter 2).

From scratch the data is the whole model, so every row in the mix needs a reason. Credit and licence the text you pretrain on, and
keep the text out of your repo; the weights can carry their own licence.

### 7.3 Held-out loss is not the pick rule

Loss bottomed out at step 5,670, then rose, but the task scores did not fall. The last save, at held-out loss about 0.20,
still scored 53.3 on the "parts all right" set and 97.1 on a held-out set of the app's own requests. An equal blend of the
5,670 and 7,560 saves, at held-out loss 0.15, scored 52.7 and 96.1. (The 5,670 save alone, at loss 0.162, scored 49.3 and 90.7.)
So pick by the scores of the thing the model is for.

### 7.4 Average checkpoints

Averaging the weights of two or three checkpoints of one run scored at or near the best of them, without choosing one by loss.
My pick was an equal blend of three saves (steps 5,670, 7,560 and 15,192): on most of the job's test sets it beat the best single
save (for example 58.7 on the edits set against 57.8 for the last save alone, and 97.6 on the app's own requests against 97.1).

```sh
python3 kit/average_checkpoints.py out/blend out/ck-5670:1 out/ck-7560:1 out/ck-15192:1
```

Score the average like any checkpoint (chapter 5).

### 7.5 Test the thing the model cannot do early

My from-scratch model could not copy exact decimals: stray format-word tokens leaked into digit strings (it wrote things like
`0.Hand` inside a number). It was the same on the unconverted model and on the GGUF, so it was the model, not the
conversion. I had first suspected the converter, and the greedy run on the unconverted checkpoint is what settled it. The cheap
test that finds this is a handful of prompts with exact numbers in them, at temperature 0. Do it before the long run, and decide
whether your application reads exact values with code instead of asking the model.

The adapter lessons of chapter 8 still apply, especially: score on the job.

### 7.6 Run it

Make the made-up sample (a shopping-list assistant, no outside text) and check the whole path in about a minute on the CPU:

```sh
python3 kit/make_scratch_sample.py                      # writes kit/data/scratch-sample/
python3 kit/train_from_scratch.py --data kit/data/scratch-sample --out out/scratch-smoke --device cpu --smoke
```

`--smoke` trains a toy-sized model for 30 steps per stage. The log ends with the event `"done"` when it worked. It proves the
path, nothing more. For the real run, bring your own data in the same layout (`pretrain.txt`, `pretrain_val.txt`,
`sft_train.jsonl`, `sft_val.jsonl`, `format_words.json`) and leave the sizes at their defaults, which are this chapter's model:

```sh
python3 kit/train_from_scratch.py --data YOUR_DATA --out out/scratch --device cuda
```

It writes the tokenizer and the stage 1 model into `out/scratch`, and keeps the best few fine-tune checkpoints there as
`sft-ck-<step>` (by held-out loss; pick by task scores, 7.3). To pause, touch `out/scratch/PAUSE`; the next start resumes.
Then average and convert:

```sh
python3 kit/average_checkpoints.py out/blend out/scratch/sft-ck-A:1 out/scratch/sft-ck-B:1 out/scratch/sft-ck-C:1
kit/convert_gguf.sh model out/blend out/tiny-q8_0.gguf q8_0
```

Before the conversion, register the tokenizer's pre-tokenizer as `gpt-2` in your llama.cpp checkout (7.1), or the converter will
not know it. Replace A, B and C with the step numbers of the checkpoints you picked.

---

## 8. The nine mistakes

Each cost me a run. Symptom, cause, fix, and where it was measured.

### 1. Training on tool results teaches the model to fake them
- **Symptom:** in 8 of 30 edit requests the model made no tool call and typed `<tool_response> Done: added ...` itself.
- **Cause:** the first model was trained with no loss mask, so the tool replies were part of what it learned to write.
- **Fix:** loss only on the model's own turns; decode three rows with the learned span marked before you start (`inspect_rows.py`).
- **Measured:** 30 edit requests, first released model against runs with assistant-only loss, which wrote none.

### 2. Shorter rows can wake up a hidden trainer bug
- **Symptom:** within 20 iterations the model broke its tool calls and answered with `!`. The starting validation loss was 0.490.
- **Cause:** `mlx_lm` 0.32.2 trains one padding target per row: after the last real token, the next target is the padding id, which
  here is the character `!`. On long unmasked rows that is about 1 target in 1,000, and nobody notices. On short per-turn masked rows it
  was 1 in 30 to 300.
- **Fix:** I patched the trainer; with the patch, the starting validation loss was 0.082. When the shape of your data changes (shorter rows,
  masks), re-read the framework's loss code, and after 20 iterations generate a few answers and look for a stray pad character.
- **Measured:** starting validation loss 0.490 with the bug, 0.082 without, on the same rows. Check whether your `mlx_lm` version still does this.

### 3. Special tokens need their own layers
- **Symptom:** a LoRA wrote correct JSON for a tool call with no tag around it, so a stock host called nothing.
- **Cause:** the tag tokens exist in the vocabulary but were never trained, so a LoRA on the linear layers alone does not learn to emit them.
- **Fix:** put `embed_tokens` and `lm_head` in the LoRA (`EMBED=1` / `--embed`), then merge. If a model "almost" works and the format is
  what is wrong, look at which layers you trained.

### 4. Narrow data overwrites
- **Symptom:** one adapter trained on 135 single-object rows got worse at everything, single objects included: pass rate fell from 20.6%
  to 9.4% on the project's benchmark.
- **Cause:** a small adapter has little room; narrow data with no replay overwrites what was there.
- **Fix:** mix older rows back in. My edit adapter used 1,421 new edit sessions plus the 2,276 original rows replayed, and did not narrow.

### 5. Never rebuild a base from a quantized file
- **Symptom:** the same 9.4% run: a drop I could not explain.
- **Cause:** I had rebuilt the base from the q8_0 GGUF, while the full-precision weights were sitting on a server all along. A quantized file is a lossy
  copy; training on a rebuild stacks error on error, and you cannot tell which part caused the drop.
- **Fix:** find the full-precision weights and prove the chain with file hashes before you train. A GGUF is for serving, not for building.

### 6. Not every failure is the model's
- **Symptom:** five failed rows in a scoring run with zero calls and 0.1-second answers; generated scripts allocating over 32 GB.
- **Cause:** the llama server answered `HTTP 503: Loading model` mid-run; a naive scorer counts that as a wrong answer.
- **Fix:** `score.py` retries and counts infrastructure failures apart; run anything the model produces under a wall-clock limit (I used 60
  seconds), record real timeouts as failures with a reason.

### 7. PASS went up, usable went down
- **Symptom:** between two checkpoints, PASS rose from 11 to 14 and the usable rate fell from 50% to 34%, with more unparseable output.
- **Cause:** one number hid what the other showed.
- **Fix:** always track a strict and a loose number side by side; when they diverge, read the outputs.

### 8. One behaviour per adapter
- **Symptom:** three attempts to teach two behaviours in one adapter failed, in a new way each time. A mixed edit-plus-build adapter halved
  single-object passes. An edit-plus-steering adapter fixed the multi-turn stall it was trained for (answering through 3.19 of 4 turns
  instead of stopping after 1.14) but, at full strength, edit answers that worked fell from about 30% to 2%, and the usable rate from 77% to 49%.
- **Cause:** the second behaviour competed with the first for the same small capacity.
- **Fix:** give an adapter one job. For two behaviours, train two adapters and choose at serve time, or sweep how strongly each is applied. What
  worked later was one adapter trained on **one mix built for all its jobs from the start**, not a second behaviour added on top of a trained one.

### 9. Adapter strength is a dial you have not measured
- **Symptom:** the same file gave two different best results (chapter 5): best usable at strength 1.0 merged, best full pass at 0.7 loaded on the base.
- **Cause:** merging is one point on a curve.
- **Fix:** measure at least three strengths before you ship, and say which is which. I shipped both options.

---

## 9. Pre-flight checklist

Before any long job on a home machine.

**Before the run**
- [ ] Close every other app. Free memory above 30%, swap low. (`memory_pressure` on a Mac.)
- [ ] One model in memory at a time.
- [ ] Old NVIDIA card: the cu118 PyTorch build, tested with `train_lora_cuda.py --check-gpu`, a real operation and not `is_available()`.
- [ ] GPU layer count set to one that survived a long prompt, not just a load.
- [ ] Shared server: take the lock, pin the cores.
- [ ] The base is full precision, with hashes recorded. Never a GGUF.

**Data and loss**
- [ ] `prepare_data.py check` passes (no train/test leak). I read ten rows myself.
- [ ] Loss masked to assistant turns. `inspect_rows.py`: three rows decoded, learned span looked at.
- [ ] After changing the shape of the rows, the framework's padding target re-checked.
- [ ] Tags or special tokens matter: `embed_tokens` and `lm_head` in the LoRA.
- [ ] Narrow new data comes with replay of older data.
- [ ] One behaviour per adapter.

**Time and counting**
- [ ] Length written as rows seen and passes over the data, not "steps".
- [ ] A 20-iteration speed trial on the real data; time worked out from it, not from a token count.

**Scoring**
- [ ] Infrastructure failures counted apart from model failures, and re-run.
- [ ] A strict and a usable number side by side.
- [ ] The same frozen test file every time.
- [ ] At least three adapter strengths measured before shipping.

**Shipping**
- [ ] q8_0 unless a smaller file was scored.
- [ ] GGUF metadata cleaned (`gguf_clean.py --scan` is clean).
- [ ] `check_served.py compare` passes on the file you ship, after cleaning.
- [ ] sha256 of every released file recorded with its recipe; the base and every parent checkpoint kept.

Home hardware makes you slow, so you cannot afford to be wrong. Check every assumption with one cheap test before a 13-hour run,
read the outputs and not just the scores, and a laptop will get real work done.

---

## Appendix: every number and what it measures

| Number | What it measures |
|---|---|
| 4.8 to 4.9 GB (4.91 max) | peak memory of the MLX LoRA recipe in chapter 1, path A |
| free memory 31% to 65%, swap 6.2 GB to 1.4 GB | the same 8 GB Mac before and after closing five apps |
| 645 tokens/s, 53 min | MLX LoRA, 0.5B, 5,000 iterations, 2.06 million tokens, M3, 4.5 GB cap |
| 709 tokens/s | MLX LoRA, 1,800 iterations in 57 min, mean over 180 progress reports (short rows) |
| 3.2 s/iteration, 13.5 h | MLX LoRA, 15,270 rows, 21.7 million tokens, one pass (long rows) |
| 9,873.6 s, 3,133 MB | PyTorch LoRA, 925 updates x 4 rows, 4 GB card |
| tokens/s table (62.2 ... 1.5) | llama.cpp serving on the 4 GB card, one 160-token answer each |
| 2,600 iterations = 1.14 passes | MLX, 2,276 rows, batch 1 |
| 8 of 30 | edit requests where the unmasked first model faked a tool reply |
| val loss 0.490 vs 0.082 | starting validation loss with and without the padding bug, same rows |
| pass 20.6% to 9.4% | the project's benchmark, after an adapter trained on 135 narrow rows |
| PASS 11 to 14, usable 50% to 34% | two checkpoints of one run |
| usable 77.2% (26/30 edits), pass 19.4% | edit adapter merged at strength 1.0, 180-request benchmark |
| usable 68.9% (24/30 edits), pass 23.9% | same adapter at strength 0.7 on the base |
| q4_k_m 18.3% pass / 67.2% usable | 180-request benchmark, one run |
| 24.5 M parameters; 17,665 tokens = 6.8 M embedding | the from-scratch model and its tokenizer |
| 58 M / 49 M / 19 M | embedding cost at width 384 of the 151k / Llama 3 / GPT-2 tokenizers |
| loss 0.15 / 0.20; 52.7 / 53.3; 96.1 / 97.1 | blend of two early saves vs the last save: held-out loss, then two job test sets |
| 40 of 40 | converted file in a stock llama-server vs the trained model, identical answers |
