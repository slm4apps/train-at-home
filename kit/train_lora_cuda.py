#!/usr/bin/env python3
"""LoRA trainer for PyTorch + PEFT (CUDA, or CPU/MPS for a trial). One file, no framework on top.

    python3 train_lora_cuda.py --base BASE_DIR --rows train.jsonl --out out/run1 --updates 100 --save-every 25
    python3 train_lora_cuda.py --check-gpu            # a real GPU operation, not torch.cuda.is_available()
    python3 train_lora_cuda.py --base ... --rows ... --dry-run   # encode rows, print token/loss-mask stats, stop

What it does:
  * loss only on assistant turns (spans found by rendering message prefixes); an assistant turn with "weight": 0 is
    context only; rows are never cut (a row longer than --max-len is skipped and counted);
  * the output layer runs only on trained positions, in chunks (a big vocabulary makes full logits too large);
  * gradient checkpointing, batch 1, gradient accumulation --accum; ONE "update" = one optimizer step = --accum rows;
  * writes a PEFT adapter every --save-every updates (checkpoint-N/) and at the end (final/), and train_log.jsonl
    with loss, seconds and peak memory per update;
  * the log line at the end states rows seen and passes over the data, so the length of the run is never ambiguous.
"""
import argparse, json, os, random, resource, sys, time


def check_gpu():
    import torch
    print("torch", torch.__version__, "built for CUDA", torch.version.cuda)
    print("is_available() says:", torch.cuda.is_available(), " (this flag alone proves nothing)")
    if not torch.cuda.is_available():
        sys.exit("no CUDA device")
    cap = torch.cuda.get_device_capability(0)
    print("device:", torch.cuda.get_device_name(0), "compute capability sm_%d%d" % cap,
          "memory %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 2**30))
    try:
        a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
        b = (a @ a).float().sum().item()
        torch.cuda.synchronize()
    except Exception as e:  # no kernel for this card in this build
        sys.exit(f"REAL GPU OPERATION FAILED: {e}\nThis build has no kernels for your card. See GUIDE.md, CUDA path.")
    print("real fp16 matrix multiply on the GPU: OK", b == b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check-gpu", action="store_true")
    p.add_argument("--base")
    p.add_argument("--rows")
    p.add_argument("--out")
    p.add_argument("--updates", type=int, default=100)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=160, help="alpha / rank is the scale; 160 / 8 = 20")
    p.add_argument("--first-layer", type=int, default=8, help="LoRA on layers first..last; 0 = all layers")
    p.add_argument("--embed", action="store_true", help="also LoRA on embed_tokens (needed when special tokens matter)")
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=367)
    p.add_argument("--resume", default="", help="adapter dir to continue from")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    p.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.check_gpu:
        return check_gpu()
    for need in ("base", "rows", "out"):
        if not getattr(args, need):
            p.error(f"--{need} is required")

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.base)

    def ids_of(x):
        return x["input_ids"] if hasattr(x, "keys") else x

    def encode(r):
        msgs, tools = r["messages"], r.get("tools")
        full = ids_of(tok.apply_chat_template(msgs, tools=tools, tokenize=True))
        labels = [-100] * len(full)
        for i, m in enumerate(msgs):
            if m["role"] != "assistant" or m.get("weight", 1) == 0:
                continue
            pre = ids_of(tok.apply_chat_template(msgs[:i], tools=tools, tokenize=True, add_generation_prompt=True))
            upto = ids_of(tok.apply_chat_template(msgs[: i + 1], tools=tools, tokenize=True))
            assert full[: len(upto)] == upto, "chat template: a message prefix does not match the whole conversation"
            for j in range(len(pre), len(upto)):
                labels[j] = full[j]
        return full, labels

    with open(args.rows) as f:
        encoded = [encode(json.loads(line)) for line in f if line.strip()]
    rows = [(i, l) for i, l in encoded if len(i) <= args.max_len]
    print(f"rows: {len(encoded)}, usable at max_len {args.max_len}: {len(rows)} (longer rows skipped, never cut)", flush=True)
    assert rows, "no rows fit; raise --max-len"
    n_tok = sum(len(i) for i, _ in rows)
    n_lab = sum(sum(1 for x in l if x != -100) for _, l in rows)
    print(f"tokens: {n_tok}, trained tokens: {n_lab} ({100 * n_lab / n_tok:.1f}%)", flush=True)
    rows_per_pass = len(rows)
    print(f"{args.updates} updates x {args.accum} rows = {args.updates * args.accum} rows seen = "
          f"{args.updates * args.accum / rows_per_pass:.2f} passes over the data", flush=True)
    if args.dry_run:
        return

    from peft import LoraConfig, get_peft_model

    dt = torch.float16 if args.dtype == "fp16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=dt).to(args.device)
    model.config.use_cache = False
    for prm in model.parameters():
        prm.requires_grad_(False)
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if args.embed:
        targets.append("embed_tokens")
    nl = model.config.num_hidden_layers
    cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0, target_modules=targets, bias="none",
                     layers_to_transform=list(range(args.first_layer, nl)) if args.first_layer else None)
    model = get_peft_model(model, cfg)
    if args.resume:
        from safetensors.torch import load_file
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(model, load_file(os.path.join(args.resume, "adapter_model.safetensors")))
        print("resumed from", args.resume, flush=True)
    for prm in model.parameters():
        if prm.requires_grad:
            prm.data = prm.data.float()  # LoRA weights in fp32, base in fp16
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    inner = model.base_model.model
    backbone, lm_head = inner.model, inner.lm_head
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    rng = random.Random(args.seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    pos = 0

    def next_row():
        nonlocal pos
        if pos >= len(order):
            rng.shuffle(order)
            pos = 0
        pos += 1
        return rows[order[pos - 1]]

    def row_loss(ids, lab):
        x = torch.tensor([ids], device=args.device)
        y = torch.tensor(lab[1:], device=args.device)
        h = backbone(input_ids=x).last_hidden_state[0, :-1]
        sel = (y != -100).nonzero(as_tuple=True)[0]
        hs, ys = h[sel], y[sel]
        total = h.new_zeros((), dtype=torch.float32)
        for c in range(0, hs.shape[0], args.chunk):
            logits = lm_head(hs[c: c + args.chunk]).float()
            total = total + F.cross_entropy(logits, ys[c: c + args.chunk], reduction="sum")
        return total, int(sel.numel())

    def peak_mb():
        if args.device == "cuda":
            return round(torch.cuda.max_memory_allocated() / 2**20, 1)
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    log = open(os.path.join(args.out, "train_log.jsonl"), "w")
    model.train()
    t_start = time.time()
    for upd in range(1, args.updates + 1):
        t0 = time.time()
        batch = [next_row() for _ in range(args.accum)]
        ntok = sum(sum(1 for x in l if x != -100) for _, l in batch)
        tot = 0.0
        for ids, lab in batch:
            loss_sum, _ = row_loss(ids, lab)
            (loss_sum / ntok).backward()
            tot += loss_sum.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        rec = {"update": upd, "loss": round(tot / ntok, 5), "sec": round(time.time() - t0, 3), "peak_mb": peak_mb(),
               "tokens": sum(len(i) for i, _ in batch)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if upd <= 3 or upd % 10 == 0:
            print(rec, flush=True)
        if args.save_every and upd % args.save_every == 0:
            model.save_pretrained(os.path.join(args.out, f"checkpoint-{upd}"))
    model.save_pretrained(os.path.join(args.out, "final"))
    seen = args.updates * args.accum
    print(f"done: {args.updates} updates = {seen} rows seen = {seen / rows_per_pass:.2f} passes; "
          f"wall {time.time() - t_start:.0f}s; peak_mb {peak_mb()}", flush=True)


if __name__ == "__main__":
    main()
