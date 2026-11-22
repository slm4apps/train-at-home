#!/usr/bin/env python3
"""Train a small Llama-shaped model from scratch, with its own tokenizer (guide chapter 7). llama.cpp can convert the
result to GGUF (see convert_gguf.sh; register the tokenizer's pre-tokenizer as gpt-2, chapter 7.1).

Stage 1 pretrains on plain text, stage 2 fine-tunes on request -> answer rows with the loss on the answer only.
What it does for you:
  - builds a byte-level BPE tokenizer from pretrain.txt and adds every format word that contains "_" as one token
    (byte-level BPE would split it); logs any format word that is still more than one token;
  - drops (and counts) a fine-tune row longer than the context instead of cutting it;
  - keeps the best few fine-tune checkpoints by validation loss (sft-ck-<step>); pick among them by task scores,
    not by loss (chapter 7.3), and try averaging them (average_checkpoints.py, chapter 7.4);
  - pause: touch OUT/PAUSE or send SIGTERM; the next start resumes exactly where it stopped;
  - a time cap per stage: after 100 steps it measures the speed and shortens the run to fit;
  - a finished stage 1 is reused when the script runs again.

DATA holds pretrain.txt and pretrain_val.txt (documents separated by a blank line), sft_train.jsonl and sft_val.jsonl
(one {"job", "request", "answer"} per line; "job" is a short tag for the kind of task) and format_words.json (a list of
the answer format's own words). make_scratch_sample.py writes a made-up example.

    python train_from_scratch.py --data DATA --out OUT [--device cuda|mps|cpu] [--smoke]

--smoke trains a toy-sized model for 30 steps per stage: it checks the whole path in a minute, nothing more.
Defaults are the 24.5M-parameter model of chapter 7 (10 layers, width 384, 6 heads, context 2,048).
Needs: torch, transformers, tokenizers, numpy.
"""
import argparse, json, math, os, random, shutil, time

import numpy as np
import torch
from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

SPECIAL = ["<pad>", "<s>", "</s>", "<job>", "<req>", "<ans>"]
FORMAT_WORDS = []


def log(f, **kw):
    kw["t"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = json.dumps(kw)
    print(line, flush=True)
    f.write(line + "\n"); f.flush()


def build_tokenizer(data, out, vocab):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=vocab, special_tokens=SPECIAL,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train([os.path.join(data, "pretrain.txt")], tr)
    # byte-level BPE splits at "_", so salt_grain / on_top_of could never be one token: add them whole (only the
    # words with "_", which cannot be inside an ordinary English word)
    tok.add_tokens([AddedToken(w, normalized=False) for w in sorted(set(FORMAT_WORDS)) if "_" in w])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", eos_token="</s>", pad_token="<pad>",
                                   additional_special_tokens=["<job>", "<req>", "<ans>"])
    fast.save_pretrained(out)
    return fast


def prompt_text(job, request):
    return f"<s><job> {job}\n<req> {request}\n<ans>\n"


# Chat template = the same prompt, so llama-server / Ollama / transformers.js chat calls (and coding agents through an
# OpenAI-style API) ask exactly what was trained. BOS is in the template, so the runtimes must not add one.
# The system message is the job tag: parts, outline or edit.
CHAT_TEMPLATE = ("{% for m in messages %}{% if m['role'] == 'system' %}<s><job> {{ m['content'] }}\n"
                 "{% elif m['role'] == 'user' %}<req> {{ m['content'] }}\n<ans>\n"
                 "{% elif m['role'] == 'assistant' %}{{ m['content'] }}</s>{% endif %}{% endfor %}")


def save_tok(tok, d):
    tok.save_pretrained(d)
    p = os.path.join(d, "tokenizer_config.json")
    c = json.load(open(p))
    c.update(chat_template=CHAT_TEMPLATE, add_bos_token=False, add_eos_token=False)
    json.dump(c, open(p, "w"), indent=1)


def _docs(path):
    buf = []
    with open(path) as f:
        for line in f:
            if line == "\n":
                if buf:
                    yield "".join(buf).rstrip("\n"); buf = []
            else:
                buf.append(line)
    if buf:
        yield "".join(buf).rstrip("\n")


def blocks_of(tok, path, block):
    import array
    ids, batch = array.array("i"), []

    def flush():
        for x in tok(batch, add_special_tokens=False)["input_ids"]:
            ids.append(tok.bos_token_id); ids.extend(x); ids.append(tok.eos_token_id)
        batch.clear()
    for d in _docs(path):
        if d.strip():
            batch.append(d)
            if len(batch) >= 2000:
                flush()
    if batch:
        flush()
    n = len(ids) // block
    t = torch.from_numpy(np.frombuffer(ids, dtype=np.int32)[: n * block].copy()).view(n, block)
    return t, len(ids)


def sft_rows(tok, path, ctx):
    """[(int32 ids, prompt length)]; rows over ctx tokens are dropped and counted."""
    out, dropped, rs = [], 0, [json.loads(l) for l in open(path)]
    for k in range(0, len(rs), 1000):
        part = rs[k: k + 1000]
        P = tok([prompt_text(r["job"], r["request"]) for r in part], add_special_tokens=False)["input_ids"]
        A = tok([r["answer"] + "</s>" for r in part], add_special_tokens=False)["input_ids"]
        for pi, ai in zip(P, A):
            if len(pi) + len(ai) > ctx:
                dropped += 1; continue
            out.append((np.array(pi + ai, dtype=np.int32), len(pi)))
    return out, dropped


PAD_TO = 128   # lengths are padded up to a multiple of 128, so only a few batch shapes ever occur: on Apple silicon
               # (MPS) every new shape keeps more memory, and an 8 GB machine runs out in stage 2


def pad_batch(rows, pad):
    m = max(len(i) for i, _ in rows)
    m = -(-m // PAD_TO) * PAD_TO
    x = torch.full((len(rows), m), pad); y = torch.full((len(rows), m), -100)
    a = torch.zeros((len(rows), m), dtype=torch.long)
    for k, (i, pl) in enumerate(rows):
        t = torch.from_numpy(i.astype(np.int64))
        x[k, : len(i)] = t; y[k, pl: len(i)] = t[pl:]; a[k, : len(i)] = 1
    return x, y, a


def lr_at(step, total, peak, warm):
    if step < warm:
        return peak * (step + 1) / warm
    return peak * 0.1 + 0.9 * peak * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))


@torch.no_grad()
def val_loss(model, batches, dev):
    model.eval(); tot = n = 0
    for x, y, a in batches:
        tot += model(input_ids=x.to(dev), labels=y.to(dev)).loss.item(); n += 1   # no mask: see run_stage
    model.train()
    return tot / max(1, n)


STOP = {"sig": False}


def _on_term(*_):
    STOP["sig"] = True                                   # kill PID (SIGTERM) = pause: saved at the next step


def _save_last(name, model, opt, st, out):
    d = os.path.join(out, f"{name}-last")
    tmp = d + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp)
    torch.save(opt.state_dict(), os.path.join(tmp, "opt.pt"))
    json.dump(st, open(os.path.join(tmp, "state.json"), "w"))
    shutil.rmtree(d, ignore_errors=True)
    os.rename(tmp, d)                                    # a crash mid-save leaves the previous -last whole


def per_index(n, seed):
    """Row / block i of the whole run (all passes): pass k is a fresh shuffle of 0..n-1 (seeded, so a resume
    continues exactly where it stopped and every item is seen once per pass)."""
    cache = {}

    def at(i):
        k, j = divmod(i, n)
        if k not in cache:
            cache.clear()
            g = torch.Generator().manual_seed(seed + k)
            cache[k] = torch.randperm(n, generator=g)
        return int(cache[k][j])
    return at


def run_stage(name, model, batch_fn, val_batches, steps, accum, peak, warm, eval_every, patience, keep, out, lf, dev,
              max_sec=0, save_all=False, save_every=200):
    """batch_fn(step, k) -> batch k of update `step` (deterministic, so a resume repeats nothing and skips nothing).
    Pause: touch OUT/PAUSE or kill the PID: {name}-last (weights, optimizer, state) is saved, exit code 10; the next
    start resumes from it. keep = how many best checkpoints to keep as <name>-ck-<step>. max_sec: on a fresh start the
    speed is measured after 100 steps and the step count (and warm-up) shortened to fit."""
    opt = torch.optim.AdamW(model.parameters(), lr=peak, betas=(0.9, 0.95), weight_decay=0.1)
    st = {"step": 0, "steps": steps, "warm": warm, "best": float("inf"), "bad": 0, "kept": [], "shortened": False}
    last = os.path.join(out, f"{name}-last")
    if os.path.exists(os.path.join(last, "state.json")):
        st = json.load(open(os.path.join(last, "state.json")))
        opt.load_state_dict(torch.load(os.path.join(last, "opt.pt"), map_location=dev))
        st["kept"] = [tuple(x) for x in st["kept"]]
        log(lf, stage=name, event="resumed", step=st["step"], steps=st["steps"], best_val=st["best"])
    t0, seen, t_mark = time.time(), 0, st["step"]
    while st["step"] < st["steps"]:
        step = st["step"]
        if STOP["sig"] or os.path.exists(os.path.join(out, "PAUSE")):
            _save_last(name, model, opt, st, out)
            log(lf, stage=name, event="paused", step=step); raise SystemExit(10)
        if step - t_mark == 100 and max_sec and not st["shortened"]:
            per = (time.time() - t0) / 100 * 1.08          # + evals
            st["shortened"] = True
            if per * st["steps"] > max_sec:
                now = int(max_sec / per)
                log(lf, stage=name, event="shortened to fit the time cap", planned=st["steps"], now=now,
                    sec_per_step=round(per, 3))
                st["steps"] = now
                st["warm"] = min(st["warm"], max(1, now // 10))
            else:
                log(lf, stage=name, event="fits the time cap", steps=st["steps"], sec_per_step=round(per, 3),
                    hours=round(per * st["steps"] / 3600, 2))
        steps, warm = st["steps"], st["warm"]
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, peak, warm)
        opt.zero_grad(set_to_none=True)
        tl = 0.0
        for k in range(accum):
            x, y, a = batch_fn(step, k)
            # no attention_mask: padding is only at the end and its labels are -100, so causal attention already
            # gives the same loss; a real mask sends MPS to a very slow path in stage 2
            loss = model(input_ids=x.to(dev), labels=y.to(dev)).loss / accum
            loss.backward(); tl += loss.item(); seen += int(a.sum())
        if not math.isfinite(tl):
            log(lf, stage=name, step=step, error="loss is not finite"); raise SystemExit(3)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        st["step"] = step + 1
        if dev == "mps":           # without this the MPS cache grows with every new length in stage 2 and an
            torch.mps.empty_cache()  # 8 GB machine starts swapping (steps go from seconds to minutes)

        if step % 50 == 0:
            log(lf, stage=name, step=step, loss=round(tl, 4), lr=round(lr_at(step, steps, peak, warm), 6),
                tok_s=int(seen / (time.time() - t0)))
        done = False
        if (step + 1) % eval_every == 0 or step + 1 == steps:
            v = val_loss(model, val_batches, dev)
            rec = {"stage": name, "step": step + 1, "val_loss": round(v, 4)}
            if v < st["best"]:
                st["best"], st["bad"] = v, 0; rec["best"] = True
                model.save_pretrained(os.path.join(out, f"{name}-best"))
            else:
                st["bad"] += 1
            if save_all:
                model.save_pretrained(os.path.join(out, f"{name}-ck-{step + 1}"))
            elif keep:
                st["kept"].append((v, step + 1)); st["kept"].sort()
                d = os.path.join(out, f"{name}-ck-{step + 1}")
                if (v, step + 1) in st["kept"][:keep]:
                    model.save_pretrained(d)
                for _, s_ in st["kept"][keep:]:
                    shutil.rmtree(os.path.join(out, f"{name}-ck-{s_}"), ignore_errors=True)
                st["kept"] = st["kept"][:keep]
            log(lf, **rec)
            if patience and st["bad"] >= patience:
                log(lf, stage=name, event="early stop", step=step + 1, best_val=round(st["best"], 4)); done = True
        if done:
            break
        if (step + 1) % save_every == 0:
            _save_last(name, model, opt, st, out)
    _save_last(name, model, opt, dict(st, finished=True), out)
    return st["best"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--hidden", type=int, default=384); ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--heads", type=int, default=6); ap.add_argument("--inter", type=int, default=1024)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--block", type=int, default=512); ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=8)     # bs 8 x accum 8: fits a 4 GB GPU at width 384
    ap.add_argument("--pre-epochs", type=float, default=3.0); ap.add_argument("--pre-lr", type=float, default=8e-4)
    ap.add_argument("--sft-epochs", type=float, default=12.0); ap.add_argument("--sft-lr", type=float, default=3e-4)
    ap.add_argument("--sft-bs", type=int, default=4); ap.add_argument("--sft-accum", type=int, default=8)  # same 32 rows a step
    ap.add_argument("--sft-patience", type=int, default=5); ap.add_argument("--keep", type=int, default=3)
    ap.add_argument("--pre-hours", type=float, default=11.0, help="time cap, stage 1"); ap.add_argument("--sft-hours", type=float, default=4.0)
    ap.add_argument("--save-every-eval", action="store_true", help="keep every fine-tune eval as sft-ck-<step>")
    ap.add_argument("--sft-eval-every", type=int, default=0, help="0: half an epoch")
    ap.add_argument("--smoke", action="store_true", help="tiny model, few steps: checks the whole path")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    if a.smoke:
        a.vocab, a.hidden, a.layers, a.heads, a.inter, a.block = 2048, 64, 2, 2, 128, 128
    random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    lf = open(os.path.join(a.out, "train.log"), "a")
    FORMAT_WORDS[:] = json.load(open(os.path.join(a.data, "format_words.json")))
    tdir = os.path.join(a.out, "tokenizer")
    tok = (PreTrainedTokenizerFast.from_pretrained(tdir) if os.path.exists(os.path.join(tdir, "tokenizer.json"))
           else build_tokenizer(a.data, tdir, a.vocab))
    multi = [w for w in FORMAT_WORDS if len(tok(w, add_special_tokens=False)["input_ids"]) > 1]
    log(lf, event="tokenizer", vocab=len(tok), format_words=len(FORMAT_WORDS), not_one_token=multi[:40],
        not_one_token_count=len(multi))

    tr, dtr = sft_rows(tok, os.path.join(a.data, "sft_train.jsonl"), a.ctx)
    va, dva = sft_rows(tok, os.path.join(a.data, "sft_val.jsonl"), a.ctx)
    lens = sorted(len(i) for i, _ in tr)
    log(lf, event="sft rows", train=len(tr), val=len(va), longest=lens[-1], median=lens[len(lens) // 2],
        dropped_over_ctx_train=dtr, dropped_over_ctx_val=dva)

    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=a.hidden, intermediate_size=a.inter, num_hidden_layers=a.layers,
                      num_attention_heads=a.heads, num_key_value_heads=a.heads, max_position_embeddings=a.ctx,
                      tie_word_embeddings=True, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
                      pad_token_id=tok.pad_token_id)
    import signal
    signal.signal(signal.SIGTERM, _on_term); signal.signal(signal.SIGINT, _on_term)
    pre_dir = os.path.join(a.out, "pretrain-best")
    if os.path.exists(os.path.join(a.out, "pretrain.done")):
        log(lf, event="stage 1 already done, reused")
    else:
        pre_last = os.path.join(a.out, "pretrain-last")
        resume = os.path.exists(os.path.join(pre_last, "state.json"))
        model = (LlamaForCausalLM.from_pretrained(pre_last) if resume else LlamaForCausalLM(cfg)).to(a.device)
        nparam = sum(p.numel() for p in model.parameters())
        tb, ntok = blocks_of(tok, os.path.join(a.data, "pretrain.txt"), a.block)
        vb, nvt = blocks_of(tok, os.path.join(a.data, "pretrain_val.txt"), a.block)
        vb = vb[:400]
        steps = max(30, int(a.pre_epochs * len(tb) / (a.bs * a.accum))) if not a.smoke else 30
        log(lf, event="stage 1", params=nparam, pretrain_tokens=ntok, val_tokens=nvt, blocks=len(tb), steps=steps,
            tokens_per_step=a.bs * a.accum * a.block, passes=a.pre_epochs, resume=resume)
        blk = per_index(len(tb), a.seed)

        def pre_batch(step, k):                 # every block once per pass, shuffled
            base = (step * a.accum + k) * a.bs
            x = tb[torch.tensor([blk(base + j) for j in range(a.bs)])].long()
            return x, x.clone(), torch.ones_like(x)
        vb = vb.long()
        vbat = [(vb[i: i + a.bs], vb[i: i + a.bs].clone(), torch.ones_like(vb[i: i + a.bs]))
                for i in range(0, len(vb), a.bs)]
        run_stage("pretrain", model, pre_batch, vbat, steps, a.accum, a.pre_lr, min(1000, steps // 10),
                  10 if a.smoke else 500, 0, 0, a.out, lf, a.device, a.pre_hours * 3600,
                  save_every=10 if a.smoke else 200)
        save_tok(tok, pre_dir)
        open(os.path.join(a.out, "pretrain.done"), "w").write("ok\n")
        del model, tb

    sft_last = os.path.join(a.out, "sft-last")
    resume = os.path.exists(os.path.join(sft_last, "state.json"))
    if resume and json.load(open(os.path.join(sft_last, "state.json"))).get("finished"):
        log(lf, event="stage 2 already finished")
    else:
        model = LlamaForCausalLM.from_pretrained(sft_last if resume else pre_dir).to(a.device)
        per_epoch = max(1, len(tr) // (a.sft_bs * a.sft_accum))
        steps = int(a.sft_epochs * per_epoch) if not a.smoke else 30
        row = per_index(len(tr), a.seed + 1000)

        def sft_batch(step, k):                 # every row once per pass, shuffled
            base = (step * a.sft_accum + k) * a.sft_bs
            return pad_batch([tr[row(base + j)] for j in range(a.sft_bs)], tok.pad_token_id)
        vs = sorted(va, key=lambda r: len(r[0]))
        vbat = [pad_batch(vs[i: i + a.sft_bs], tok.pad_token_id) for i in range(0, len(vs), a.sft_bs)]
        log(lf, event="stage 2", steps=steps, per_epoch=per_epoch, resume=resume)
        run_stage("sft", model, sft_batch, vbat, steps, a.sft_accum, a.sft_lr, min(200, steps // 10),
                  10 if a.smoke else (a.sft_eval_every or max(50, per_epoch // 2)), a.sft_patience, a.keep, a.out, lf,
                  a.device, a.sft_hours * 3600, save_all=a.save_every_eval, save_every=10 if a.smoke else 300)
    for d in os.listdir(a.out):
        if d.startswith("sft-ck-") or d == "sft-best":
            save_tok(tok, os.path.join(a.out, d))
    log(lf, event="done")


if __name__ == "__main__":
    main()
