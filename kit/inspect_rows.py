#!/usr/bin/env python3
"""Decode training rows back to text and mark the part the model will learn. Run it before every training run.

    python3 inspect_rows.py --base BASE_MODEL_DIR_OR_ID --rows data/sample/train.jsonl [--n 3]

Trained text is wrapped in [[[ ... ]]]. If a tool reply, a system prompt or the user's words sit inside the brackets,
stop: the model would learn to write them. Uses the same span rule as train_lora_cuda.py (rendered message prefixes).
Needs transformers; loads only the tokenizer.
"""
import argparse, json
from transformers import AutoTokenizer


def ids_of(x):
    return x["input_ids"] if hasattr(x, "keys") else x


def spans(tok, row):
    msgs, tools = row["messages"], row.get("tools")
    full = ids_of(tok.apply_chat_template(msgs, tools=tools, tokenize=True))
    labels = [False] * len(full)
    for i, m in enumerate(msgs):
        if m["role"] != "assistant" or m.get("weight", 1) == 0:
            continue
        pre = ids_of(tok.apply_chat_template(msgs[:i], tools=tools, tokenize=True, add_generation_prompt=True))
        upto = ids_of(tok.apply_chat_template(msgs[: i + 1], tools=tools, tokenize=True))
        assert full[: len(upto)] == upto, "chat template: a message prefix does not match the whole conversation"
        for j in range(len(pre), len(upto)):
            labels[j] = True
    return full, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.base)
    with open(a.rows) as f:
        for k, line in enumerate(f):
            if k >= a.n:
                break
            full, lab = spans(tok, json.loads(line))
            out, run, cur = [], [], lab[0]
            for t, l in zip(full, lab):
                if l != cur:
                    out.append((cur, tok.decode(run)))
                    run, cur = [], l
                run.append(t)
            out.append((cur, tok.decode(run)))
            print(f"----- row {k}: {len(full)} tokens, {sum(lab)} trained -----")
            print("".join(f"[[[{s}]]]" if t else s for t, s in out))


if __name__ == "__main__":
    main()
