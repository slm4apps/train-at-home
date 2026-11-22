#!/usr/bin/env python3
"""Check a chat dataset and write it in the shape each trainer wants. Needs only the Python standard library.

    python3 prepare_data.py check DATA_DIR            # train.jsonl / valid.jsonl [/ test.jsonl]: format + leak checks
    python3 prepare_data.py per-turn DATA_DIR OUT_DIR # MLX: one row per assistant turn, loss on that turn only

Row format ("messages" chat format, one JSON object per line):
    {"messages": [{"role": "user", "content": "..."},
                  {"role": "assistant", "content": "", "tool_calls": [{"type": "function",
                       "function": {"name": "add_item", "arguments": {"item": "milk", "qty": 2}}}]},
                  {"role": "tool", "content": "ok: milk x2 added"},
                  {"role": "assistant", "content": "Added 2 milk."}],
     "tools": [...optional tool schemas...]}
An assistant message may carry "weight": 0 = keep as context, do not learn it (train_lora_cuda.py honours it).

Why per-turn: mlx_lm's --mask-prompt trains only the LAST assistant message of a row. A row with a tool reply in the
middle would otherwise teach the model to write the tool reply itself. So each assistant turn becomes its own row,
with everything before it as context.
"""
import json, os, sys

ROLES = {"system", "user", "assistant", "tool"}


def read(path):
    rows = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                sys.exit(f"{path}:{n}: not JSON ({e})")
    return rows


def problems(row):
    out = []
    msgs = row.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return ["no messages"]
    for i, m in enumerate(msgs):
        if m.get("role") not in ROLES:
            out.append(f"message {i}: bad role {m.get('role')!r}")
        if "content" not in m:
            out.append(f"message {i}: no content key")
        for tc in m.get("tool_calls", []) or []:
            fn = tc.get("function", {})
            if not fn.get("name"):
                out.append(f"message {i}: tool call without a name")
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    json.loads(args)
                except json.JSONDecodeError:
                    out.append(f"message {i}: tool call arguments are not valid JSON")
    if msgs[-1].get("role") != "assistant":
        out.append("last message is not an assistant turn")
    if not any(m.get("role") == "assistant" and m.get("weight", 1) != 0 for m in msgs):
        out.append("no assistant turn to learn from")
    return out


def key(row):
    first = next((m for m in row["messages"] if m["role"] == "user"), {})
    return " ".join(str(first.get("content", "")).lower().split())


def cmd_check(d):
    bad = 0
    keys = {}
    for name in ("train", "valid", "test"):
        p = os.path.join(d, name + ".jsonl")
        if not os.path.exists(p):
            if name != "test":
                sys.exit(f"missing {p}")
            continue
        rows = read(p)
        for n, r in enumerate(rows, 1):
            for pr in problems(r):
                bad += 1
                print(f"{name}.jsonl row {n}: {pr}")
        keys[name] = [key(r) for r in rows]
        calls = sum(len(m.get("tool_calls") or []) for r in rows for m in r["messages"])
        print(f"{name}: {len(rows)} rows, {calls} tool calls, {len(set(keys[name]))} distinct first requests")
    for other in ("valid", "test"):
        if other in keys:
            leak = set(keys["train"]) & set(keys[other])
            if leak:
                bad += len(leak)
                print(f"LEAK: {len(leak)} {other} request(s) also in train, e.g. {sorted(leak)[0]!r}")
    print("OK" if not bad else f"{bad} problem(s)")
    return 1 if bad else 0


def cmd_per_turn(d, out):
    os.makedirs(out, exist_ok=True)
    for name in ("train", "valid", "test"):
        p = os.path.join(d, name + ".jsonl")
        if not os.path.exists(p):
            continue
        n_out = 0
        with open(os.path.join(out, name + ".jsonl"), "w") as f:
            for r in read(p):
                msgs = r["messages"]
                for i, m in enumerate(msgs):
                    if m["role"] == "assistant" and m.get("weight", 1) != 0:
                        row = {"messages": [{k: v for k, v in x.items() if k != "weight"} for x in msgs[: i + 1]]}
                        if r.get("tools"):
                            row["tools"] = r["tools"]
                        f.write(json.dumps(row) + "\n")
                        n_out += 1
        print(f"{name}: {n_out} per-turn rows -> {out}/{name}.jsonl")
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 2 and a[0] == "check":
        sys.exit(cmd_check(a[1]))
    if len(a) == 3 and a[0] == "per-turn":
        sys.exit(cmd_per_turn(a[1], a[2]))
    sys.exit(__doc__)
