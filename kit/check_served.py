#!/usr/bin/env python3
"""Does the file you ship answer what the model you trained answered?

Step 1, on the trained (unconverted) model, greedy decoding -> reference answers:
    python3 check_served.py reference --model HF_DIR --test data/sample/test.jsonl --out out/reference.jsonl
Step 2, a stock llama-server on the converted GGUF, with the chat template the GGUF carries -> compare:
    python3 check_served.py compare --gguf model.gguf --test data/sample/test.jsonl --reference out/reference.jsonl

Both ask the same prompts at temperature 0. For plain-text answers the texts must be identical. For tool calls the
reference holds the raw text with its tags, so the check compares the tool name and every argument value instead.
It prints one line per row and a summary, and exits 0 only when every row matches. On my from-scratch model this
check found a newline the chat template had dropped; without it that would have shipped. A different FIRST word, a
missing tag or a missing newline is a real bug.
"""
import argparse, json, subprocess, sys, time, urllib.request


def prompts(path):
    for line in open(path):
        if not line.strip():
            continue
        row = json.loads(line)
        msgs = []
        for m in row["messages"]:
            msgs.append(m)
            if m["role"] == "user":
                break
        yield msgs, row.get("tools")


def cmd_reference(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    with open(a.out, "w") as f:
        for i, (msgs, tools) in enumerate(prompts(a.test)):
            if i >= a.n:
                break
            enc = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, return_tensors="pt",
                                          return_dict=True)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False)
            text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=False)
            f.write(json.dumps({"i": i, "reply": text.strip()}) + "\n")
    print("wrote", a.out)
    return 0


def matches(msg, want):
    calls = msg.get("tool_calls") or []
    if calls:
        fn = calls[0]["function"]
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            args = json.loads(args)
        return fn["name"] in want and all(str(v) in want for v in args.values())
    return (msg.get("content") or "").strip() == want


def cmd_compare(a):
    ref = {r["i"]: r["reply"] for r in map(json.loads, open(a.reference))}
    cmd = [a.server, "-m", a.gguf, "--jinja", "-c", "4096", "--temp", "0", "--host", "127.0.0.1", "--port", str(a.port)]
    if a.lora_gguf:
        cmd += ["--lora", a.lora_gguf]
    srv = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=open(a.gguf + ".server.log", "w"))
    url = f"http://127.0.0.1:{a.port}"
    same = total = 0
    try:
        for _ in range(240):
            try:
                if json.loads(urllib.request.urlopen(url + "/health", timeout=2).read()).get("status") == "ok":
                    break
            except Exception:
                time.sleep(0.5)
        for i, (msgs, tools) in enumerate(prompts(a.test)):
            if i >= a.n or i not in ref:
                break
            body = {"messages": msgs, "temperature": 0, "max_tokens": a.max_new}
            if tools:
                body["tools"] = tools
            req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            msg = json.loads(urllib.request.urlopen(req, timeout=300).read())["choices"][0]["message"]
            ok = matches(msg, ref[i])
            same += ok
            total += 1
            print(json.dumps({"i": i, "same": ok, "got": (msg.get("content") or json.dumps(msg.get("tool_calls")))[:100],
                              "ref": ref[i][:100]}))
    finally:
        srv.terminate()
    print(json.dumps({"same": same, "of": total}))
    return 0 if same == total and total else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reference")
    r.add_argument("--model", required=True)
    r.add_argument("--test", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--n", type=int, default=40)
    r.add_argument("--max-new", type=int, default=300)
    c = sub.add_parser("compare")
    c.add_argument("--gguf", required=True)
    c.add_argument("--reference", required=True)
    c.add_argument("--test", required=True)
    c.add_argument("--lora-gguf")
    c.add_argument("--server", default="llama-server")
    c.add_argument("--port", type=int, default=8134)
    c.add_argument("--n", type=int, default=40)
    c.add_argument("--max-new", type=int, default=300)
    a = ap.parse_args()
    sys.exit(cmd_reference(a) if a.cmd == "reference" else cmd_compare(a))
