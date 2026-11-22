#!/usr/bin/env python3
"""Score a served model on the JOB, not the loss. Standard library only.

    python3 score.py --url http://127.0.0.1:8080 --test data/sample/test.jsonl [--out results.jsonl]

Sends each test row's request (messages up to and including the first user turn, plus the row's tools) to an
/v1/chat/completions server at temperature 0 and reads the tool call the model made. It reports TWO
numbers side by side, plus the failures that are not the model's fault:

  strict   the tool name and every argument equal the row's "expect"          (did it do the job exactly?)
  usable   a call to a known tool, arguments valid JSON, required keys present (could a host run it at all?)
  no_call  the model answered in plain text, or wrote a tool reply ("<tool_response>") itself
  infra    HTTP errors, 503 "loading model", timeouts, after 3 tries: NOT counted against the model

When strict and usable move in opposite directions between two checkpoints, stop and read the outputs.
Use the same frozen test file every time you compare two models.
"""
import argparse, json, sys, time, urllib.request, urllib.error


def ask(url, row, timeout=300):
    msgs = []
    for m in row["messages"]:
        msgs.append(m)
        if m["role"] == "user":
            break
    body = {"messages": msgs, "temperature": 0, "max_tokens": 400}
    if row.get("tools"):
        body["tools"] = row["tools"]
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    last = ""
    for attempt in range(3):
        try:
            data = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
            return data["choices"][0]["message"], ""
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
        except Exception as e:  # connection refused, timeout, bad JSON
            last = type(e).__name__
        time.sleep(2 + 3 * attempt)  # a server still loading its model looks like this; wait, then retry
    return None, last


def first_call(msg):
    calls = msg.get("tool_calls") or []
    if not calls:
        return None, None
    fn = calls[0]["function"]
    args = fn.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return fn.get("name"), None
    return fn.get("name"), args


def usable(name, args, tools):
    spec = {t["function"]["name"]: t["function"] for t in tools or []}
    if name not in spec or not isinstance(args, dict):
        return False
    return all(k in args for k in spec[name].get("parameters", {}).get("required", []))


def score(url, rows):
    res = {"n": len(rows), "strict": 0, "usable": 0, "no_call": 0, "infra": 0, "fake_reply": 0}
    detail = []
    for i, row in enumerate(rows):
        msg, err = ask(url, row)
        if msg is None:
            res["infra"] += 1
            detail.append({"row": i, "result": "infra", "why": err})
            continue
        name, args = first_call(msg)
        text = msg.get("content") or ""
        if "<tool_response>" in text:
            res["fake_reply"] += 1
        if name is None:
            res["no_call"] += 1
            detail.append({"row": i, "result": "no_call", "text": text[:120]})
            continue
        ok_use = usable(name, args, row.get("tools"))
        exp = row.get("expect")
        ok_strict = bool(exp) and name == exp["name"] and args == exp["arguments"]
        res["usable"] += ok_use
        res["strict"] += ok_strict
        detail.append({"row": i, "result": "strict" if ok_strict else ("usable" if ok_use else "bad"),
                       "got": {"name": name, "arguments": args}, "expect": exp})
    scored = res["n"] - res["infra"]
    res["scored"] = scored
    res["strict_pct"] = round(100 * res["strict"] / scored, 1) if scored else None
    res["usable_pct"] = round(100 * res["usable"] / scored, 1) if scored else None
    return res, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(a.test) if line.strip()]
    if a.limit:
        rows = rows[: a.limit]
    res, detail = score(a.url, rows)
    print(json.dumps(res))
    if a.out:
        with open(a.out, "w") as f:
            f.write(json.dumps(res) + "\n")
            for d in detail:
                f.write(json.dumps(d) + "\n")
    return 0 if res["infra"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
