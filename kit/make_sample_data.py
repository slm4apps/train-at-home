#!/usr/bin/env python3
"""Write the tiny synthetic sample dataset (a made-up shopping-list assistant). Deterministic; no outside data.

    python3 make_sample_data.py [OUT_DIR]        # default: data/sample

Writes train.jsonl, valid.jsonl, test.jsonl. Every row is one chat conversation in the "messages" format:
user request -> assistant tool call -> tool result -> assistant answer. Test rows also carry "expect": the tool call
a correct model must make, so score.py can check it.
"""
import json, os, random, sys

TOOLS = [
    {"type": "function", "function": {
        "name": "add_item", "description": "Add an item to the shopping list.",
        "parameters": {"type": "object", "properties": {
            "item": {"type": "string"}, "qty": {"type": "integer"}}, "required": ["item", "qty"]}}},
    {"type": "function", "function": {
        "name": "remove_item", "description": "Remove an item from the shopping list.",
        "parameters": {"type": "object", "properties": {"item": {"type": "string"}}, "required": ["item"]}}},
    {"type": "function", "function": {
        "name": "show_list", "description": "Show the whole shopping list.",
        "parameters": {"type": "object", "properties": {}}}},
]
ITEMS = ["milk", "eggs", "bread", "rice", "apples", "coffee", "tea", "butter", "pasta", "onions", "carrots", "soap",
         "beans", "cheese", "flour", "sugar", "salt", "oil", "yogurt", "lemons"]
NUM = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
ADD = ["add {q} {i} to my list", "I need {q} {i}", "put {q} {i} on the shopping list", "please buy {q} {i}",
       "we are out of {i}, get {q}", "{q} {i}, add it"]
REMOVE = ["remove {i} from the list", "I do not need {i} any more", "take {i} off my list", "drop {i}"]
SHOW = ["what is on my list?", "show the shopping list", "read my list back to me", "what do I still need to buy?"]


def call(name, **args):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": name, "arguments": args}}]}


def make(rng):
    kind = rng.choice(["add", "add", "add", "remove", "show"])
    if kind == "add":
        i, q = rng.choice(ITEMS), rng.randint(1, 6)
        word = str(q) if rng.random() < 0.6 else NUM[q]
        user = rng.choice(ADD).format(q=word, i=i)
        msgs = [{"role": "user", "content": user}, call("add_item", item=i, qty=q),
                {"role": "tool", "content": f"ok: {i} x{q} added"},
                {"role": "assistant", "content": f"Added {q} {i}."}]
        expect = {"name": "add_item", "arguments": {"item": i, "qty": q}}
    elif kind == "remove":
        i = rng.choice(ITEMS)
        msgs = [{"role": "user", "content": rng.choice(REMOVE).format(i=i)}, call("remove_item", item=i),
                {"role": "tool", "content": f"ok: {i} removed"},
                {"role": "assistant", "content": f"Removed {i}."}]
        expect = {"name": "remove_item", "arguments": {"item": i}}
    else:
        n = rng.randint(1, 3)
        listed = ", ".join(f"{rng.choice(ITEMS)} x{rng.randint(1, 6)}" for _ in range(n))
        msgs = [{"role": "user", "content": rng.choice(SHOW)}, call("show_list"),
                {"role": "tool", "content": listed},
                {"role": "assistant", "content": f"You need: {listed}."}]
        expect = {"name": "show_list", "arguments": {}}
    return {"messages": msgs, "tools": TOOLS}, expect


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample")
    os.makedirs(out, exist_ok=True)
    rng = random.Random(7)
    seen, rows = set(), []
    while len(rows) < 160:
        row, expect = make(rng)
        key = row["messages"][0]["content"]
        if key in seen:  # a request appears once, so valid/test never repeat a train request
            continue
        seen.add(key)
        rows.append((row, expect))
    split = {"train": rows[:120], "valid": rows[120:140], "test": rows[140:]}
    for name, part in split.items():
        with open(os.path.join(out, name + ".jsonl"), "w") as f:
            for row, expect in part:
                if name == "test":
                    row = dict(row, expect=expect)
                f.write(json.dumps(row) + "\n")
        print(name, len(part))


if __name__ == "__main__":
    main()
