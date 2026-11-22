#!/usr/bin/env python3
"""Write a tiny made-up dataset for train_from_scratch.py (the same shopping-list world as make_sample_data.py).
Deterministic; standard library only; no outside text.

    python3 make_scratch_sample.py [OUT_DIR]        # default: data/scratch-sample (next to this script)

Writes pretrain.txt, pretrain_val.txt (documents separated by a blank line), sft_train.jsonl, sft_val.jsonl
(one {"job", "request", "answer"} per line) and format_words.json. The answer format is compact:
    add_item item=milk qty=2   |   remove_item item=milk   |   show_list
It exists to prove the pipeline, not to teach a useful model.
"""
import json, os, random, sys

ITEMS = ["milk", "eggs", "bread", "rice", "apples", "coffee", "tea", "butter", "pasta", "onions", "carrots", "soap",
         "beans", "cheese", "flour", "sugar", "salt", "oil", "yogurt", "lemons"]
NUM = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
ADD = ["add {q} {i} to my list", "I need {q} {i}", "put {q} {i} on the shopping list", "please buy {q} {i}",
       "we are out of {i}, get {q}", "{q} {i}, add it"]
REMOVE = ["remove {i} from the list", "I do not need {i} any more", "take {i} off my list", "drop {i}"]
SHOW = ["what is on my list?", "show the shopping list", "read my list back to me", "what do I still need to buy?"]
FORMAT_WORDS = ["add_item", "remove_item", "show_list"]

SENTENCES = [
    "The kitchen was quiet, and {a} sat next to {b} on the counter.",
    "She wrote {a}, {b} and {c} on the shopping list before she left.",
    "We ran out of {a} this morning, so {b} went on the list too.",
    "He carried a bag with {a} and {b} home from the shop.",
    "A good list is short: {a}, {b}, {c}.",
    "The recipe needs {a}, a little {b} and some {c}.",
    "Do not forget the {a}. The {b} can wait until next week.",
    "In the cupboard there were {a}, {b} and a half empty jar of {c}.",
    "Every week the family buys {a} and {b}, and sometimes {c}.",
    "The shop was busy, but the {a} was easy to find.",
    "Cross {a} off the list once it is in the basket.",
    "Breakfast is {a} with {b}; lunch is {c}.",
]


def request_pair(rng):
    kind = rng.choice(["add", "add", "add", "remove", "show"])
    if kind == "add":
        i, q = rng.choice(ITEMS), rng.randint(1, 6)
        word = str(q) if rng.random() < 0.6 else NUM[q]
        return rng.choice(ADD).format(q=word, i=i), f"add_item item={i} qty={q}"
    if kind == "remove":
        i = rng.choice(ITEMS)
        return rng.choice(REMOVE).format(i=i), f"remove_item item={i}"
    return rng.choice(SHOW), "show_list"


def document(rng):
    lines = []
    for _ in range(rng.randint(3, 6)):
        a, b, c = rng.sample(ITEMS, 3)
        lines.append(rng.choice(SENTENCES).format(a=a, b=b, c=c))
    if rng.random() < 0.5:
        lines.append("List: " + ", ".join(rng.sample(ITEMS, rng.randint(2, 4))) + ".")
    return " ".join(lines)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "scratch-sample")
    os.makedirs(out, exist_ok=True)
    rng = random.Random(11)
    seen, rows = set(), []
    while len(rows) < 340:  # a request appears once, so val never repeats a train request
        req, ans = request_pair(rng)
        if req in seen:
            continue
        seen.add(req)
        rows.append({"job": "list", "request": req, "answer": ans})
    split = {"sft_train.jsonl": rows[:300], "sft_val.jsonl": rows[300:]}
    for name, part in split.items():
        with open(os.path.join(out, name), "w") as f:
            for r in part:
                f.write(json.dumps(r) + "\n")
    train_requests = [r["request"] + "." for r in rows[:300]]
    docs = [document(rng) for _ in range(400)]
    # the request texts join the pretraining text (train requests only), one short document each group of five
    docs += [" ".join(train_requests[k:k + 5]) for k in range(0, len(train_requests), 5)]
    rng.shuffle(docs)
    val_docs, train_docs = docs[:40], docs[40:]
    with open(os.path.join(out, "pretrain.txt"), "w") as f:
        f.write("\n\n".join(train_docs) + "\n")
    with open(os.path.join(out, "pretrain_val.txt"), "w") as f:
        f.write("\n\n".join(val_docs) + "\n")
    with open(os.path.join(out, "format_words.json"), "w") as f:
        json.dump(FORMAT_WORDS, f)
    print("sft_train", len(split["sft_train.jsonl"]), "sft_val", len(split["sft_val.jsonl"]),
          "pretrain docs", len(train_docs), "val docs", len(val_docs))


if __name__ == "__main__":
    main()
