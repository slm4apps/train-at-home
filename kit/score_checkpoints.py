#!/usr/bin/env python3
"""Score every checkpoint (and several adapter strengths) with llama-server, one at a time, on the same test file.

    python3 score_checkpoints.py --base base.gguf --test data/sample/test.jsonl \
        --adapters out/ck100.gguf out/ck200.gguf out/final.gguf --scales 0.7 1.0

--base      one or more model GGUFs, each scored alone; the LoRA adapters below are applied to the FIRST one
            (full precision or q8_0). Several whole-model files = scoring fused checkpoints one by one
--adapters  LoRA GGUF files (convert each checkpoint with convert_gguf.sh lora); omit to score the base alone
--scales    adapter strengths to try for each adapter; at least three is a good rule (0.7, 1.0 and one more)
Prints one table row per (checkpoint, scale): strict %, usable %, no_call, infra. Pick by this table, not by the loss.
One server at a time, on a free port, stopped afterwards. On a shared machine set LOCK=/tmp/score.lock: a second
run then refuses to start while the first holds the lock.
"""
import argparse, json, os, subprocess, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score as S


def wait_ready(url, proc, tries=240):
    for _ in range(tries):
        if proc.poll() is not None:
            return False
        try:
            if json.loads(urllib.request.urlopen(url + "/health", timeout=2).read()).get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def run_one(a, base, adapter, scale, rows):
    cmd = [a.server, "-m", base, "--jinja", "-c", str(a.ctx), "--host", "127.0.0.1", "--port", str(a.port),
           "-t", str(a.threads), "--temp", "0"]
    if adapter:
        cmd += ["--lora-scaled", adapter, str(scale)]
    log = open(os.path.join(a.logdir, "server-%d.log" % int(time.time() * 1000)), "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{a.port}"
    try:
        if not wait_ready(url, proc):
            return {"error": "server did not start (see " + log.name + ")"}
        return S.score(url, rows)[0]
    finally:
        proc.terminate()
        proc.wait()
        time.sleep(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, nargs="+")
    ap.add_argument("--test", required=True)
    ap.add_argument("--adapters", nargs="*", default=[])
    ap.add_argument("--scales", nargs="*", type=float, default=[1.0])
    ap.add_argument("--server", default="llama-server")
    ap.add_argument("--port", type=int, default=8133)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--logdir", default=".")
    ap.add_argument("--out", help="append one JSON line per result here")
    a = ap.parse_args()
    lock = os.getenv("LOCK")
    if lock:
        if os.path.exists(lock):
            sys.exit(f"lock {lock} exists: another scoring run is active (one at a time)")
        open(lock, "w").write(str(os.getpid()))
    try:
        rows = [json.loads(line) for line in open(a.test) if line.strip()]
        jobs = [(b, None, 0.0) for b in a.base] + [(a.base[0], ad, s) for ad in a.adapters for s in a.scales]
        print(f"{'checkpoint':40} {'scale':>5} {'strict%':>8} {'usable%':>8} {'no_call':>7} {'infra':>5}")
        for b, ad, s in jobs:
            r = run_one(a, b, ad, s, rows)
            name = os.path.basename(ad) if ad else os.path.basename(b) + " (alone)"
            if "error" in r:
                print(f"{name:40} {s:>5} {r['error']}")
            else:
                print(f"{name:40} {s:>5} {r['strict_pct']:>8} {r['usable_pct']:>8} {r['no_call']:>7} {r['infra']:>5}")
            if a.out:
                open(a.out, "a").write(json.dumps({"checkpoint": name, "scale": s, **r}) + "\n")
    finally:
        if lock and os.path.exists(lock):
            os.remove(lock)


if __name__ == "__main__":
    main()
