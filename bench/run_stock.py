"""Stock Lightning on the bench sets (conditions 'stock' and 'stock_python'), through a plain vLLM server.

    LIGHTNING_URL=SERVER/v1/completions LOOP_MODEL=lightning LOOP_TOKENIZER=BASE_DIR VLLM_API_KEY=... \
    [LOOP_TEMP=0] python bench/run_stock.py COND RUN [SET ...] [--workers 64]
COND: stock         no tools, qwenputer's system line "Do every arithmetic step exactly."
      stock_python  a ```python block runs in a fresh Python subprocess (10 s timeout, user nobody when we are root)
                    and its stdout comes back in an ```output block, as the Forth machine's output does
RUN:  a label (s0..s3 sampled at the card's temperature 1.0 / top-p 0.95; greedy with LOOP_TEMP=0).
The turn is loop_sim's: thinking on, closed at --think-budget (12000), the whole turn capped at --max-tokens (24000).
Each finished rollout is appended to bench/raw/COND/SET.RUN.jsonl; the same command resumes.
"""

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "training"), str(HERE)]
import loop_sim as LS  # noqa: E402  (complete(): the server call with the card's sampling)
from score import exact  # noqa: E402

SYSTEM = {"stock": "Do every arithmetic step exactly.",  # qwenputer rv/evalarith.py NUDGE_NONE
          "stock_python": ("You have a Python interpreter for exact arithmetic. To run code, write it in a ```python "
                           "block; when you close the block, it runs and its printed output comes back in an ```output "
                           "block. Each block runs in a fresh interpreter, so print what you need. Use it for every "
                           "arithmetic step instead of calculating by hand.")}  # qwenputer's NUDGE, for Python
OPEN, GRACE, ROUNDS = "```python\n", 1000, 12
PY = sys.executable  # the bench venv's python (not the system one)


def run_python(code):
    """One block in a fresh interpreter: stdout, then the last line of a traceback; 10 s; as nobody when root."""
    with tempfile.TemporaryDirectory() as d:
        os.chmod(d, 0o777)
        f = Path(d) / "main.py"
        f.write_text(code)
        f.chmod(0o644)
        drop = dict(user=65534, group=65534, extra_groups=[]) if os.geteuid() == 0 else {}
        try:
            r = subprocess.run([PY, "-I", str(f)], cwd=d, capture_output=True, text=True, timeout=10,
                               env={"PATH": "/usr/bin:/bin", "PYTHONIOENCODING": "utf-8"}, **drop)
            err = r.stderr.strip().splitlines()
            out = r.stdout.rstrip("\n") + (("\n" if r.stdout.strip() else "") + err[-1] if r.returncode and err else "")
        except subprocess.TimeoutExpired:
            out = "error: timed out after 10 s"
    return (out[-2000:] if len(out) > 2000 else out) or "(no output)"


def solve(p, cond, tok, think_budget, max_tokens):
    msgs = [{"role": "system", "content": SYSTEM[cond]}, {"role": "user", "content": p["q"]}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    text, blocks, forced = "", [], None
    ntok = lambda s: len(tok(s, add_special_tokens=False).input_ids)  # noqa: E731
    while True:
        used = ntok(text)
        left = max_tokens - used
        if left <= 0:
            break
        n = left
        if "</think>" not in text:  # the card's think budget (+GRACE for a python block still open)
            k = text.rfind(OPEN)
            budget = think_budget + GRACE * (cond == "stock_python" and k >= 0 and "```" not in text[k + len(OPEN):])
            if used >= budget:
                text, forced = text + "\n</think>\n\n", used
                continue
            n = min(left, budget - used)
        piece, fin = LS.complete(prompt + text, n, LS.URLS[0])
        text += piece
        if not piece.endswith(("```", "```\n")):
            if fin == "length" and n < left:
                text += "" if piece else "\n</think>\n\n"
                continue
            break
        end = text.rstrip("\n")
        k = text.rfind(OPEN)
        body = end[k + len(OPEN):-3] if k >= 0 else None
        if cond != "stock_python" or body is None or "```" in body or "</think>" in body or len(blocks) >= ROUNDS:
            continue
        out = run_python(body)
        text = end + "\n```output\n" + out + "\n```\n"
        blocks.append({"program": body, "output": out})
    return {"text": text, "tokens": ntok(text), "blocks": blocks, "forced_think_close_at": forced,
            "turn_cut": ntok(text) >= max_tokens}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("cond", choices=list(SYSTEM))
    a.add_argument("run")
    a.add_argument("sets", nargs="*", default=["nupa", "mul_big", "div_big", "add_or_sub_big", "mul_div_multiple_longer"])
    a.add_argument("--workers", type=int, default=64)
    a.add_argument("--think-budget", type=int, default=12000)
    a.add_argument("--max-tokens", type=int, default=24000)
    a.add_argument("--limit", type=int, default=0, help="first N problems per set (smoke test)")
    args = a.parse_args()
    if args.cond == "stock":
        LS.STOPS = []  # nothing to run: no stop at fences
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ["LOOP_TOKENIZER"])
    lock, todo = threading.Lock(), []
    for s in args.sets:
        P = json.load(open(HERE / "sets" / f"{s}.json"))["problems"][:args.limit or None]
        out = HERE / "raw" / args.cond / f"{s}.{args.run}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        done = {json.loads(x)["id"] for x in open(out)} if out.exists() else set()
        todo += [(p, out) for p in P if p["id"] not in done]
    print(f"{len(todo)} rollouts to run", flush=True)

    def one(job):
        p, out = job
        try:
            r = solve(p, args.cond, tok, args.think_budget, args.max_tokens)
        except Exception as e:  # noqa: BLE001 - not saved: a rerun retries it
            print(f"SKIPPED {p['id']}: {e!r}", flush=True)
            return
        r = {"id": p["id"], "set": p["set"], "cond": args.cond, "run": args.run, "temp": LS.TEMP,
             "question": p["question"], "gold": p["gold"]} | exact(r["text"], p["gold"]) | r
        with lock, open(out, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(f"{'OK  ' if r['exact'] else 'MISS'} {p['id']} want {p['gold']} got {r['pred']} | {r['tokens']} tok, "
              f"{len(r['blocks'])} block(s)", flush=True)

    with cf.ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(one, todo))


if __name__ == "__main__":
    main()
