"""Simulated kernel loop (no compiled front end needed): stock Lightning (vLLM server, OpenAI /v1/completions) writes;
whenever it closes a ```forth block, the TEXT-MODE REFERENCE MACHINE (kernel/text_vm.py) runs the block - with the
kernel state of every earlier block in the turn - and its ```output block (results, or `error: ...`) is appended
exactly as the compiled kernel will emit it; generation then continues. Up to --rounds blocks per problem.
  VLLM_API_KEY=... python training/loop_sim.py PROBLEMS.json RUBRIC.txt [--think] [--rounds 4] --out OUT.json
PROBLEMS: [{"q": ..., "a": expected final answer}]; prints each transcript's blocks / outputs and the final answer.
RUBRIC: a text system prompt, or a .json {"system": ..., "shots": [[user, assistant], ...]} (few-shot turns).
Earlier assistant turns lose their <think> part in the chat template: put a shot's program in its visible answer.
A problem may carry "prefill" (a prebuilt library: its block and the kernel's output start the turn) and "lead" (text
before it). Local runs (Mac) set LOOP_SERVER=mlx (mlx_lm.server: training/mlx_local.sh) or LOOP_SERVER=llamacpp
(llama-server; LIGHTNING_URL=http://localhost:PORT/completion) and LOOP_TOKENIZER=<Lightning snapshot>.
"""

import argparse
import collections
import concurrent.futures as cf
import json
import os
import re
import resource
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "kernel"), str(ROOT)]
from forth import sp  # noqa: E402
import text_vm as T  # noqa: E402

URLS = os.environ.get("LIGHTNING_URL", "http://localhost:8200/v1/completions").split(",")  # replicas: comma list
MODEL = os.environ.get("LOOP_MODEL", "lightning")  # served model name
TEMP, TOP_P = float(os.environ.get("LOOP_TEMP", "1.0")), float(os.environ.get("LOOP_TOP_P", "0.95"))  # model card
CHAT_TOK = os.environ.get("LOOP_CHAT_TOKENIZER")  # chat template of another model (the kernel keeps Lightning's tokenizer)
SERVER = os.environ.get("LOOP_SERVER", "vllm")  # vllm | mlx (mlx_lm.server) | llamacpp (llama-server's /completion)
# | ollama (/api/generate)
NUM_CTX = int(os.environ.get("LOOP_NUM_CTX", "32768"))  # ollama: context; keep it fixed (a change reloads the model)
STOPS = ["```\n", "```"]  # a fence: the kernel takes over at the closing ``` token itself (the model may end its turn right after)
GRACE = 1000  # tokens past the thinking budget for a ```forth block that is open when the budget runs out
FENCE_TOKS = {2640: "```\n", 1975: "```"}  # Lightning's fence tokens (mlx drops the stop from the text; its token id tells)


def complete(prompt, max_tokens, url):
    if SERVER == "llamacpp":  # native endpoint: it says which stop ended the text (the OpenAI one drops it)
        body = {"prompt": prompt, "n_predict": max_tokens, "temperature": TEMP, "top_p": TOP_P, "stop": STOPS,
                "top_k": 0, "min_p": 0.0}  # the model card's sampling only (llama.cpp defaults top_k 40, min_p 0.05)
        r = json.load(urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(),
                                                                    {"Content-Type": "application/json"}), timeout=3600))
        return r["content"] + (r["stopping_word"] if r["stop_type"] == "word" else ""), \
            "length" if r["stop_type"] == "limit" else "stop"
    if SERVER == "ollama":  # native /api/generate, raw (the prompt is already templated). Ollama drops a stop string and
        body = {"model": MODEL, "prompt": prompt, "raw": True, "stream": True,  # does not say which one ended the text,
                "options": {"num_predict": max_tokens, "temperature": TEMP, "top_p": TOP_P, "top_k": 0, "min_p": 0.0,
                            "num_ctx": NUM_CTX}}  # so stream and cut at the first fence here (= the "```" stop above)
        text, fin = "", "length"
        with urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(),
                                                           {"Content-Type": "application/json"}), timeout=3600) as resp:
            for line in resp:  # closing the stream early cancels the generation
                r = json.loads(line)
                text += r.get("response", "")
                if (k := text.find("```")) >= 0:
                    return text[:k + 3], "stop"
                if r.get("done"):
                    fin = "length" if r.get("done_reason") == "length" else "stop"
        return text, fin
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": TEMP, "top_p": TOP_P, "stop": STOPS,
            "include_stop_str_in_output": True} | ({"logprobs": True} if SERVER == "mlx" else {})
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json",
                                                                 "Authorization": f"Bearer {os.environ.get('VLLM_API_KEY', '')}"})
    c = json.load(urllib.request.urlopen(req, timeout=3600))["choices"][0]
    if SERVER == "mlx" and c["finish_reason"] == "stop" and c["logprobs"]["content"][-1]["id"] in FENCE_TOKS:
        return c["text"] + FENCE_TOKS[c["logprobs"]["content"][-1]["id"]], "stop"
    return c["text"], c["finish_reason"]


def kernel_output(blocks, tok):
    """Run every ```forth block of the turn so far (state carries) in the text-mode reference; -> the output block text
    the kernel emits at the LAST block's closing fence."""
    text = "".join(f"```forth\n{b}\n```\n" for b in blocks)
    ids = tok(text, add_special_tokens=False).input_ids
    try:
        out, owned, _ = T.stream(ids, tok)
    except Exception:  # a reference-machine bug: show the blocks that reproduce it (run() skips the rollout)
        print("KERNEL CRASH on blocks:", json.dumps(blocks), flush=True)
        raise
    shown = tok.decode([out[o] for o in owned], skip_special_tokens=True)  # the kernel's visible text: its output blocks
    blocks_out = re.findall(r"```output\n(.*?)\n```", shown, re.S)
    return blocks_out[-1] if blocks_out else "(no output block)"


def last_boxed(text):
    """The last \\boxed{...}'s content, braces balanced (\\boxed{\\frac{5}{33}})."""
    i = text.rfind("\\boxed{")
    depth = 1
    for k in range(i + 7, len(text)) if i >= 0 else ():
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if not depth:
            return text[i + 7:k]
    return None


def norm(ans):
    """'1,234' / '3, 4' / '3\\ 4' (LaTeX spaces) -> '1234' / '3 4' / '3 4'; \\frac{5}{33} -> 5/33; \\text{Yes} / true -> True."""
    ans = re.sub(r"\\text\{([^{}]*)\}", r"\1", re.sub(r"\\d?frac\{(-?\d+)\}\{(\d+)\}", r"\1/\2", ans))
    ans = re.sub(r"\\[ ,;!]", " ", ans)
    ans = " ".join(re.sub(r"(?<=\d),(?=\d{3}\b)", "", ans).replace(",", " ").split())
    return {"yes": "True", "true": "True", "no": "False", "false": "False"}.get(ans.lower().rstrip("."), ans)


def same(got, want, p):
    """The exact grader: equal after norm(); the roots of an equation are a set (DeepMind lists them sorted, the
    question asks for no order), so they compare in any order."""
    if got is None:
        return False
    g, w = norm(got), norm(want)
    if w in ("True", "False"):  # a yes / no question: "Yes, 7432 is composite" answers it (the first word decides)
        if g not in ("True", "False"):
            first = norm(re.split(r"[\s,.;:]", g.strip(), maxsplit=1)[0])
            says = re.search(r"((?:\bnot|n't)\s+(?:a\s+)?)?\b(prime|composite)\b", g, re.I)
            asks = re.search(r"\b(prime|composite)\b", p.get("q", ""), re.I)
            if first in ("True", "False") or not (says and asks):
                g = first
            else:  # "51353 is composite" / "is not prime" answers "Is 51353 prime?" (No) by the property it states
                prime = (says.group(2).lower() == "prime") != bool(says.group(1))
                g = str(prime == (asks.group(1).lower() == "prime"))
    if "polynomial_roots" in (p.get("family") or ""):  # "g = 9823 or g = -22": drop "x =" and the joining words
        g = re.sub(r"\\q?quad|\((?:double|triple|multiplicity \d+)(?: root)?\)", " ", g)  # "-5 (double root)", "\quad"
        g = " ".join(t for t in re.sub(r"\b[a-z]\s*=\s*", " ", g).split() if t not in ("or", "and"))
        return sorted(g.split()) == sorted(w.split())
    if re.fullmatch(r"-?[\d.]+", w):  # "15500 micrograms" answers "how many micrograms": a number then unit words only
        t = g.split()
        return bool(t) and t[0] == w and all(re.fullmatch(r"[a-zA-Z]+", x) for x in t[1:])
    if "lcg_trace" in (p.get("family") or ""):  # "every value produced" is ambiguous about the seed: with or without it
        return g == w or g.split()[1:] == w.split()
    if "euclid_trace" in (p.get("family") or ""):  # two readings of the question are reasonable: the smaller first number
        ws, ok = w.split(), [w.split()]          # counted as the first 'remainder' or not; the gcd (= the last remainder)
        if len(ws) > 2 and ws[1] in re.findall(r"\d+", p.get("q", "")):  # printed again at the end or not
            ok.append(ws[:1] + ws[2:])
        ok += [x[:-1] for x in ok if len(x) > 2 and x[-1] == x[-2]]
        return g.split() in ok
    return g == w


def solve(p, system, tok, chat_tok, think, rounds, max_tokens, think_budget=0, history=()):
    shots = system.get("shots", []) if isinstance(system, dict) else []
    msgs = [{"role": "system", "content": system["system"] if isinstance(system, dict) else system}]
    for q, a in shots:  # earlier, finished turns: worked examples of the whole loop
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    msgs += [{"role": "user", "content": p["q"]}, *history]  # history: earlier attempts and the feedback on them
    prompt = chat_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=think)
    text, blocks, log, spans = "", [], [], []  # spans: [start, end) of each output block the kernel wrote
    if p.get("prefill"):  # a prebuilt library: its block and the kernel's output start the turn (prefilled, not generated)
        blocks.append(p["prefill"])
        text = p.get("lead", "") + f"```forth\n{p['prefill']}\n```\n"
        spans.append([len(text), len(text) + len(o := f"```output\n{kernel_output(blocks, tok)}\n```")])
        text += o + "\n"
    url = URLS[sum(map(ord, p["q"])) % len(URLS)]  # one replica per problem keeps its prefix cache warm
    while True:  # the token budget ends the turn; after --rounds + 1 runs, further blocks are not executed
        used = len(tok(text, add_special_tokens=False).input_ids)
        left = max_tokens - used  # --max-tokens is the whole turn's budget
        if left <= 0:
            break
        n = left
        if think and think_budget and "</think>" not in text:  # budget control (Nemotron cards): close the thinking
            k = text.rfind("```forth\n")                          # at the budget and let the answer continue; a
            budget = think_budget + GRACE * (k >= 0 and "```" not in text[k + 9:])  # block being written may finish
            if used >= budget:  # (still open then: the kernel drops it at </think>)
                text += "\n</think>\n\n"
                continue
            n = min(left, budget - used)
        try:
            piece, fin = complete(prompt + text, n, url)
        except (urllib.error.URLError, ConnectionError) as e:  # over the server's context, a dropped connection:
            log.append({"program": "", "output": f"server error {getattr(e, 'code', e)}"})  # end this problem only
            break
        text += piece
        if not piece.endswith(("```", "```\n")):
            if fin == "length" and n < left:  # stopped at the thinking budget, not at the end of the turn
                text += "" if piece else "\n</think>\n\n"  # a token with no text (special, partial byte) re-asks forever
                continue
            break  # finished (or out of tokens)
        end = text.rstrip("\n")  # ends with the fence
        opener = text.rfind("```forth\n")
        body = end[opener + len("```forth\n"):-len("```")] if opener >= 0 else None
        if body is None or "```" in body or "</think>" in body:  # an opening fence, a snippet's fence, or a block cut
                                                                 # by the end of the thinking (the kernel drops it): go on
            continue
        if len(log) > rounds:  # out of runs: let the model finish its answer
            continue
        blocks.append(body.strip())
        out = kernel_output(blocks, tok)  # text_vm resets the error at each block; earlier state carries
        spans.append([len(end) + 1, len(end) + 1 + len(o := f"```output\n{out}\n```")])
        text = end + "\n" + o + "\n"
        log.append({"program": body.strip(), "output": out})
    final = text.rsplit("</think>", 1)[-1]
    got = last_boxed(final)
    print(f"done: {same(got, str(p['a']), p)} {len(log)} block(s) want {p['a']} got {got} | {p['q'][:60]}", flush=True)
    return {"q": p["q"], "family": p.get("family"), "a": str(p["a"]), "got": got, "ok": same(got, str(p["a"]), p),
            "blocks": log, "tail": final[-300:], "text": text, "spans": spans, "tokens": len(tok(text, add_special_tokens=False).input_ids)}


def attempt(p, system, tok, chat_tok, think, rounds, max_tokens, think_budget, retries=0, judge=None):
    """Up to 1 + retries turns: after a wrong answer, a user turn says why and asks again (training/rubric.py REMEDY: the
    reason from code when it can tell (rubric.diagnose), else the judge's diagnosis; never the answer). The last turn's row, with every turn in "attempts". shortcut: each turn starts a fresh kernel (the
    compiled kernel would carry state across turns), fine while retries rewrite their programs anyway."""
    import rubric  # (it imports this module)
    history, tries = [], []
    for k in range(retries + 1):
        r = solve(p, system, tok, chat_tok, think, rounds, max_tokens, think_budget, history)
        tries.append(r)
        if r["ok"] or k == retries:
            break
        why = rubric.diagnose(r)
        two = norm(str(p["a"])) in ("True", "False") or "comparison.pair" in (p.get("family") or "")
        if why is None and judge and not two:
            ans = rubric.ask(judge, r)
            why = ans and ans.get("diagnosis", {}).get("choice")
        r["diagnosis"] = why
        if two and why in (None, "ignored_output"):  # "not correct" on a two-way question gives the answer away
            break
        r["feedback"] = (("" if two else "That answer is not correct. ") +  # two-way: the process reason only
                         (f"You {rubric.REMEDY[why][0]}. {rubric.REMEDY[why][1]} " if why else "") + "Try again.")
        history = [*history, {"role": "assistant", "content": r["text"].rsplit("</think>", 1)[-1].strip()},
                   {"role": "user", "content": r["feedback"]}]
    return dict(tries[-1], attempts=tries) if retries else tries[-1]


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("problems")
    a.add_argument("rubric")
    a.add_argument("--think", action="store_true")
    a.add_argument("--rounds", type=int, default=12)
    a.add_argument("--max-tokens", type=int, default=12000)
    a.add_argument("--think-budget", type=int, default=0, help="close the thinking after this many tokens (0: never)")
    a.add_argument("--samples", type=int, default=1, help="attempts per problem (sampling is stochastic)")
    a.add_argument("--retries", type=int, default=0, help="after a wrong answer, say why and ask again (up to N times)")
    a.add_argument("--judge", help="Mammoth URL for the retry diagnosis (training/rubric.py), e.g. http://localhost:8768/")
    a.add_argument("--workers", type=int, default=16, help="problems in flight (spread over the LIGHTNING_URL replicas)")
    a.add_argument("--out", required=True)
    args = a.parse_args()
    # one socket per rollout in flight: --workers 2080 passes the usual 1024 soft limit (the frontier run lost half)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, min(hard, args.workers + 1024)), hard))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ.get("LOOP_TOKENIZER", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"))  # local runs: the HF snapshot
    chat_tok = AutoTokenizer.from_pretrained(CHAT_TOK, trust_remote_code=True) if CHAT_TOK else tok
    sp.MAX_TOKENS = 10**6
    P = [p for p in json.load(open(args.problems)) for _ in range(args.samples)]
    system = json.load(open(args.rubric)) if args.rubric.endswith(".json") else open(args.rubric).read()  # {system, shots}
    # each finished rollout is appended to OUT.partial.jsonl: a preempted (interruptible) box loses none of them, and the
    # same command run again does only the samples not yet there
    part, lock = args.out + ".partial.jsonl", threading.Lock()
    prev = [json.loads(line) for line in open(part)] if os.path.exists(part) else []
    have, todo = collections.Counter(r["q"] for r in prev), []
    for p in P:
        if have[p["q"]]:
            have[p["q"]] -= 1
        else:
            todo.append(p)

    def run(p):
        try:
            r = attempt(p, system, tok, chat_tok, args.think, args.rounds, args.max_tokens, args.think_budget,
                        args.retries, args.judge)
        except Exception as e:  # noqa: BLE001 - one crashing rollout must not end the run; it isn't saved (rerun = retry)
            print(f"SKIPPED ({e!r}): {p['q'][:70]}", flush=True)
            return None
        with lock, open(part, "a") as f:
            f.write(json.dumps(r) + "\n")
        return r

    print(f"{len(prev)} rollouts already in {part}, {len(todo)} to run", flush=True)
    with cf.ThreadPoolExecutor(args.workers) as ex:
        res = prev + [r for r in ex.map(run, todo) if r]
    json.dump(res, open(args.out, "w"), indent=1)
    for r in res:
        print(f"{'OK  ' if r['ok'] else 'MISS'} {len(r['blocks'])} block(s) | {r['q'][:70]} | want {r['a'][:20]} got {str(r['got'])[:20]}")
        for i, b in enumerate(r["blocks"]):
            print(f"     [{i}] {b['program'].strip()[:110].replace(chr(10), ' / ')}  =>  {b['output'][:60]}")
    print("correct", sum(r["ok"] for r in res), "/", len(res))


assert norm("1,234") == "1234" and norm("3, 4") == "3 4" and norm("385958\\ 210641") == "385958 210641"
assert last_boxed("so \\boxed{\\frac{5}{33}}.") == "\\frac{5}{33}" and last_boxed("none") is None
assert same("-31,\\,-840017", "-840017, -31", {"family": "algebra.polynomial_roots"}) and not same("1, 2", "2, 1", {"family": "comparison.sort"})
_pr = {"family": "algebra.polynomial_roots_big"}
assert same("u = -5\\ \\text{(double root)},\\ u = -4", "-5, -4", _pr) and same("u = 2,\\quad u = -549", "-549, 2", _pr)
assert not same("-5 \\text{(double root)}", "-5, -4", _pr)
_eu = {"family": "L6.euclid_trace", "q": "Euclid on 69020 and 584217: print the second number, each nonzero remainder in order, and finally the gcd."}
_ew = "584217 69020 32057 4906 2621 2285 336 269 67 1 1"
assert all(same(x, _ew, _eu) for x in (_ew, "584217, 32057, 4906, 2621, 2285, 336, 269, 67, 1", "584217 69020 32057 4906 2621 2285 336 269 67 1"))
assert not any(same(x, _ew, _eu) for x in ("1", "584217 69020 32057 4906 2621 2285 336 269 66 1 1", "584217 69020 4906 2621 2285 336 269 67 1 1"))
assert not same("761244 232628 63360", "761244 528616 232628 63360", {"family": "L6.euclid_trace", "q": "Euclid on (761244, 2)"})
_ip, _ic = {"q": "Is 51353 a prime number?"}, {"q": "Is 51353 a composite number?"}
assert same("51353 \\text{ is composite}", "True", _ic) and same("51353 \\text{ is composite}", "False", _ip)
assert same("51353 is not prime", "True", _ic) and same("it is not a prime", "False", _ip) and same("isn't composite", "True", _ip)
assert same("is prime", "True", _ip) and same("is prime", "False", _ic) and same("not composite", "False", _ic)
assert not same("51353 is composite", "True", _ip) and not same("is not prime", "True", _ip) and not same("is prime", "True", _ic)
assert same("No, it is prime", "False", _ic) and not same("Yes, it is prime", "False", _ic) and not same("51353", "True", _ip)
assert not same("x is composite", "True", {"q": "Is 3 greater than 2?"})  # the property fallback only for prime / composite
assert norm("\\frac{5}{33}") == "5/33" and norm("-\\dfrac{1}{4}") == "-1/4" and norm("\\text{No}") == "False" and norm("True") == "True"

if __name__ == "__main__":
    main()
