"""End to end: the HF composite (Lightning + compiled kernel) answers math problems on its own. The model writes ```forth
blocks, the in-network kernel writes the ```output blocks; every kernel-owned position is checked against text_vm run on
the model-written tokens (text_vm.stream). Batch 1, the card's sampling, chat template + thinking.
  [E2E_ONLY=7,4] CUDA_VISIBLE_DEVICES=3 python kernel/composite/e2e.py CKPT OUT.json [SYSTEM.txt] [MAX_NEW]
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "training"), str(ROOT / "kernel"), str(ROOT)]
import text_vm as TV  # noqa: E402
from loop_sim import last_boxed, same  # noqa: E402

OWNED = 2.0**16
PICK = {"training/data/dm_frontier.json": {"arithmetic.mul": 2, "arithmetic.add_sub_multiple": 2,
                                       "polynomials.evaluate": 3, "numbers.div_remainder": 2,
                                       "arithmetic.mul_div_multiple": 1},
        "training/data/demo_set.json": {"DMbig.div_remainder": 3, "DMbig.lcm": 2}}


def problems():
    out = []
    for f, fams in PICK.items():
        P = json.load(open(ROOT / f))
        for fam, n in fams.items():  # integer answers only (fractions need gcd, outside piece 2b)
            out += [p for p in P if p["family"] == fam and re.fullmatch(r"-?\d+", str(p["a"]))][:n]
    return out


class Claims(LogitsProcessor):
    """Records the decode steps the kernel owned (its logits are x 2^16)."""

    def __init__(self):
        self.steps, self.step = [], 0

    def __call__(self, input_ids, scores):
        if float(scores[0].max()) >= OWNED / 2:
            self.steps.append(self.step)
        self.step += 1
        return scores


class Timer:
    """Synced wall time inside a module's forward (per-step cost split: base vs kernel)."""

    def __init__(self, mod):
        self.s, self.n, f = 0.0, 0, mod.forward

        def fwd(*a, **k):
            torch.cuda.synchronize()
            t = time.time()
            r = f(*a, **k)
            torch.cuda.synchronize()
            self.s, self.n = self.s + time.time() - t, self.n + 1
            return r
        mod.forward = fwd


def check(tok, prompt, gen, claimed):
    """Reference: text_vm on prompt + the model-written generated tokens. Kernel output text_vm puts inside the prompt
    (the system prompt mentions ```forth ... ```output) cannot be in the composite's stream: dropped and counted."""
    model = [t for i, t in enumerate(gen) if i not in claimed]
    out, owned, _ = TV.stream(prompt + model, tok)
    own = set(owned)
    pend, seen = 0, 0  # pend: the position in out right after the prompt's last token
    while seen < len(prompt):
        seen, pend = seen + (pend not in own), pend + 1
    in_prompt = [o for o in owned if o < pend]
    ref = [t for i, t in enumerate(out) if not (i in own and i < pend)][len(prompt):]
    ref_own = {o - pend for o in owned if o >= pend}
    n = min(len(ref), len(gen))
    bad = next((i for i in range(n) if ref[i] != gen[i] or ((i in ref_own) != (i in claimed))), None)
    blocks = lambda ids, ow: re.findall(r"```output\n(.*?)\n```", tok.decode([ids[i] for i in sorted(ow) if i < len(ids)],
                                                                         skip_special_tokens=True), re.S)
    r = {"kernel_tokens": len(claimed), "ref_kernel_tokens": len([o for o in ref_own if o < len(gen)]),
         "ref_prompt_kernel_tokens": len(in_prompt), "exact": bad is None and len(ref) >= len(gen),
         "ref_longer_by": len(ref) - len(gen), "blocks_kernel": blocks(gen, claimed), "blocks_ref": blocks(ref, ref_own)}
    r["blocks_equal"] = r["blocks_kernel"] == r["blocks_ref"]
    # per kernel run (tokens it emitted after the n-th model token), so one early divergence doesn't hide later ones
    gk, gr = groups(prompt + gen, {len(prompt) + i for i in claimed}), groups(out, own)
    runs = []
    for n in sorted(set(gk) | set(gr)):
        if n < len(prompt) or n == len(prompt) + len(model) and r["ref_longer_by"] > 0:
            continue  # inside the prompt, or a run the token cap cut
        pre = tok.decode(model[:n - len(prompt)])
        k = pre.rfind("```forth")
        real = k >= 0 and pre[k + 8:k + 9] == "\n"  # a real block ('```forth' + newline), else a fence opened in prose
        runs.append({"n": n, "real": real, "ok": gk.get(n) == gr.get(n), "got": tok.decode(gk.get(n, [])),
                     "want": tok.decode(gr.get(n, [])), "context": pre[-60:]})
    for kind in (True, False):
        rs = [x for x in runs if x["real"] == kind]
        r["runs_real" if kind else "runs_prose"] = [len(rs), sum(not x["ok"] for x in rs)]
    r["bad_runs"] = [x for x in runs if not x["ok"]][:12]
    if bad is not None:
        r["mismatch"] = {"pos": bad, "want": ref[bad], "got": gen[bad], "want_kernel": bad in ref_own,
                         "got_kernel": bad in claimed, "want_s": tok.decode([ref[bad]]), "got_s": tok.decode([gen[bad]]),
                         "context": tok.decode(gen[max(0, bad - 40):bad])}
    return r


def groups(seq, own):
    """{n: the kernel tokens right after the n-th non-kernel token}"""
    g, n = {}, 0
    for i, t in enumerate(seq):
        if i in own:
            g.setdefault(n, []).append(t)
        else:
            n += 1
    return g


@torch.no_grad()
def main(ckpt, outp, system=str(ROOT / "training/data/cards/oneline.txt"), max_new=3000):
    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True, dtype=torch.bfloat16,
                                             device_map="cuda").eval()
    tok = AutoTokenizer.from_pretrained(m.config.base_model)
    tb, tk = Timer(m.base.model), Timer(m.kernel.model)
    print("loaded", round(time.time() - t0, 1), "s", flush=True)
    sysm, R = open(system).read(), []
    only = os.environ.get("E2E_ONLY")  # comma list of problem indices (short checked runs first)
    for i, p in enumerate(problems()):
        if only and str(i) not in only.split(","):
            continue
        msgs = [{"role": "system", "content": sysm}, {"role": "user", "content": p["q"]}]
        prompt = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True),
                     add_special_tokens=False).input_ids
        x = torch.tensor([prompt], device="cuda")
        c, b0, k0 = Claims(), (tb.s, tb.n), (tk.s, tk.n)
        torch.manual_seed(i)
        t1 = time.time()
        g = m.generate(x, attention_mask=torch.ones_like(x), logits_processor=LogitsProcessorList([c]),
                       max_new_tokens=int(max_new), do_sample=True, temperature=1.0, top_p=0.95,
                       pad_token_id=0)[0, len(prompt):].tolist()
        dt = time.time() - t1
        text = tok.decode(g, skip_special_tokens=True)
        got = last_boxed(text.rsplit("</think>", 1)[-1])
        r = {"family": p["family"], "q": p["q"], "a": str(p["a"]), "got": got, "ok": same(got, str(p["a"]), p),
             "tokens": len(g), "cut": len(g) >= int(max_new), "s": round(dt, 1), "tok_s": round(len(g) / dt, 1),
             "base_s": round(tb.s - b0[0], 1), "base_calls": tb.n - b0[1], "kernel_s": round(tk.s - k0[0], 1),
             "programs": re.findall(r"```forth\n(.*?)```", text, re.S)}
        r |= check(tok, prompt, g, set(c.steps))
        r["text"], r["ids"], r["claimed"] = text, g, c.steps
        R.append(r)
        print(json.dumps({k: v for k, v in r.items() if k not in ("text", "ids", "claimed")})[:3000], flush=True)
        json.dump(R, open(outp, "w"), indent=1)
    tot = {"n": len(R), "correct": sum(r["ok"] for r in R), "exact": sum(r["exact"] for r in R),
           "blocks_equal": sum(r["blocks_equal"] for r in R), "tokens": sum(r["tokens"] for r in R),
           "kernel_tokens": sum(r["kernel_tokens"] for r in R), "s": round(sum(r["s"] for r in R), 1),
           "base_s": round(tb.s, 1), "kernel_s": round(tk.s, 1)}
    tot["tok_s"] = round(tot["tokens"] / tot["s"], 1)
    print("TOTAL", json.dumps(tot), flush=True)
    json.dump({"total": tot, "rows": R}, open(outp, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
