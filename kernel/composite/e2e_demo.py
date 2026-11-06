"""e2e.py on the deliverable setup: demo10.json (hard), reference.txt, loop_sim's think budget (a forced '\n</think>\n\n'
once THINK generated tokens are spent with no </think>; +GRACE while a ```forth block is open; never on a kernel-owned
step), optionally the s1_cold LoRA on composite.base (eval_sp.attach_lora). Batch 1, card sampling.
  [E2E_SET=training/data/demo_set.json E2E_SYS=training/data/cards/reference_v2.txt E2E_THINK=8000 E2E_TURN=16000] \
  CUDA_VISIBLE_DEVICES=7 python kernel/composite/e2e_demo.py CKPT OUT.json IDX[,IDX] [ADAPTER_DIR] [MAX_ALL=200000]
Resumes: problems already in OUT.json are skipped (written after each problem, atomically).
Budgets count visible tokens only (Budget); hidden kernel steps are free, as in training/loop_sim.py.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

from modeling_forth_lightning import HIDDEN_BELOW
from e2e import OWNED, ROOT, Claims, Timer, check, last_boxed, same

THINK, GRACE, TURN = int(os.environ.get("E2E_THINK", 4000)), 1000, int(os.environ.get("E2E_TURN", 8000))  # training/loop_sim.py --think-budget 4000 (the eval default), its GRACE, --max-tokens 8000 (visible)
MAX_ALL, MAX_S = 200_000, 3600  # safety caps per problem: all tokens (hidden included), wall seconds


class Budget(LogitsProcessor):
    """loop_sim's budgets on the VISIBLE transcript only (model tokens + visible output-block tokens; the kernel's hidden
    ids < HIDDEN_BELOW never count, as in training where the machine is external): the think close at THINK (+GRACE
    while a ```forth block is open), and the turn cap TURN (forced eos). Never on a kernel-owned step. Also splits the
    wall time between calls by the kind of the step's input token (hidden: the base is skipped; else model / visible)."""

    def __init__(self, tok, n0, tb, tk):
        self.tok, self.n0, self.tb, self.tk = tok, n0, tb, tk
        self.close = tok("\n</think>\n\n", add_special_tokens=False).input_ids
        self.q, self.done, self.at, self.cut, self.vis, self.own, self.last = [], False, None, False, [], False, None
        self.kind = {k: {"steps": 0, "s": 0.0, "base_s": 0.0, "kernel_s": 0.0} for k in ("hidden", "visible")}

    def __call__(self, ids, scores):
        now, snap = time.time(), (self.tb.s, self.tk.s)
        if ids.shape[1] > self.n0:  # the token sampled last step (its forward ran between the two calls)
            t = int(ids[0, -1])
            hidden = self.own and t < HIDDEN_BELOW
            if not hidden:
                self.vis.append(t)
            d = self.kind["hidden" if hidden else "visible"]
            d["steps"] += 1
            d["s"] += now - self.last[0]
            d["base_s"] += snap[0] - self.last[1][0]
            d["kernel_s"] += snap[1] - self.last[1][1]
        self.last = (now, snap)
        self.own = float(scores[0].max()) >= OWNED / 2
        if self.own:
            return scores
        if len(self.vis) >= TURN:
            self.cut = True
            self.q = [self.tok.eos_token_id]
        elif not self.q and not self.done and len(self.vis) >= THINK:
            text = self.tok.decode(self.vis)
            if "</think>" in text:
                self.done = True
            else:
                k = text.rfind("```forth\n")
                if len(self.vis) >= THINK + GRACE * (k >= 0 and "```" not in text[k + 9:]):
                    self.q, self.at = list(self.close), len(self.vis)
        if self.q:
            t = self.q.pop(0)
            scores[:] = -float("inf")
            scores[0, t] = 0.0
            self.done = self.done or not self.q
        return scores


def lora_hooks(base, adapter):
    sys.path.insert(0, str(ROOT / "training"))
    from eval_sp import attach_lora
    lora = attach_lora(base, adapter)
    names = {id(m): n for n, m in base.named_modules()}
    calls = {}
    for m in lora:
        m.register_forward_hook(lambda mod, a, o: calls.__setitem__(names[id(mod)], calls.get(names[id(mod)], 0) + 1))
    return lora, calls


@torch.no_grad()
def main(ckpt, outp, only, adapter=None, max_new=MAX_ALL):
    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda").eval()
    tok = AutoTokenizer.from_pretrained(m.config.base_model)
    info = {"ckpt": ckpt, "adapter": adapter}
    if adapter:
        lora, calls = lora_hooks(m.base, adapter)
        x = torch.tensor([tok("Compute 17 * 23 with Forth.", add_special_tokens=False).input_ids], device="cuda")
        on = m.base(x).logits.float()
        for l in lora:
            l.enable_adapters(False)
        off = m.base(x).logits.float()
        for l in lora:
            l.enable_adapters(True)
        ids = {id(l) for l in lora}
        info |= {"lora_modules": len(lora), "lora_called": len(calls), "lora_not_called": sorted(
            n for n, mm in m.base.named_modules() if id(mm) in ids and n not in calls),
            "logit_maxdiff_on_off": float((on - off).abs().max())}
        print("LORA", json.dumps(info), flush=True)
    tb, tk = Timer(m.base.model), Timer(m.kernel.model)
    print("loaded", round(time.time() - t0, 1), "s", flush=True)
    sysm = open(ROOT / os.environ.get("E2E_SYS", "training/data/cards/reference.txt")).read()
    P = json.load(open(ROOT / os.environ.get("E2E_SET", "training/data/demo10.json")))
    R = json.load(open(outp))["rows"] if os.path.exists(outp) else []
    info |= {"set": os.environ.get("E2E_SET", "training/data/demo10.json"), "sys": os.environ.get("E2E_SYS", "reference.txt"),
             "think": THINK, "turn": TURN}
    for i in map(int, only.split(",")):
        if i in {r["idx"] for r in R}:
            continue
        p = P[i]
        msgs = [{"role": "system", "content": sysm}, {"role": "user", "content": p["q"]}]
        prompt = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True),
                     add_special_tokens=False).input_ids
        x = torch.tensor([prompt], device="cuda")
        c, bud, b0, k0 = Claims(), Budget(tok, len(prompt), tb, tk), (tb.s, tb.n), (tk.s, tk.n)
        torch.manual_seed(i)
        t1 = time.time()
        g = m.generate(x, attention_mask=torch.ones_like(x), logits_processor=LogitsProcessorList([c, bud]),
                       max_new_tokens=int(max_new), max_time=MAX_S, do_sample=True, temperature=1.0, top_p=0.95,
                       pad_token_id=0)[0, len(prompt):].tolist()
        dt = time.time() - t1
        text = tok.decode(g, skip_special_tokens=True)
        got = last_boxed(text.rsplit("</think>", 1)[-1])
        own = set(c.steps)
        marked = "".join(("⟦" + tok.decode([t]) + "⟧") if j in own else tok.decode([t]) for j, t in enumerate(g))
        r = {"idx": i, "family": p["family"], "q": p["q"], "a": str(p["a"]), "got": got, "ok": same(got, str(p["a"]), p),
             "tokens": len(g), "visible_tokens": len(bud.vis), "hidden_tokens": len(g) - len(bud.vis), "turn_cut": bud.cut,
             "safety_cut": len(g) >= int(max_new) or dt >= MAX_S, "forced_think_close_at": bud.at,
             "by_kind": {k: v | {"tok_s": round(v["steps"] / max(v["s"], 1e-9), 1)} for k, v in bud.kind.items()}, "s": round(dt, 1),
             "tok_s": round(len(g) / dt, 1), "base_s": round(tb.s - b0[0], 1), "kernel_s": round(tk.s - k0[0], 1),
             "programs": re.findall(r"```forth\n(.*?)```", text, re.S)}
        r |= check(tok, prompt, g, own)
        r["text"], r["marked"], r["ids"], r["claimed"] = text, marked, g, c.steps
        R.append(r)
        print(json.dumps({k: v for k, v in r.items() if k not in ("text", "marked", "ids", "claimed")})[:4000], flush=True)
        json.dump({"info": info, "rows": R}, open(outp + ".tmp", "w"), indent=1)
        os.replace(outp + ".tmp", outp)  # a preemption mid-write never corrupts the resume file


if __name__ == "__main__":
    main(*sys.argv[1:])
