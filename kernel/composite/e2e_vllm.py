"""Loop-style eval IN THE WEIGHTS (e2e_demo.py in vLLM): the composite (plugin vllm_forth.py, kernel inside) + a LoRA on
a problem set, loop_sim's visible-only think budget (Budget, as e2e_demo.Budget), the card's sampling, every kernel-owned
position checked against text_vm (e2e.check: 'exact', 'bad_runs'). All problems run concurrently (one request each,
seed = its position in IDX; IDX may repeat or be 'all'). In-process engine (Budget's per-request notes).
One command (NVFP4 base + any LoRA, the base reads only the visible stream), from the repo copy on the machine
and the composite checkpoint (kernel/composite/build.py OUT nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16; NVFP4 dir as in serve_forth.sh):
  FORTH_VISIBLE=1 CUDA_VISIBLE_DEVICES=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_DISABLE_COMPILE_CACHE=1 \
  PYTHONPATH=kernel/composite:kernel PROBLEMS=training/data/demo10.json SYSTEM=training/data/cards/reference.txt \
  VLLM_KW='{"moe_backend": "marlin", "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"}}' \
      venv-kernel/bin/python kernel/composite/e2e_vllm.py forth-lightning-nvfp4 OUT.json all ADAPTER_DIR \
      [MAX_ALL=200000]
  -> OUT.json: total {n, correct, exact, tokens, wall_s} + per problem got / ok / tokens / visible / hidden / turn_cut.
kernel/ must be the source the checkpoint's kernel was built from (the text_vm reference). Budgets: env THINK / GRACE /
TURN override e2e_demo's 4000 / 1000 / 8000 (visible tokens); MAX_SEQS (64) concurrent; TEMP (1.0; 0 = greedy).
"""

import json
import os
import re
import sys
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.v1.sample.logits_processor import LogitsProcessor

from e2e import OWNED, ROOT, check, last_boxed, same
from e2e_demo import GRACE, MAX_ALL, THINK, TURN

THINK, GRACE, TURN = (int(os.environ.get(k, v)) for k, v in (("THINK", THINK), ("GRACE", GRACE), ("TURN", TURN)))
from modeling_forth_lightning import HIDDEN_BELOW

NOTES = {}  # extra_args['key'] -> Budget's per-request record (in-process engine only)


class Budget(LogitsProcessor):
    """e2e_demo.Budget per request: counts VISIBLE tokens (a token is hidden when the step before it was kernel-owned
    and its id < HIDDEN_BELOW), forces '\\n</think>\\n\\n' at THINK (+GRACE inside an open ```forth block), eos at TURN;
    never on a kernel-owned step (max logit >= OWNED / 2)."""

    def __init__(self, vllm_config, device, is_pin_memory):
        self.tok = AutoTokenizer.from_pretrained(vllm_config.model_config.tokenizer)
        self.close = self.tok("\n</think>\n\n", add_special_tokens=False).input_ids
        self.reqs = {}

    def is_argmax_invariant(self):
        return False

    def update_state(self, bu):
        if bu is None:
            return
        for i in bu.removed:
            self.reqs.pop(i, None)
        for i, sp, _, out in bu.added:
            key = (sp.extra_args or {}).get("key")
            if key is None:
                self.reqs.pop(i, None)
                continue
            NOTES[key] = {"vis": [], "own": False, "seen": 0, "q": [], "done": False, "at": None, "cut": False,
                          "reserved_mass": 0.0, "reserved_max": 0.0}
            self.reqs[i] = (NOTES[key], out)
        for a, b, kind in bu.moved:
            ra, rb = self.reqs.pop(a, None), self.reqs.pop(b, None)
            if ra is not None:
                self.reqs[b] = ra
            if rb is not None and kind.name == "SWAP":
                self.reqs[a] = rb

    def apply(self, logits):
        if not self.reqs:
            return logits
        mx = logits.max(-1).values.tolist()
        # probability the model itself puts on reserved ids 18..999 (a superset of the kernel's hidden ids): the
        # expected number of such tokens it would sample (temperature 1); the id rule needs this to be ~0
        res = (logits[:, 18:1000].logsumexp(-1) - logits.logsumexp(-1)).exp().tolist()
        for i, (s, out) in self.reqs.items():
            for t in out[s["seen"]:]:  # the token sampled last step
                if not (s["own"] and t < HIDDEN_BELOW):
                    s["vis"].append(t)
            s["seen"] = len(out)
            s["own"] = mx[i] >= OWNED / 2
            if s["own"]:
                continue
            s["reserved_mass"] += res[i]
            s["reserved_max"] = max(s["reserved_max"], res[i])
            if len(s["vis"]) >= TURN:
                s["cut"], s["q"] = True, [self.tok.eos_token_id]
            elif not s["q"] and not s["done"] and len(s["vis"]) >= THINK:
                text = self.tok.decode(s["vis"])
                if "</think>" in text:
                    s["done"] = True
                else:
                    k = text.rfind("```forth\n")
                    if len(s["vis"]) >= THINK + GRACE * (k >= 0 and "```" not in text[k + 9:]):
                        s["q"], s["at"] = list(self.close), len(s["vis"])
            if s["q"]:
                t = s["q"].pop(0)
                logits[i] = -float("inf")
                logits[i, t] = 0.0
                s["done"] = s["done"] or not s["q"]
        return logits


def main(model, outp, only, adapter, max_new=MAX_ALL):
    kw = dict(max_model_len=int(max_new) + 4096, enable_prefix_caching=False, gpu_memory_utilization=0.85,
              max_num_seqs=int(os.environ.get("MAX_SEQS", 64)), enable_flashinfer_autotune=False,
              logprobs_mode="raw_logits", async_scheduling=False,
              enable_lora=True, max_lora_rank=64, logits_processors=[Budget]) | json.loads(os.environ.get("VLLM_KW", "{}"))
    llm = LLM(model, **kw)
    tok = AutoTokenizer.from_pretrained(model)
    sysm = open(ROOT / os.environ.get("SYSTEM", "training/data/cards/reference.txt")).read()
    P = json.load(open(ROOT / os.environ.get("PROBLEMS", "training/data/demo10.json")))
    idx = list(range(len(P))) if only == "all" else list(map(int, only.split(",")))
    prompts = []
    for i in idx:
        msgs = [{"role": "system", "content": sysm}, {"role": "user", "content": P[i]["q"]}]
        prompts.append(tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True),
                           add_special_tokens=False).input_ids)
    sps = [SamplingParams(max_tokens=int(max_new), temperature=float(os.environ.get("TEMP", 1.0)), top_p=0.95, seed=j, logprobs=1,
                          extra_args={"key": j}) for j, i in enumerate(idx)]  # IDX may repeat (load tests)
    t0 = time.time()
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts], sps, use_tqdm=False,
                        lora_request=LoRARequest("a", 1, adapter) if adapter else None)
    wall = time.time() - t0
    R = []
    for j, (i, prompt, o) in enumerate(zip(idx, prompts, outs)):
        g, p, s = list(o.outputs[0].token_ids), P[i], NOTES[j]
        claimed = {j for j, d in enumerate(o.outputs[0].logprobs) if max(x.logprob for x in d.values()) >= OWNED / 2}
        text = tok.decode(g, skip_special_tokens=True)
        got = last_boxed(text.rsplit("</think>", 1)[-1])
        hidden = sum(j in claimed and t < HIDDEN_BELOW for j, t in enumerate(g))
        m = o.metrics
        span = (m.last_token_ts - m.first_token_ts) if m and getattr(m, "last_token_ts", 0) else None
        r = {"idx": i, "seed": j, "decode_tok_s": round(len(g) / span, 1) if span else None, "family": p["family"],
             "a": str(p["a"]), "got": got, "ok": same(got, str(p["a"]), p),
             "tokens": len(g), "visible_tokens": len(g) - hidden, "hidden_tokens": hidden, "turn_cut": s["cut"],
             "safety_cut": len(g) >= int(max_new), "forced_think_close_at": s["at"],
             "reserved_mass": s["reserved_mass"], "reserved_max": s["reserved_max"],
             "programs": re.findall(r"```forth\n(.*?)```", text, re.S)}
        r |= check(tok, prompt, g, claimed)
        print(json.dumps({k: v for k, v in r.items() if k not in ("bad_runs",)})[:2000], flush=True)
        r["text"], r["ids"], r["claimed"] = text, g, sorted(claimed)
        R.append(r)
    tot = {"n": len(R), "correct": sum(r["ok"] for r in R), "exact": sum(r["exact"] for r in R),
           "tokens": sum(r["tokens"] for r in R), "wall_s": round(wall, 1)}
    tot["tok_s_aggregate"] = round(tot["tokens"] / wall, 1)
    print("TOTAL", json.dumps(tot), flush=True)
    json.dump({"info": {"model": model, "adapter": adapter, "llm_kw": {k: str(v) for k, v in kw.items()}},
               "total": tot, "rows": R}, open(outp, "w"), indent=1)
    os._exit(0)  # vLLM 0.29's shutdown can hang after the results are written


if __name__ == "__main__":
    main(*sys.argv[1:])
