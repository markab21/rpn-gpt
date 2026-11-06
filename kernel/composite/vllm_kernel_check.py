"""Kernel exactness through vLLM (the composite plugin vllm_forth.py, or the kernel alone as a stock NemotronH): every
position of sp_test.py's text cases (+ real programs / turns files, as kernel_test.yml) against text_vm.
  prefill: prompt_logprobs over the whole stream, chunked prefill in 2048-token chunks (chunk edges land inside
           kernel-owned runs; prompt logits over a chunk are 4 GB at 2048 x 131072 x fp32);
  decode:  one request per stream, prompt = its first 2 tokens, every later token decoded with the stream's token forced
           (Force), so every position is a cached decode step (sp_test's 'cached').
Raw logits (logprobs_mode raw_logits, top 2), S = 2^16 in the composite (kernel-owned logits are x OWNED), 1 alone:
  kernel-owned position: top-1 = text_vm's token and top-1 - top-2 >= 0.5 S (sp_test.MIN_MARGIN)
  model position (also after the last token): top-1 < mine S
  CUDA_VISIBLE_DEVICES=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTHONPATH=composite:kernel venv-kernel/bin/python \
      kernel/composite/vllm_kernel_check.py MODEL OUT.json prefill|decode|both|cache [PROGRAMS_OR_TURNS.json ...]
  cache: with VLLM_KW enable_prefix_caching + mamba_cache_mode align: warm half of 3/4 of each stream, then decode-check
         the last quarter twice (prompt sharing only a prefix with the cache; then an exact repeat), num_cached_tokens kept
env CASES=regex (case names), LIMIT=n (first n cases), VLLM_KW='{json LLM kwargs}' (e.g. enforce_eager, NVFP4 flags).
"""

import json
import os
import re
import sys
import time

import psutil
from vllm import LLM, SamplingParams
from vllm.v1.sample.logits_processor import LogitsProcessor

import sp_test as T

OWNED = 2.0**16


class Force(LogitsProcessor):
    """Forces each request's extra_args['force'][k] at its k-th generated token (none past the end). Raw-logit logprobs
    are taken before this runs, so they show the model's own logits at every step."""

    def __init__(self, vllm_config, device, is_pin_memory):
        self.reqs = {}

    def is_argmax_invariant(self):
        return False

    def update_state(self, bu):
        if bu is None:
            return
        for i in bu.removed:
            self.reqs.pop(i, None)
        for i, sp, _, out in bu.added:
            f = (sp.extra_args or {}).get("force")
            self.reqs.pop(i, None) if f is None else self.reqs.__setitem__(i, (f, out))
        for a, b, kind in bu.moved:
            ra, rb = self.reqs.pop(a, None), self.reqs.pop(b, None)
            if ra is not None:
                self.reqs[b] = ra
            if rb is not None and kind.name == "SWAP":
                self.reqs[a] = rb

    def apply(self, logits):
        for i, (f, out) in self.reqs.items():
            k = len(out)
            if k < len(f):
                logits[i] = -float("inf")
                logits[i, f[k]] = 0.0
        return logits


def top2(d):
    v = {x.rank: (t, x.logprob) for t, x in d.items()}
    return v[1][0], v[1][1], v.get(2, (None, -float("inf")))[1]


def score(rows, ids, own, S, mine, tag):
    """rows[j] = top-2 dict of the logits predicting ids[j] (j = len(ids): after the last token)."""
    wrong, low, loud, margin = [], [], [], float("inf")
    for j, d in rows:
        t, v1, v2 = top2(d)
        if j in own:
            margin = min(margin, (v1 - v2) / S)
            if t != ids[j]:
                wrong.append((j, ids[j], t))
            elif v1 - v2 < T.MIN_MARGIN * S:
                low.append((j, (v1 - v2) / S))
        elif v1 >= mine * S:
            loud.append((j, t, v1 / S))
    return {f"{tag}_wrong": len(wrong), f"{tag}_low_margin": len(low), f"{tag}_loud": len(loud),
            f"{tag}_min_margin": margin, f"{tag}_first": (wrong + low + loud)[:4]}


def cases(tok, files):
    cs = T.text_cases(tok)
    for f in files:  # real programs / turns: text_cases' 'real_' entries for that file
        stem = os.path.basename(f).rsplit(".", 1)[0]
        cs |= {f"{stem}_{k[5:]}": v for k, v in T.text_cases(tok, f).items() if k.startswith("real_")}
    pat = re.compile(os.environ.get("CASES", ".*"))
    cs = {k: v for k, v in cs.items() if pat.search(k)}
    return dict(list(cs.items())[:int(os.environ.get("LIMIT", 10**9))])


if __name__ == "__main__":
    model, outp, mode, files = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    cfg = json.load(open(os.path.join(model, "config.json")))
    comp = cfg.get("model_type") == "forth_lightning"
    S, mine = (OWNED, cfg.get("mine", 0.5)) if comp else (1.0, 0.5)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    cs = {k: (list(ids), set(own)) for k, (ids, own) in cases(tok, files).items()}
    n = max(len(ids) for ids, _ in cs.values())
    kw = dict(max_model_len=n + 16, enable_prefix_caching=False, gpu_memory_utilization=0.8, max_num_seqs=64,
              max_num_batched_tokens=2048, enable_flashinfer_autotune=False, logprobs_mode="raw_logits",
              async_scheduling=False, logits_processors=["vllm_kernel_check:Force"]) | json.loads(
        os.environ.get("VLLM_KW", "{}"))
    t0 = time.time()
    llm = LLM(model, **kw)
    R = {"model": model, "composite": comp, "llm_kw": {k: str(v) for k, v in kw.items()}, "load_s": round(time.time() - t0),
         "cases": {}}
    names = list(cs)
    for k in names:
        R["cases"][k] = {"tokens": len(cs[k][0]), "kernel_tokens": len(cs[k][1])}
    if mode in ("prefill", "both"):
        t1 = time.time()
        outs = llm.generate([{"prompt_token_ids": cs[k][0]} for k in names],
                            SamplingParams(max_tokens=1, prompt_logprobs=2, logprobs=2, temperature=0), use_tqdm=False)
        for k, o in zip(names, outs):
            ids, own = cs[k]
            rows = [(j, o.prompt_logprobs[j]) for j in range(1, len(ids))] + [(len(ids), o.outputs[0].logprobs[0])]
            R["cases"][k] |= score(rows, ids, own, S, mine, "prefill")
        R["prefill_s"] = round(time.time() - t1, 1)
    if mode in ("decode", "both"):
        t1 = time.time()
        # prompt = the first 2 tokens: vLLM 0.29 runs a 1-token first prefill as a FULL-cudagraph decode, which reads
        # the request's uninitialised Mamba state instead of zeros (the kernel breaks; real prompts are longer)
        outs = llm.generate([{"prompt_token_ids": cs[k][0][:2]} for k in names],
                            [SamplingParams(max_tokens=len(cs[k][0]) - 1, logprobs=2, temperature=0, ignore_eos=True,
                                            extra_args={"force": cs[k][0][2:]}) for k in names], use_tqdm=False)
        for k, o in zip(names, outs):
            ids, own = cs[k]
            got = list(o.outputs[0].token_ids)
            assert got[:len(ids) - 2] == ids[2:], (k, "forcing failed")
            rows = [(j + 2, d) for j, d in enumerate(o.outputs[0].logprobs)]
            R["cases"][k] |= score(rows, ids, own, S, mine, "decode")
        R["decode_s"] = round(time.time() - t1, 1)
    if mode == "cache":  # prefix caching on (VLLM_KW): decode-check the last quarter after a cached prompt
        t1 = time.time()
        cut = {k: len(cs[k][0]) * 3 // 4 for k in names}
        llm.generate([{"prompt_token_ids": cs[k][0][:cut[k] // 2]} for k in names], SamplingParams(max_tokens=1),
                     use_tqdm=False)  # warm: half of each prompt
        for tag in ("cache_prefix", "cache_repeat"):  # prompt shares only a prefix with the cache; then exactly cached
            outs = llm.generate([{"prompt_token_ids": cs[k][0][:cut[k]]} for k in names],
                                [SamplingParams(max_tokens=len(cs[k][0]) - cut[k] + 1, logprobs=2, temperature=0,
                                                ignore_eos=True, extra_args={"force": cs[k][0][cut[k]:]})
                                 for k in names], use_tqdm=False)
            for k, o in zip(names, outs):
                ids, own = cs[k]
                assert list(o.outputs[0].token_ids)[:len(ids) - cut[k]] == ids[cut[k]:], (k, "forcing failed")
                rows = [(cut[k] + j, d) for j, d in enumerate(o.outputs[0].logprobs)]
                R["cases"][k] |= score(rows, ids, own, S, mine, tag) | {f"{tag}_cached_tokens": o.num_cached_tokens}
        R["cache_s"] = round(time.time() - t1, 1)
    bad = [k for k, r in R["cases"].items() if any(v for f, v in r.items() if f.endswith(("_wrong", "_low_margin", "_loud")))]
    R["verdict"] = f"{len(names) - len(bad)}/{len(names)} cases exact ({mode})"
    R["bad"] = {k: {f: v for f, v in R["cases"][k].items() if v} for k in bad[:20]}
    json.dump(R, open(outp, "w"), indent=1)
    print(R["verdict"], json.dumps(R["bad"])[:3000], flush=True)
    for c in psutil.Process().children(recursive=True):  # vLLM 0.29's shutdown can hang (as vllm_composite_test.py)
        c.kill()
    os._exit(0)
