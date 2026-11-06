"""Visible-stream proof for the vLLM composite (FORTH_VISIBLE=1): on kernel test streams (vllm_kernel_check.cases),
  composite: the composite on the FULL stream (hidden kernel tokens included), batched (max_num_seqs 64, streams of
             mixed lengths: requests finish and new ones join mid-run, so b_last rows are reused), prefill + forced
             decode: kernel positions scored as vllm_kernel_check; at every model position the raw logit of the
             stream's next token is kept;
  base:      stock Lightning on the VISIBLE stream only (prefill), the same logit at the same model positions;
  compare:   |composite - base| per position, before / after the stream's first hidden token (before = the noise
             floor of two batched bf16 runs). With FORTH_VISIBLE=0 the same compare shows what reading hidden costs.
  CUDA_VISIBLE_DEVICES=0 [FORTH_VISIBLE=1] VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTHONPATH=composite:kernel \
      venv-kernel/bin/python kernel/composite/vllm_visible_check.py composite|base MODEL OUT.json [FILES...]
  python kernel/composite/vllm_visible_check.py compare COMPOSITE.json BASE.json
env CASES / LIMIT / VLLM_KW as vllm_kernel_check.
"""

import json
import os
import sys

HIDDEN_BELOW = 1000


def hidden_mask(ids, own):
    return [j in own and t < HIDDEN_BELOW for j, t in enumerate(ids)]


def run(kind, model, outp, files):
    import psutil
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import vllm_kernel_check as C
    tok = AutoTokenizer.from_pretrained(model)
    cs = {k: (list(ids), set(own)) for k, (ids, own) in C.cases(tok, files).items()}
    seqs = {}
    for k, (ids, own) in cs.items():
        h = hidden_mask(ids, own)
        seqs[k] = ids if kind == "composite" else [t for t, x in zip(ids, h) if not x]
    kw = dict(max_model_len=max(map(len, seqs.values())) + 16, enable_prefix_caching=False, gpu_memory_utilization=0.8,
              max_num_seqs=64, max_num_batched_tokens=2048, enable_flashinfer_autotune=False,
              logprobs_mode="raw_logits", async_scheduling=False, logits_processors=["vllm_kernel_check:Force"]) | \
        json.loads(os.environ.get("VLLM_KW", "{}"))
    llm = LLM(model, **kw)
    names = list(cs)
    R = {"kind": kind, "model": model, "visible_env": os.environ.get("FORTH_VISIBLE", "0"), "cases": {}}
    outs = llm.generate([{"prompt_token_ids": seqs[k]} for k in names],
                        SamplingParams(max_tokens=1, prompt_logprobs=2, logprobs=2, temperature=0), use_tqdm=False)
    pre = {k: o for k, o in zip(names, outs)}
    dec = {}
    if kind == "composite":
        outs = llm.generate([{"prompt_token_ids": cs[k][0][:2]} for k in names],
                            [SamplingParams(max_tokens=len(cs[k][0]) - 1, logprobs=2, temperature=0, ignore_eos=True,
                                            extra_args={"force": cs[k][0][2:]}) for k in names], use_tqdm=False)
        dec = {k: o for k, o in zip(names, outs)}
    for k in names:
        ids, own = cs[k]
        h = hidden_mask(ids, own)
        vis_at = {}  # full position -> position in the visible stream
        n = 0
        for j, x in enumerate(h):
            if not x:
                vis_at[j] = n
                n += 1
        seen, first_hidden = False, None
        model_pos = []  # full positions j >= 1 whose token the model wrote (not kernel-owned)
        for j in range(len(ids)):
            if h[j] and first_hidden is None:
                first_hidden = j
            if j >= 1 and j not in own:
                model_pos.append(j)
        r = {"tokens": len(ids), "hidden": sum(h), "first_hidden": first_hidden}
        if kind == "composite":
            pl = pre[k].prompt_logprobs
            r |= C.score([(j, pl[j]) for j in range(1, len(ids))], ids, own, C.OWNED, 0.5, "prefill")
            dl = dec[k].outputs[0].logprobs
            r |= C.score([(j + 2, d) for j, d in enumerate(dl)], ids, own, C.OWNED, 0.5, "decode")
            r["model"] = {j: [pl[j][ids[j]].logprob, dl[j - 2][ids[j]].logprob if j >= 2 else None] for j in model_pos}
        else:
            pl = pre[k].prompt_logprobs
            r["model"] = {j: [pl[vis_at[j]][ids[j]].logprob] for j in model_pos if vis_at[j] >= 1}
        R["cases"][k] = r
    json.dump(R, open(outp, "w"))
    print("done", kind, len(names), flush=True)
    for c in psutil.Process().children(recursive=True):
        c.kill()
    os._exit(0)


def compare(fa, fb):
    A, B = json.load(open(fa)), json.load(open(fb))
    agg = {}
    kernel_bad = 0
    for k, a in A["cases"].items():
        b = B["cases"][k]
        kernel_bad += any(v for f, v in a.items() if f.endswith(("_wrong", "_low_margin", "_loud")))
        fh = a["first_hidden"]
        for j, (lp, ld) in a["model"].items():
            if j not in b["model"]:
                continue
            part = "after_hidden" if fh is not None and int(j) > fh else "before_hidden"
            for tag, v in (("prefill", lp), ("decode", ld)):
                if v is not None:
                    agg.setdefault(f"{tag}:{part}", []).append(abs(v - b["model"][j][0]))
    out = {"cases": len(A["cases"]), "kernel_cases_not_exact": kernel_bad, "visible_env": A["visible_env"]}
    for key, d in sorted(agg.items()):
        d.sort()
        out[key] = {"positions": len(d), "mean_abs_dlogit": round(sum(d) / len(d), 4),
                    "p99": round(d[int(0.99 * (len(d) - 1))], 4), "max": round(d[-1], 4),
                    "frac_gt_1": round(sum(x > 1 for x in d) / len(d), 5)}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
