"""vLLM exactness of the stored-program kernel: in-process vLLM (TP1, eager or compiled), prompt_logprobs at every
kernel-owned position of the same token streams sp_test.py checks with HF.
  CUDA_VISIBLE_DEVICES=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 python sp_vllm.py CKPT eager|compile WHAT [ARGS...]
WHAT: all | hanoi N.. | blackjack SEED.. | arith N OPS | mulmod N OPS | block N DEPTH
Extra LLM(...) kwargs as JSON in env VLLM_KW (e.g. a compilation_config); TAG env suffixes the result file.
"""

import json
import os
import random
import sys

from vllm import LLM, SamplingParams

import sp_test as T

ckpt, mode, what = sys.argv[1:4]
progs = T.programs(what, sys.argv[4:])
llm = LLM(ckpt, enforce_eager=mode == "eager", enable_prefix_caching=False, max_model_len=32768,
          gpu_memory_utilization=0.5, max_num_seqs=16, enable_flashinfer_autotune=False,
          **json.loads(os.environ.get("VLLM_KW", "{}")))
sp = SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=0)
R = {}
for name, runs in progs.items():
    ids, want = T.flatten(runs)
    o = llm.generate([{"prompt_token_ids": ids}], sp, use_tqdm=False)[0]
    top = [None] + [next(t for t, v in d.items() if v.rank == 1) for d in o.prompt_logprobs[1:]]
    bad = [p for p in sorted(want) if top[p + 1] != want[p]]
    R[name] = {"tokens": len(ids), "kernel_tokens": len(want), "wrong": len(bad),
               "first_wrong (pos, want, got)": [(p, want[p], top[p + 1]) for p in bad[:5]]}
    print(mode, name, R[name], flush=True)
json.dump(R, open(f"{ckpt}/sp_vllm_{what}_{mode}{os.environ.get('TAG', '')}.json", "w"), indent=1)
