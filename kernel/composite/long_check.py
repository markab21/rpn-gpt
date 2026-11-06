"""Kernel exactness on long streams (far past the ~20k tokens training rows reach): the composite in vLLM (plugin),
prompt_logprobs at every kernel-owned position of the reference stream of the longest L6 (needs-the-tool) programs.
  CUDA_VISIBLE_DEVICES=g PYTHONPATH=/srv/nemo/composite venv-kernel/bin/python kernel/composite/long_check.py \
      [TN_eval.jsonl] [per_family=2]          -> prints one line per stream, writes composite_runs/long_check.json
A stream passes when the composite's top token equals the kernel's at every owned position.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2])]
from forth import sp  # noqa: E402

COMP = "/srv/nemo/forth-lightning"
rows_path = sys.argv[1] if len(sys.argv) > 1 else "data/rendered/sp_kernel/TN_eval.jsonl"
per = int(sys.argv[2]) if len(sys.argv) > 2 else 2
sp.MAX_TOKENS = 10**6
tok = sp.tokenizer()
by = defaultdict(list)
for r in map(json.loads, open(rows_path)):
    p, ids, owned = sp.problems(r["forth"], r["output"], tok)
    assert p is None, (r["family"], p)
    by[r["family"]].append((len(ids), r["family"], ids, owned))
streams = [s for fam in sorted(by) for s in sorted(by[fam], reverse=True)[:per]]  # the longest of each family

from vllm import LLM, SamplingParams  # noqa: E402

llm = LLM(COMP, max_model_len=max(n for n, *_ in streams) + 16, enable_prefix_caching=False,
          gpu_memory_utilization=0.85, max_num_seqs=4, enable_flashinfer_autotune=False)
outs = llm.generate([{"prompt_token_ids": ids} for _, _, ids, _ in streams],
                    SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=0))
R = []
for (n, fam, ids, owned), o in zip(streams, outs):
    top = [None] + [next(t for t, v in d.items() if v.rank == 1) for d in o.prompt_logprobs[1:]]
    bad = [p for p in owned if top[p] != ids[p]]
    R.append({"family": fam, "tokens": n, "kernel_tokens": len(owned), "wrong": len(bad), "first_wrong": bad[:3]})
    print(json.dumps(R[-1]), flush=True)
Path("composite_runs").mkdir(exist_ok=True)
json.dump(R, open("composite_runs/long_check.json", "w"), indent=1)
print("ALL EXACT" if not any(r["wrong"] for r in R) else "NOT EXACT", flush=True)
