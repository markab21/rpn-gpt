"""Decode-path gap between two served model names on ONE vLLM engine (e.g. the base name vs an all-zero LoRA): from the
same prefixes (each saved rollout cut at several points inside the model's own text), greedy continuations of K tokens
with sampled-token logprobs; reports mean |dlogp| over the shared prefix of the two continuations, how often and how
early they diverge. Mode 'serial' (one request at a time) and 'batch' (all requests in flight together).
  python training/decode_gap.py ROLLOUTS.json PROMPT.txt URL NAME_A NAME_B [K] [CUTS]"""
import concurrent.futures as cf, json, os, sys, urllib.request
from transformers import AutoTokenizer

rollouts, prompt_f, url, na, nb = sys.argv[1:6]
K, CUTS = int(sys.argv[6]) if len(sys.argv) > 6 else 200, int(sys.argv[7]) if len(sys.argv) > 7 else 4
tok = AutoTokenizer.from_pretrained("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16")
system = open(prompt_f).read()
pre = []
for r in json.load(open(rollouts)):
    p = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": r["q"]}],
                                tokenize=False, add_generation_prompt=True, enable_thinking=True)
    ids = tok(p + r["text"], add_special_tokens=False).input_ids
    n0 = len(tok(p, add_special_tokens=False).input_ids)
    for c in range(1, CUTS + 1):
        pre.append(ids[:n0 + (len(ids) - n0) * c // (CUTS + 1)])


def gen(name, ids):
    body = {"model": name, "prompt": ids, "max_tokens": K, "temperature": 0.0, "logprobs": 1}
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json",
                                 "Authorization": "Bearer " + os.environ["VLLM_API_KEY"]})
    lp = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["logprobs"]
    return lp["tokens"], lp["token_logprobs"]


for mode in ("serial", "batch"):
    if mode == "serial":
        A = [gen(na, p) for p in pre]
        B = [gen(nb, p) for p in pre]
    else:
        with cf.ThreadPoolExecutor(2 * len(pre)) as ex:
            fa = [ex.submit(gen, na, p) for p in pre]
            fb = [ex.submit(gen, nb, p) for p in pre]
            A, B = [f.result() for f in fa], [f.result() for f in fb]
    d, div, first = [], 0, []
    for (ta, la), (tb, lb) in zip(A, B):
        k = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
        m = len(ta) if k is None else k
        d += [abs(la[i] - lb[i]) for i in range(m)]
        if k is not None:
            div += 1
            first.append(k)
    print(f"{mode}: {len(pre)} prefixes, {len(d)} shared tokens, mean |dlogp| {sum(d) / max(len(d), 1):.4f}, "
          f">1 {100 * sum(x > 1 for x in d) / max(len(d), 1):.2f}%, diverged {div}/{len(pre)} "
          f"(median first divergence at token {sorted(first)[len(first) // 2] if first else '-'} of {K})", flush=True)
