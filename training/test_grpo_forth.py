"""CPU check of grpo_forth's turn loop against real loop_sim rollouts: replay each rollout's model text in the pieces a
vLLM stop at the fences would give (plus random mid-text chunk cuts), check every output block and the grade match.
  python training/test_grpo_forth.py SPEC_DIR TOKENIZER ROLLOUTS.json [N]
"""
import json
import random
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from grpo_forth import CLOSE, EOS, advance, load_loop_sim, new_state  # noqa: E402

spec, tokp, runs = sys.argv[1:4]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 40
from transformers import AutoTokenizer  # noqa: E402

L, tok = load_loop_sim(spec), AutoTokenizer.from_pretrained(tokp)
cfg = {"think_budget": 4000, "cap": 8000, "rounds": 12}
R = [r for r in json.load(open(runs)) if r["tokens"] < 7500]  # capped rollouts end on loop_sim's exact budget
rng, bad, outs = random.Random(0), 0, 0
for r in rng.sample(R, min(n, len(R))):
    s, text, k, pieces = new_state(r, cfg), r["text"], 0, []
    for a, b in r["spans"]:  # model text up to each output block (it ends with the fence), then after the last one
        pieces.append(text[k:a])
        k = b + 1  # loop_sim: block + "\n"
    pieces.append(text[k:] + EOS)
    got_obs = []
    for i, pc in enumerate(pieces):
        offs = [a for a, _ in tok(pc, add_special_tokens=False, return_offsets_mapping=True).offset_mapping]
        cut = rng.choice(offs[1:-2]) if len(offs) > 4 and rng.random() < 0.5 else None  # a length stop: a token boundary
        parts = [pc[:cut], pc[cut:]] if cut else [pc]
        for part in parts:
            o, _, done = advance(s, part, tok, L)
            if o and o != CLOSE:
                got_obs.append(o)
    want = [text[a:b] for a, b in r["spans"]]
    same_out = [o.strip("\n").replace(CLOSE.strip("\n"), "").strip("\n") for o in got_obs] == want
    outs += len(want)
    if not (done and same_out and s["ok"] == r["ok"]):
        bad += 1
        print("MISMATCH", r["q"][:60], done, same_out, s.get("ok"), r["ok"], len(got_obs), len(want))
print(f"{n - bad}/{n} rollouts reproduced ({outs} output blocks)")

# think budget: 4100 tokens of thinking in 500-token chunks -> one forced close, no reward until the end of the turn
s, filler = new_state({"q": "q", "a": "7"}, cfg), " seven" * 500
closes = sum(advance(s, filler, tok, L)[0] == CLOSE for _ in range(9))
o, _, done = advance(s, "The answer is \\boxed{7}." + EOS, tok, L)
assert closes == 1 and done and s["ok"] and "</think>" in s["text"], (closes, done, s.get("ok"))
# cap: no end of turn, past 8000 visible tokens -> done, graded on what is there
s = new_state({"q": "q", "a": "7"}, dict(cfg, think_budget=0))
done = [advance(s, filler, tok, L)[2] for _ in range(17)]
assert done.index(True) == 15 and not s["ok"], done
print("budget / cap checks ok")
sys.exit(bool(bad))
