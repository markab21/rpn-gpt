"""Stage-1 (context distillation) problem sets, disjoint from every eval set:
  hard: fresh L6 problems (forth/curriculum.L6 generators, new seed, answers from the reference machine) + fresh
        scaled-up DM problems (training/demo_gen.dmbig, new seed) -> rolled out WITH the card (rubric_v23)
  easy: DM_train problems of the families Lightning solves ~100% with no machine (ablation 2026-11-04, >= 85%)
        -> rolled out with NO machine (training/data/cards/nomachine.txt)
  python training/stage1_sets.py   (writes training/data/stage1_hard.json, training/data/stage1_easy.json)"""
import collections, json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training")]
from forth.curriculum import L6  # noqa: E402
from forth.ref import Forth  # noqa: E402
from demo_gen import dmbig  # noqa: E402

N_HARD, N_EASY = 18, 13  # per family
EASY = ["algebra.linear_1d", "algebra.linear_2d", "algebra.sequence_next_term", "arithmetic.add_or_sub",
        "arithmetic.add_sub_multiple", "arithmetic.div", "arithmetic.mixed", "arithmetic.mul",
        "arithmetic.mul_div_multiple", "arithmetic.nearest_integer_root", "comparison.closest", "comparison.kth_biggest",
        "comparison.sort", "measurement.time", "numbers.div_remainder", "numbers.gcd", "numbers.lcm",
        "numbers.place_value", "numbers.round_number", "polynomials.evaluate"]
L = ROOT / "data/loop"
held = {p["q"] for f in ("demo_set", "demo_pool", "demo10", "dm_frontier", "dm_frontier_small")
        for p in json.load(open(L / f"{f}.json"))}
held |= {json.loads(x)["prompt"] for x in open(ROOT / "data/rendered/sp_kernel/TN_eval.jsonl")}

R, hard = random.Random("stage1-hard-2026-11-04"), []
for fam, (args, prog, qs) in sorted(L6.items()):
    got = 0
    while got < N_HARD:
        a = args(R)
        q = R.choice(qs).format(*a)
        out = Forth().run(prog(*a)).strip()
        if q in held or not out or any(p["q"] == q for p in hard):
            continue
        hard.append({"q": q, "a": out, "family": "L6." + fam})
        got += 1
hard += [{k: p[k] for k in ("q", "a", "family")} for p in dmbig(R, N_HARD) if p["q"] not in held]

R = random.Random("stage1-easy-2026-11-04")
dm = collections.defaultdict(list)
for x in map(json.loads, open(ROOT / "data/rendered/sp_kernel/DM_train.jsonl")):
    if x["prompt"] not in held:
        dm[x["family"]].append({"q": x["prompt"], "a": str(x["answer"]), "family": x["family"]})
easy = [p for f in EASY for p in R.sample(dm[f], N_EASY)]

assert not held & {p["q"] for p in hard + easy} and len({p["q"] for p in hard}) == len(hard)
json.dump(hard, open(L / "stage1_hard.json", "w"), indent=1)
json.dump(easy, open(L / "stage1_easy.json", "w"), indent=1)
print("hard", len(hard), collections.Counter(p["family"] for p in hard))
print("easy", len(easy), collections.Counter(p["family"] for p in easy))
