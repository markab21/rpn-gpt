"""Dev set for round decisions (r3 vs r1 and later rounds): N problems from mass_pool.json, stratified like frozen_test
(the same (family, kind) cells, N / 600 of each cell's count), disjoint from grpo_pool, dev_codefull and every eval /
held-out set in data/loop. Training rows exclude it (training/r3_rows.py HELD); frozen_test stays the once-per-milestone
number.  python training/dev_round.py [N]   -> training/data/dev_round.json"""
import collections, json, random, sys
from pathlib import Path

L = Path(__file__).resolve().parents[1] / "data/loop"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
used = {p["q"] for f in ["frozen_test", "grpo_pool", "dev_codefull", "demo_set", "demo_pool", "demo10", "demo10_divrem",
                         "dm_frontier", "dm_frontier_small", "math500", "s2a_eval10x2"] for p in json.load(open(L / f"{f}.json"))}
frozen = collections.Counter((p["family"], p["kind"]) for p in json.load(open(L / "frozen_test.json")))
by = collections.defaultdict(list)
for p in json.load(open(L / "mass_pool.json")):
    if p["q"] not in used:
        by[(p["family"], p["kind"])].append(p)
R, out = random.Random("dev_round-2026-11-05"), []
for c in sorted(frozen):
    out += R.sample(by[c], min(len(by[c]), round(frozen[c] * N / sum(frozen.values()))))
assert len({p["q"] for p in out}) == len(out) and not {p["q"] for p in out} & used
json.dump(out, open(L / "dev_round.json", "w"), indent=1)
print(len(out), "problems;", collections.Counter(p["kind"] for p in out))
