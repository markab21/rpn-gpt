"""Round-4 rollout problems (s2a_r3 at 12k): the weakest cells of s2a_r3's buckets. Fresh problems (new seed, the
generators of training/mass_sets.py, every answer run on the reference machine) where mass_pool has run out, mass_pool
problems elsewhere; disjoint from every data/loop set except mass_pool (frozen_test, dev_round, dev_codefull, grpo_pool,
r3_weak12k, the eval sets) and from the r3 training rows (ROWS: rows_v8 train / val jsonl).
  LOOP_TOKENIZER=<Lightning tokenizer> python training/r4_sets.py ROWS_TRAIN ROWS_VAL   -> training/data/r4_weak12k.json"""
import collections, glob, json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training")]
from forth.curriculum import L6  # noqa: E402
from demo_gen import dmbig  # noqa: E402
import loop_sim  # noqa: E402

L = ROOT / "data/loop"
FRESH = {("L6.recurrence", "hard"): 200, ("DMbig.nearest_integer_root", "hard"): 200, ("DMbig.is_prime", "hard"): 200}
POOL = {("DMbig.is_prime", "variant"): 84, ("numbers.is_prime", "easy"): 132, ("algebra.polynomial_roots", "easy"): 97,
        ("algebra.polynomial_roots_big", "easy_big"): 102, ("L6.lcg_trace", "hard"): 110, ("L6.lcg_trace", "small"): 140,
        ("L6.euclid_trace", "hard"): 100, ("L6.euclid_trace", "small"): 100, ("L6.euclid_trace", "variant"): 100}

used = {json.loads(x)["q"] for f in sys.argv[1:3] for x in open(f)}
for f in glob.glob(str(L / "**/*.json"), recursive=True):
    d = json.load(open(f))
    if isinstance(d, list) and d and isinstance(d[0], dict) and "q" in d[0] and Path(f).name != "mass_pool.json":
        used |= {p["q"] for p in d}
pool = json.load(open(L / "mass_pool.json"))
R, out = random.Random("r4-2026-11-05"), []
by = collections.defaultdict(list)
for p in pool:
    if p["q"] not in used:
        by[(p["family"], p["kind"])].append(p)
for c, n in sorted(POOL.items()):
    out += R.sample(sorted(by[c], key=lambda p: p["q"]), min(n, len(by[c])))
# the pool's leftovers of the FRESH cells first, then new draws
for c in FRESH:
    take = R.sample(sorted(by[c], key=lambda p: p["q"]), min(FRESH[c], len(by[c])))
    out += take
    FRESH[c] -= len(take)
seen = used | {p["q"] for p in pool} | {p["q"] for p in out}
from transformers import AutoTokenizer  # noqa: E402
import os  # noqa: E402
tok = AutoTokenizer.from_pretrained(os.environ.get("LOOP_TOKENIZER", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"))


def add(p, kind):  # (= mass_sets.add) new, and the machine runs its program without error to the expected answer
    if p["q"] in seen:
        return False
    o = None
    if p.get("forth"):  # (DMbig roots / is_prime at full size: Python's answer, no program)
        o = " ".join(loop_sim.kernel_output([p["forth"]], tok).split())
        if o.startswith("error") or not o or (p["a"] is not None and o != p["a"]):
            return False
    seen.add(p["q"])
    out.append({"q": p["q"], "a": str(p["a"] or o), "family": p["family"], "kind": kind})
    return True


fam = "recurrence"
while FRESH[("L6.recurrence", "hard")] > 0:
    a = L6[fam][0](R)
    FRESH[("L6.recurrence", "hard")] -= add({"q": R.choice(L6[fam][2]).format(*a), "a": None, "family": "L6." + fam,
                                             "forth": L6[fam][1](*a)}, "hard")
while any(FRESH[(f, "hard")] > 0 for f in ("DMbig.nearest_integer_root", "DMbig.is_prime")):
    for p in dmbig(R, 50):
        if FRESH.get((p["family"], "hard"), 0) > 0:
            FRESH[(p["family"], "hard")] -= add(p, "hard")
qs = [p["q"] for p in out]
assert len(set(qs)) == len(qs) and not set(qs) & used
json.dump(out, open(L / "r4_weak12k.json", "w"), indent=1)
print(len(out), sorted(collections.Counter((p["family"], p["kind"]) for p in out).items()))
