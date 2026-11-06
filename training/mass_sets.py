"""Mass-sampling problems (2026-11-04): ~22k fresh problems over every family we have, a spread of difficulty where the
generator has a knob, disjoint from every existing problem set. FIRST a frozen test set is carved out (300 machine
families + 300 DM families, balanced round-robin over families; never trained on, never used for selection), the rest
is the sampling pool (stage A x4, stage B x16 on the informative part, GRPO prompt selection, rows).
  kind: hard     L6 full size (curriculum.L6), DMbig full size (demo_gen.dmbig)
        small    L6 at L6_SMALL sizes (5-40 iterations)
        variant  L6_EASY (recurrence, euclid_trace), DMbig easy=True (roots, is_prime)
        easy     DeepMind train-difficulty modules (training/dm_gen.py train --seed 1104, 26 families)
        easy_big DeepMind extrapolation modules (dm_gen.py eval --seed 1104, 15 families; conversion -> conversion_big)
Every problem with a program is run on the reference machine (text_vm via loop_sim.kernel_output); a program that errors
(outside the machine's limits) is redrawn. Answers for programs come from the machine (Python must agree where it has one).
  LOOP_TOKENIZER=<Lightning tokenizer> python training/mass_sets.py
  (reads data/rendered/sp_kernel/DMmass_{train,eval}.jsonl; writes training/data/frozen_test.json, training/data/mass_pool.json)"""
import collections, glob, json, os, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training")]
from forth.curriculum import L6, L6_SMALL, L6_EASY  # noqa: E402
from demo_gen import dmbig  # noqa: E402
import loop_sim  # noqa: E402

N_L6, N_BIG, N_DM, N_DMBIG_EXTRA, N_TEST = 600, 600, 360, 260, 300  # per family; N_TEST per half
L, SK = ROOT / "data/loop", ROOT / "data/rendered/sp_kernel"
held = set()  # every problem set we have: data/loop (and ablation/), every rendered eval, DM_train
for f in glob.glob(str(L / "**/*.json"), recursive=True):
    d = json.load(open(f))
    if isinstance(d, list) and d and isinstance(d[0], dict) and "q" in d[0] and Path(f).name not in ("frozen_test.json", "mass_pool.json"):
        held |= {p["q"] for p in d}
for f in [f for f in glob.glob(str(SK / "*_eval.jsonl")) if "DMmass" not in f] + [str(SK / "DM_train.jsonl")]:  # (not our inputs)
    held |= {json.loads(x)["prompt"] for x in open(f)}
print("held-out / used problems:", len(held))


def root_prog(n, k):  # (= stage2a_sets.root_prog) floor root by binary search, then the nearer of r, r+1
    pw = "dup *" if k == 2 else "dup dup * *"
    return (f"variable n variable lo variable hi\n: pw ( m -- m^k ) {pw} ;\n: root ( -- r ) 0 lo ! {10**6 if k == 2 else 10**5} hi ! "
            "begin lo @ hi @ < while lo @ hi @ + 1+ 2/ dup pw n @ > if 1- hi ! else lo ! then repeat "
            "lo @ 1+ pw n @ - n @ lo @ pw - < if lo @ 1+ else lo @ then ;\n"
            f"{n} n ! root .")


def prime_prog(n, word):  # (= stage2a_sets.prime_prog)
    return (": prime ( n -- f ) dup 2 < if drop 0 exit then 2 begin 2dup dup * >= while 2dup mod 0= if 2drop 0 exit "
            f"then 1+ repeat 2drop -1 ;\n{n} prime {'0= ' if word == 'composite' else ''}.")


from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(os.environ.get("LOOP_TOKENIZER", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"))
seen, out, redrawn = set(held), [], collections.Counter()


def add(p, kind):
    """-> True when p is new and (if it has a program) the machine runs it without error to the expected answer."""
    if p["q"] in seen:
        return False
    if p.get("forth"):
        o = " ".join(loop_sim.kernel_output([p["forth"]], tok).split())
        want = {"-1": "True", "0": "False"}.get(o, o) if p["family"] == "DMbig.is_prime" else o
        if o.startswith("error") or not o or (p["a"] is not None and want != p["a"]):
            redrawn[p["family"]] += 1
            return False
        p["a"] = p["a"] or o
    seen.add(p["q"])
    out.append({"q": p["q"], "a": str(p["a"]), "family": p["family"], "kind": kind})
    return True


R = random.Random("mass-2026-11-04")
for fam in sorted(L6):  # L6: equal parts per tier (full, small, and easy where L6_EASY has the family)
    tiers = [("hard", L6[fam][0]), ("small", L6_SMALL[fam])] + ([("variant", L6_EASY[fam])] if fam in L6_EASY else [])
    for kind, args in tiers:
        got = 0
        while got < N_L6 // len(tiers):
            a = args(R)
            got += add({"q": R.choice(L6[fam][2]).format(*a), "a": None, "family": "L6." + fam, "forth": L6[fam][1](*a)}, kind)
BIG4, ODD2 = ("DMbig.mul", "DMbig.div_remainder", "DMbig.gcd", "DMbig.lcm"), ("DMbig.nearest_integer_root", "DMbig.is_prime")
for easy, quota in ((False, {**dict.fromkeys(BIG4, N_BIG), **dict.fromkeys(ODD2, N_BIG // 2)}),  # roots / is_prime:
                    (True, dict.fromkeys(ODD2, N_BIG // 2))):                                   # half full, half easy
    got = collections.Counter()
    while any(got[f] < n for f, n in quota.items()):
        for p in dmbig(R, 50, easy=easy):
            if got[p["family"]] >= quota.get(p["family"], 0):
                continue
            if easy and p["family"] == "DMbig.nearest_integer_root":  # the easy roots fit root_prog's search range
                p = p | {"forth": root_prog(int(p["q"].split()[-5]), 2 if "square" in p["q"] else 3)}
            elif easy:
                p = p | {"forth": prime_prog(int(p["q"].split()[1]), p["q"].split()[3])}
            got[p["family"]] += add(p, "variant" if easy else "hard")
for f, kind, n in (("DMmass_train.jsonl", "easy", N_DM), ("DMmass_eval.jsonl", "easy_big", N_DMBIG_EXTRA)):
    by = collections.defaultdict(list)
    for x in map(json.loads, open(SK / f)):
        fam = x["family"] + ("_big" if kind == "easy_big" and x["family"] == "measurement.conversion" else "")
        by[fam].append({"q": x["prompt"], "a": x["answer"], "family": fam})
    for fam, ps in sorted(by.items()):
        R.shuffle(ps)
        got = 0
        for p in ps:
            if got == n:
                break
            got += add(p, kind)
        assert got == n, (fam, got)

R = random.Random("mass-frozen-test-2026-11-04")
test = set()
for half in (("hard", "small", "variant"), ("easy", "easy_big")):  # round-robin over families: balanced
    by = collections.defaultdict(list)
    for p in out:
        if p["kind"] in half:
            by[p["family"]].append(p)
    for ps in by.values():
        R.shuffle(ps)
    fams, k = sorted(by), 0
    while sum(p["kind"] in half for p in out if p["q"] in test) < N_TEST:
        test.add(by[fams[k % len(fams)]][k // len(fams)]["q"])
        k += 1
frozen = [p for p in out if p["q"] in test]
pool = [p for p in out if p["q"] not in test]
qs = [p["q"] for p in out]
assert len(set(qs)) == len(qs) and not held & set(qs) and not {p["q"] for p in frozen} & {p["q"] for p in pool}
assert len(frozen) == 2 * N_TEST
json.dump(frozen, open(L / "frozen_test.json", "w"), indent=1)
json.dump(pool, open(L / "mass_pool.json", "w"), indent=1)
print("redrawn (machine error or disagreement):", dict(redrawn))
print("frozen", len(frozen), sorted(collections.Counter((p["family"], p["kind"]) for p in frozen).items()))
print("pool", len(pool), sorted(collections.Counter((p["family"], p["kind"]) for p in pool).items()))
