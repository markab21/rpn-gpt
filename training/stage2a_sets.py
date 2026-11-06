"""Stage-2a (expert iteration round 1) problems, fresh seeds, disjoint from every eval set AND the stage-1 sets:
  hard:    L6 + DMbig families where the policy (s1_cold) or the card shows some success, full size
  variant: easier variants of the families nobody solves (curriculum.L6_EASY, demo_gen.dmbig(easy=True))
  easy:    DM_train problems of the 20 stage-1 easy families + comparison.pair (s1_cold lost it: 10 -> 6)
Every problem with a program (all hard + variant ones) is checked on the reference machine (kernel/text_vm.py via
loop_sim.kernel_output): the machine's output must be the answer, with no error (so it is inside the machine's limits).
  python training/stage2a_sets.py [2]   (writes training/data/stage2a_problems.json, or stage2a_r2_problems.json for round 2:
  new seeds, disjoint from round 1 too, weighted toward the machine families with low round-1 acceptance; needs the
  Lightning tokenizer: LOOP_TOKENIZER)"""
import collections, json, os, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training")]
from forth.curriculum import L6, L6_EASY  # noqa: E402
from demo_gen import dmbig  # noqa: E402
import loop_sim  # noqa: E402

ROUND = int(sys.argv[1]) if len(sys.argv) > 1 else 1
N_L6 = {"compound": 30, "lcg": 30, "lcg_trace": 30, "crt_search": 25, "count_quadres": 25, "count_digitsum": 20}
N_BIG = {"DMbig.div_remainder": 30, "DMbig.lcm": 30, "DMbig.gcd": 20, "DMbig.mul": 20}
N_VAR, N_EASY, N_PAIR = 25, 8, 20
if ROUND == 2:  # round-1 acceptance: compound / lcg / DMbig 24-34%, lcg_trace 6%, variants 4-45%; crt / quadres ~95% direct
    N_L6 = {"compound": 50, "lcg": 50, "lcg_trace": 50, "crt_search": 10, "count_quadres": 10, "count_digitsum": 15}
    N_BIG = {"DMbig.div_remainder": 50, "DMbig.lcm": 50, "DMbig.gcd": 40, "DMbig.mul": 40}
    N_VAR, N_EASY, N_PAIR = 40, 4, 10
TAG = "stage2a-r1" if ROUND == 1 else f"stage2a-r{ROUND}"
L = ROOT / "data/loop"
held = {p["q"] for f in ("demo_set", "demo_pool", "demo10", "dm_frontier", "dm_frontier_small", "stage1_hard", "stage1_easy")
        for p in json.load(open(L / f"{f}.json"))}
held |= {json.loads(x)["prompt"] for x in open(ROOT / "data/rendered/sp_kernel/TN_eval.jsonl")}
EASY = sorted({p["family"] for p in json.load(open(L / "stage1_easy.json"))})  # the 20 stage-1 easy families
if ROUND > 1:
    held |= {p["q"] for p in json.load(open(L / "stage2a_problems.json"))}


def root_prog(n, k):  # floor root by binary search, then the nearer of r, r+1 (no ties for k = 2, 3)
    pw = "dup *" if k == 2 else "dup dup * *"
    return (f"variable n variable lo variable hi\n: pw ( m -- m^k ) {pw} ;\n: root ( -- r ) 0 lo ! {10**6 if k == 2 else 10**5} hi ! "
            "begin lo @ hi @ < while lo @ hi @ + 1+ 2/ dup pw n @ > if 1- hi ! else lo ! then repeat "
            "lo @ 1+ pw n @ - n @ lo @ pw - < if lo @ 1+ else lo @ then ;\n"
            f"{n} n ! root .")


def prime_prog(n, word):
    return (": prime ( n -- f ) dup 2 < if drop 0 exit then 2 begin 2dup dup * >= while 2dup mod 0= if 2drop 0 exit "
            f"then 1+ repeat 2drop -1 ;\n{n} prime {'0= ' if word == 'composite' else ''}.")


R, hard, var = random.Random(f"{TAG}-2026-11-04"), [], []
for fams, dst in ((N_L6, hard), ({f: N_VAR for f in L6_EASY}, var)):
    for fam, n in sorted(fams.items()):
        got = 0
        while got < n:
            a = L6[fam][0](R) if dst is hard else L6_EASY[fam](R)
            q = R.choice(L6[fam][2]).format(*a)
            if q in held or any(p["q"] == q for p in hard + var):
                continue
            dst.append({"q": q, "a": None, "family": "L6." + fam, "forth": L6[fam][1](*a)})
            got += 1
big = collections.defaultdict(list)
for p in dmbig(R, max(N_BIG.values())):
    big[p["family"]].append(p)
hard += [p for f, n in sorted(N_BIG.items()) for p in big[f][:n]]
for p in dmbig(R, N_VAR + 1, easy=True):
    if p["family"] == "DMbig.nearest_integer_root":
        n, k = int(p["q"].split()[-5]), 2 if "square" in p["q"] else 3
        var.append(p | {"forth": root_prog(n, k)})
    elif p["family"] == "DMbig.is_prime":
        var.append(p | {"forth": prime_prog(int(p["q"].split()[1]), p["q"].split()[3])})

from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(os.environ.get("LOOP_TOKENIZER", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"))
for p in hard + var:  # the reference machine's answer; for Python-answered problems it must agree
    out = " ".join(loop_sim.kernel_output([p["forth"]], tok).split())
    want = {"-1": "True", "0": "False"}.get(out, out) if p["family"] == "DMbig.is_prime" else out
    assert not out.startswith("error") and out and (p["a"] is None or want == p["a"]), (p, out)
    p["a"] = p["a"] or out

R = random.Random(f"{TAG}-easy-2026-11-04")
dm = collections.defaultdict(list)
for x in map(json.loads, open(ROOT / "data/rendered/sp_kernel/DM_train.jsonl")):
    if x["prompt"] not in held:
        dm[x["family"]].append({"q": x["prompt"], "a": str(x["answer"]), "family": x["family"]})
easy = [p for f in EASY for p in R.sample(dm[f], N_EASY)] + R.sample(dm["comparison.pair"], N_PAIR)

out = [{k: p[k] for k in ("q", "a", "family")} | {"kind": kind} for kind, ps in (("hard", hard), ("variant", var),
                                                                                 ("easy", easy)) for p in ps]
assert not held & {p["q"] for p in out} and len({p["q"] for p in out}) == len(out)
json.dump(out, open(L / ("stage2a_problems.json" if ROUND == 1 else f"stage2a_r{ROUND}_problems.json"), "w"), indent=1)
print(len(out), collections.Counter((p["kind"], p["family"]) for p in out))
