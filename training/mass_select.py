"""Selections from the mass-sampling map (training/mass_report.py OUT.problems.jsonl):
  python training/mass_select.py stageb POOL.json REPORT.problems.jsonl OUT.json [N_PER_CELL]
      stage B (more tries): problems solved sometimes (0 < pass < 1), plus every problem of a (family, kind) cell whose
      pass rate is < 25% (rare successes need more tries to be seen)
  python training/mass_select.py grpo POOL.json REPORT.problems.jsonl OUT.json [N]
      GRPO prompts: pass rate in [0.1, 0.9] (on all samples so far), at most N (default 1000), round-robin over
      (family, kind) cells so no family dominates; each prompt keeps its pass rate and sample count"""
import collections, json, random, sys

mode, pool, rep, out = sys.argv[1:5]
P = {p["q"]: p for p in json.load(open(pool))}
rows = [json.loads(x) for x in open(rep)]
cell = collections.defaultdict(lambda: [0, 0])
for x in rows:
    cell[(x["family"], x["kind"])][0] += x["ok"]
    cell[(x["family"], x["kind"])][1] += x["n"]
if mode == "stageb":
    sel = [x for x in rows if 0 < x["ok"] < x["n"] or cell[(x["family"], x["kind"])][0] < 0.25 * cell[(x["family"], x["kind"])][1]]
    if len(sys.argv) > 5:  # at most N per (family, kind) cell, random (stage B budget)
        R, by = random.Random("stageb-2026-11-04"), collections.defaultdict(list)
        for x in sel:
            by[(x["family"], x["kind"])].append(x)
        sel = [x for c in sorted(by) for x in R.sample(by[c], min(len(by[c]), int(sys.argv[5])))]
else:
    by = collections.defaultdict(list)
    for x in rows:
        if 0.1 <= x["ok"] / x["n"] <= 0.9:
            by[(x["family"], x["kind"])].append(x)
    R, sel, N = random.Random("grpo-2026-11-04"), [], int(sys.argv[5]) if len(sys.argv) > 5 else 1000
    for xs in by.values():
        R.shuffle(xs)
    k = 0
    while len(sel) < N and any(by.values()):
        c = sorted(by)[k % len(by)]
        if by[c]:
            sel.append(by[c].pop())
        k += 1
json.dump([P[x["q"]] | {"pass": round(x["ok"] / x["n"], 3), "samples": x["n"]} for x in sel], open(out, "w"), indent=1)
print(len(sel), "selected;", sorted(collections.Counter(x["family"] + "/" + x["kind"] for x in sel).items()))
