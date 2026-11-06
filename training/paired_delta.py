"""Paired A/B of two training/loop_sim.py result files on the same problems (e.g. frozen_test x2, base vs adapter on one
engine): per problem the mean correct of each arm, the delta B - A, a 95% bootstrap CI over problems, overall and per
problem "kind" (and family with --family).
  python training/paired_delta.py A.json B.json [--family]
"""
import collections
import json
import random
import sys


def load(path):
    by = collections.defaultdict(list)
    for r in json.load(open(path)):
        by[r["q"]].append((r["ok"], r))
    return by


def ci(d, n=10000, seed=0):
    rng = random.Random(seed)
    m = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(n))
    return m[int(0.025 * n)], m[int(0.975 * n)]


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    probs = json.load(open(sys.argv[3])) if len(sys.argv) > 3 and sys.argv[3].endswith(".json") else None
    kind = {p["q"]: p.get("kind", "?") for p in probs} if probs else {}
    qs = sorted(set(a) & set(b))
    print(f"{len(qs)} paired problems ({len(set(a) ^ set(b))} in one arm only)")
    groups = collections.defaultdict(list)
    for q in qs:
        fam = a[q][0][1].get("family") or "?"
        key = fam if "--family" in sys.argv else kind.get(q, fam.split(".")[0])
        pa, pb = sum(o for o, _ in a[q]) / len(a[q]), sum(o for o, _ in b[q]) / len(b[q])
        groups["ALL"].append((pa, pb))
        groups[key].append((pa, pb))
    for k in ["ALL"] + sorted(set(groups) - {"ALL"}):
        g = groups[k]
        d = [pb - pa for pa, pb in g]
        lo, hi = ci(d)
        print(f"{k:28s} n={len(g):4d}  A {100 * sum(p for p, _ in g) / len(g):5.1f}%  B {100 * sum(p for _, p in g) / len(g):5.1f}%"
              f"  delta {100 * sum(d) / len(d):+5.1f}  [{100 * lo:+.1f}, {100 * hi:+.1f}]  B better on {sum(x > 0 for x in d)}, worse on {sum(x < 0 for x in d)}")


if __name__ == "__main__":
    main()
