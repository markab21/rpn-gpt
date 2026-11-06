"""Regrade training/loop_sim.py results on MATH-style answers (training/data/math500.json) with math_verify: the last \\boxed{}
after </think> vs the gold answer, equivalence by math_verify (exact, symbolic; no judge). Writes a copy with "ok"
replaced (the loop_sim grade kept as "ok_exact"), for training/paired_delta.py.
  python training/math_grade.py IN.json OUT.json
Also equal (canon): a one-column pmatrix / bmatrix and the same tuple; an inequality chain and its interval.
"""
import json
import re
import sys
from pathlib import Path

from math_verify import parse, verify

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loop_sim import last_boxed  # noqa: E402

LE = r"(<=|<|\\leq?|\\lt|≤)"
INEQ = re.compile(r"^\s*([^<≤\\]+?)\s*" + LE + r"\s*[A-Za-z\\]+\s*" + LE + r"\s*(.+?)\s*$")


def canon(s):
    """forms math_verify does not equate: a column vector (pmatrix / bmatrix of one column) -> a tuple '(a, b, c)';
    an inequality chain 'a < x <= b' -> the interval '(a, b]'"""
    m = re.fullmatch(r"\s*\\begin\{[pb]matrix\}(.*)\\end\{[pb]matrix\}\s*", s, re.S)
    if m and "&" not in m.group(1):
        return "(" + ", ".join(x.strip() for x in m.group(1).split("\\\\")) + ")"
    m = INEQ.match(s)
    if m:
        return ("(" if m.group(2) in ("<", "\\lt") else "[") + m.group(1) + ", " + m.group(4) + (")" if m.group(3) in ("<", "\\lt") else "]")
    return s


def same(a, got):
    return bool(got) and (verify(parse(f"${a}$"), parse(f"${got}$")) or verify(parse(f"${canon(a)}$"), parse(f"${canon(got)}$")))


rows = json.load(open(sys.argv[1])) if __name__ == "__main__" else []
for r in rows:
    got = last_boxed(r["text"].rsplit("</think>", 1)[-1]) if "</think>" in r["text"] else None
    got = got and re.sub(r"\\\\\[[^\]]*\]", r"\\\\", got)  # matrix row spacing '\\[2pt]' -> '\\' (math_verify misparses it)
    r["ok_exact"], r["ok"] = r["ok"], same(r["a"], got)
if __name__ == "__main__":
    json.dump(rows, open(sys.argv[2], "w"), indent=1)
    print(f"{sys.argv[1]}: {sum(r['ok'] for r in rows)} / {len(rows)} (loop_sim exact {sum(r['ok_exact'] for r in rows)})")
assert verify(parse("$\\frac{1}{2}$"), parse("$0.5$")) and not verify(parse("$3$"), parse("$4$"))
_v = "\\begin{pmatrix} -18 \\\\ -49 \\\\ 96 \\end{pmatrix}"
assert same(_v, "(-18, -49, 96)") and not same(_v, "(-18, -49, 95)") and same(_v, _v)
assert same("(3,4]", "3 < \\lambda \\le 4") and not same("(3,4]", "3 \\le \\lambda \\le 4") and same("[1, 2)", "1 \\leq x < 2")
