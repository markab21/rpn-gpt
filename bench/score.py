"""Exact scoring, as qwenputer's arith_bench.strict_pred / exact_fields (MIT, commit cb1c969): the last \\boxed{} after
the thinking (else the last number), read as an exact rational (thousands commas and spaces removed, decimals and a/b
fractions compared exactly, no tolerance), equal to the gold."""

import re
import sys
from decimal import Decimal, InvalidOperation
from fractions import Fraction

sys.set_int_max_str_digits(0)  # a runaway digit string in an answer must not end the run
NUMR = r"-?\s*\d+(?:\.\d+)?"


def _norm(s):
    s = re.sub(r"\\(?:text|mathrm|textbf|mbox|mathbf)\{([^{}]*)\}", r" \1 ", s)
    s = s.replace("\u2212", "-").replace("\\left", "").replace("\\right", "").replace("{,}", ",")
    for x in ("\\$", "$", "\\,", "\\!", "\\;", "\\ ", "~", "**"):
        s = s.replace(x, " ")
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)  # thousands separators
    s = re.sub(r"(?<=\d)\s+(?=\d{3}(?!\d))", "", s)  # 1 234 567
    s = re.sub(r"\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", s)
    return s.strip()


def _dec(x):
    return Fraction(Decimal(x.replace(" ", "")))


def parse_value(s):
    s = _norm(s)
    if "=" in s:
        s = s.rsplit("=", 1)[1]
    s = s.strip().rstrip(".").strip()
    m = re.fullmatch(rf"(-)?\s*\(?\s*({NUMR})\s*\)?\s*/\s*\(?\s*({NUMR})\s*\)?", s)
    if m:
        b = _dec(m[3])
        return None if b == 0 else (-1 if m[1] else 1) * _dec(m[2]) / b
    m = re.fullmatch(rf"\(?\s*({NUMR})\s*\)?", s)
    if m:
        return _dec(m[1])
    nums = re.findall(NUMR, s)
    return _dec(nums[0]) if len(nums) == 1 else None


def _balanced(t, i):
    depth = 1
    for k in range(i, len(t)):
        depth += {"{": 1, "}": -1}.get(t[k], 0)
        if not depth:
            return t[i:k]
    return t[i:]


def pred(text):
    """-> (value or None, source)."""
    t = text.rsplit("</think>", 1)[1] if "</think>" in text else ""  # thinking is on: no </think>, no answer
    t = re.sub(r"```output\n.*?(```|$)", " ", t, flags=re.S)  # a machine's printout is not the answer
    boxes = [_balanced(t, b.end()) for b in re.finditer(r"\\boxed\s*\{", t)]
    try:
        if boxes:
            return parse_value(boxes[-1]), "boxed"
        nums = re.findall(r"-?\d+(?:\.\d+)?", _norm(t))
        return (_dec(nums[-1]), "last_number") if nums else (None, "none")
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None, "unparsed"


def gold_value(g):
    g = g.replace(",", "").strip()
    return Fraction(g) if "/" in g else Fraction(Decimal(g))


def exact(text, gold):
    v, src = pred(text)
    return {"pred": None if v is None else str(v), "src": src, "exact": v is not None and v == gold_value(gold)}


assert exact("2+2=5</think> so \\boxed{1,234.50}", "1234.5")["exact"]
assert exact("</think>\\boxed{\\frac{-1}{3}}", "-1/3")["exact"] and not exact("</think>\\boxed{0.333}", "-1/3")["exact"]
assert exact("</think>the answer is 42.", "42")["exact"] and not exact("\\boxed{7}", "7")["exact"]
assert not exact("</think>```output\n7\n```\n", "7")["exact"]
