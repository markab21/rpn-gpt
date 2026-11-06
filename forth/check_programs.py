"""Validate preferred-example programs: only supported words, and `output` reproduces exactly in ref.py.

Usage: python3 -m forth.check_programs data/programs/*.jsonl   (prints per-file pass counts; exits 1 on any failure)
Record: {"family": str, "prompt": str, "forth": str, "output": str}
"""

import json
import re
import sys

from .ref import Forth

SUPPORTED = set("+ - * / MOD NEGATE 1+ 1- 0= < > DUP DROP SWAP OVER PICK @ ! . : ; IF ELSE THEN EXIT BEGIN WHILE "
                "REPEAT VARIABLE".split())
BANNED = re.compile(r"HANOI|BLACKJACK|\bCARD|\bDEAL|\b21\b", re.I)  # the prize programs must not be in training data
# Kernel capacities (user: no artificial limits; row-addressed storage). Overflow is a defined kernel error.
MAX_DATA, MAX_RETURN, MAX_DIGITS = 1024, 1024, 31


def problems(r):
    src = re.sub(r"\( [^)]*\)", " ", r["forth"])
    names = set(re.findall(r"(?::|VARIABLE)\s+(\S+)", src, re.I))
    unknown = {w for w in src.upper().split() if not re.fullmatch(r"-?\d+", w)} - SUPPORTED - {n.upper() for n in names}
    if unknown:
        return f"unsupported words {sorted(unknown)}"
    if BANNED.search(r["prompt"] + " " + r["forth"]):
        return "mentions a prize program"
    vm = Forth()
    try:
        got = vm.run(r["forth"])
    except Exception as e:  # noqa: BLE001 - any interpreter failure rejects the record
        return f"{type(e).__name__}: {e}"
    if got != r["output"]:
        return f"output {got!r} != {r['output']!r}"
    if vm.peak["data"] > MAX_DATA or vm.peak["return"] > MAX_RETURN:
        return f"exceeds kernel stacks: data {vm.peak['data']}/{MAX_DATA}, return {vm.peak['return']}/{MAX_RETURN}"
    if max((len(w.lstrip("-")) for w in got.split()), default=0) > MAX_DIGITS:
        return f"a printed number exceeds {MAX_DIGITS} digits"
    return None


def main():
    bad = 0
    for path in sys.argv[1:]:
        rows = [json.loads(line) for line in open(path) if line.strip()]
        fails = [(i, p) for i, r in enumerate(rows) if (p := problems(r))]
        bad += len(fails)
        print(f"{path}: {len(rows) - len(fails)}/{len(rows)} pass")
        for i, p in fails[:10]:
            print(f"  row {i}: {p}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
