"""Per-family pass rates of training/loop_sim.py result files side by side (e.g. base vs. distilled student on the
held-out pool), plus: judged correct (Mammoth, on .rubric.json files), the machine printed exactly the answer, the
answer came from a machine output, a ```forth block was written, and the mean rubric quality.
  python training/loop_report.py base=lora_runs/r1_student_eval.json r1=lora_runs/r1_lora_eval.json [--json OUT]
"""

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distill_rows import accepted  # noqa: E402
from loop_sim import norm  # noqa: E402
from rubric import quality  # noqa: E402


def stats(rows):
    fam = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        f = fam[r.get("family") or "?"]
        f[0] += r["ok"]
        f[1] += 1
    return {"pass": sum(r["ok"] for r in rows), "n": len(rows), "families": {k: v for k, v in sorted(fam.items())},
            "wrote_block": sum(bool(r["blocks"]) for r in rows), "machine_backed": sum(accepted(r) for r in rows),
            "machine_printed": sum(norm(r["a"]) in [" ".join(b["output"].split()) for b in r["blocks"]] for r in rows),
            "judged": sum(bool(r.get("judge_ok")) for r in rows),
            "quality%": round(100 * sum(map(quality, rows)) / max(len(rows), 1)),
            "tokens": round(sum(r.get("tokens", 0) for r in rows) / max(len(rows), 1))}


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("runs", nargs="+", help="label=results.json")
    a.add_argument("--json")
    args = a.parse_args()
    S = {lab: stats(json.load(open(path))) for lab, path in (x.split("=", 1) for x in args.runs)}
    labs = list(S)
    fams = sorted({f for s in S.values() for f in s["families"]})
    print(f"{'family':16}" + "".join(f"{lab:>14}" for lab in labs))
    for f in fams:
        print(f"{f:16}" + "".join(f"{'%d/%d' % tuple(S[lab]['families'].get(f, [0, 0])):>14}" for lab in labs))
    for key in ("pass", "judged", "machine_printed", "machine_backed", "wrote_block"):
        print(f"{key:16}" + "".join(f"{'%d/%d' % (S[lab][key], S[lab]['n']):>14}" for lab in labs))
    print(f"{'quality% (judge)':16}" + "".join(f"{S[lab]['quality%']:>14}" for lab in labs))
    print(f"{'mean tokens':16}" + "".join(f"{S[lab]['tokens']:>14}" for lab in labs))
    if args.json:
        Path(args.json).write_text(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
