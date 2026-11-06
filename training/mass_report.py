"""Mass-sampling map: per-problem and per-(family, kind) pass rate, machine-backed rate, accepted rate (the stage-2a
row rules: correct, machine-backed or direct, clean vs card rubric_v24 under reference.txt), median tokens, capped,
multiple </think>. Every rollout is regraded with this tree's grader. Rollouts that hit a server error are not counted (rerun them: see --drop-errors).
  python training/mass_report.py PROBLEMS.json OUT_PREFIX ROLLOUTS.json[l] ...
    -> OUT_PREFIX.problems.jsonl (one line per problem), OUT_PREFIX.families.tsv, table on stdout
  python training/mass_report.py --merge OUT.json ROLLOUTS...   (one list for distill_rows; drops leaky retry successes)
  python training/mass_report.py --drop-errors PARTIAL.jsonl   (rewrites it without server-error rollouts, so a rerun of the
    same loop_sim command redoes them)"""
import collections, json, os, re, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
import distill_rows as D  # noqa: E402

CAP = int(os.environ.get("MASS_CAP", 8000))  # --max-tokens of the runs


def server_error(r):
    return any(b["output"].startswith("server error") for b in r["blocks"])


def regrade(r):
    """ok again with THIS tree's grader (loop_sim.same; offline: the saved got / a / family), on every attempt"""
    for x in [r] + r.get("attempts", []):
        x["ok"] = D.loop_sim.same(x["got"], x["a"], x)
    return r


def load(f):
    return [regrade(json.loads(x)) for x in open(f)] if f.endswith("l") else [regrade(r) for r in json.load(open(f))]


if sys.argv[1] == "--drop-errors":
    rs = [json.loads(x) for x in open(sys.argv[2])]  # raw: the rollout file is not regraded
    keep = [r for r in rs if not server_error(r)]
    open(sys.argv[2], "w").write("".join(json.dumps(r) + "\n" for r in keep))
    print(f"dropped {len(rs) - len(keep)} server-error rollouts, kept {len(keep)}")
    sys.exit()

# a successful retry is generated after the feedback turns, but its row has only the original prompt: drop it when its
# text cites that history ("the previous attempt gave 227277"), or the row teaches citing attempts that never happened
RETRY_REF = re.compile(r"\b(previous|earlier|last|first|prior|other)\s+(attempt|answer|response|try|tries|solution|run)|"
                       r"\btry again\b|\bnot correct\b|\bincorrect\b|\bfeedback\b|\b(user|they) (says?|said)\b|\bagain\b", re.I)
if sys.argv[1] == "--merge":  # --merge OUT.json FILES...: one rollout list for distill_rows (no server errors, no leaky retries)
    rs = [r for f in sys.argv[3:] for r in load(f) if not server_error(r)]
    keep = [r for r in rs if len(r.get("attempts", [r])) == 1 or not RETRY_REF.search(r["text"])]
    json.dump(keep, open(sys.argv[2], "w"))
    print(f"{len(rs)} rollouts, {len(rs) - len(keep)} retry successes / finals citing earlier attempts dropped, {len(keep)} kept")
    sys.exit()

probs, prefix, files = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3:]
ref = open(ROOT / "training/data/cards" / os.environ.get("MASS_REF", "reference.txt")).read()  # the rollouts' prompt
card = D.grams(json.load(open(ROOT / "training/data/cards/rubric_v24.json"))["system"]) - D.grams(ref)
st = {p["q"]: collections.Counter() for p in probs}
toks, errs = collections.defaultdict(list), 0
for f in files:
    for r in load(f):
        if r["q"] not in st:
            continue
        if server_error(r):
            errs += 1
            continue
        fin, r = r, r.get("attempts", [r])[0]  # --retries: the map counts first attempts; ok_final / acc_final the last
        s, m = st[r["q"]], r["ok"] and D.accepted(r)
        s["ok_final"] += fin["ok"]
        s["acc_final"] += fin["ok"] and (D.accepted(fin) or D.accepted_direct(fin)) and D.clean(fin, card, ref)
        s["n"] += 1
        s["ok"] += r["ok"]
        s["mb"] += m
        s["uses"] += bool(r["blocks"])
        s["acc"] += r["ok"] and (m or D.accepted_direct(r)) and D.clean(r, card, ref)
        s["capped"] += r["tokens"] >= CAP - 50 and r["got"] is None
        s["multi"] += r["text"].count("</think>") > 1
        toks[r["q"]].append(r["tokens"])
rows = [{"q": p["q"], "family": p["family"], "kind": p["kind"], **st[p["q"]],
         "tok_med": statistics.median(toks[p["q"]]) if toks[p["q"]] else None} for p in probs if st[p["q"]]["n"]]
with open(prefix + ".problems.jsonl", "w") as f:
    f.write("".join(json.dumps(x) + "\n" for x in rows))
fam = collections.defaultdict(list)
for x in rows:
    fam[(x["family"], x["kind"])].append(x)
head = "family\tkind\tproblems\trollouts\tpass\tmachine_backed\taccepted\tuses_machine\tcapped\tmulti_think\tpass_final\tacc_final\tmed_tokens\tp_all0\tp_mixed\tp_all1"
lines = [head]
for (f, k), xs in sorted(fam.items()):
    n = sum(x["n"] for x in xs)
    rate = lambda key: sum(x[key] for x in xs) / n
    mixed = sum(0 < x["ok"] < x["n"] for x in xs)
    lines.append("\t".join(map(str, [f, k, len(xs), n] + [f"{rate(key):.3f}" for key in ("ok", "mb", "acc", "uses", "capped", "multi", "ok_final", "acc_final")]
                                + [statistics.median(t for x in xs for t in toks[x["q"]]), sum(x["ok"] == 0 for x in xs), mixed,
                                   sum(x["ok"] == x["n"] for x in xs)])))
n = sum(x["n"] for x in rows)
lines.append("\t".join(map(str, ["ALL", "", len(rows), n] + [f"{sum(x[key] for x in rows) / max(n, 1):.3f}" for key in
                                                               ("ok", "mb", "acc", "uses", "capped", "multi", "ok_final", "acc_final")])))
open(prefix + ".families.tsv", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"server-error rollouts skipped: {errs}")
