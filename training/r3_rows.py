"""Stage-2a r3 rows (SFT only, no RL): the mass-sampling rollouts (training/mass_report.py --merge lists), rebuilt
composite-exact under THIS tree's text_vm with the row prompt (card v3), cleaned, at most 2 rows per problem, capped
per (family, kind) cell, with a by-problem validation split.
  python training/r3_rows.py --pool training/data/mass_pool.json --src B.json:training/data/cards/reference_v2.txt:8000 \
      --src A.json:training/data/cards/reference.txt:4000 --prompt training/data/cards/reference_v3.txt --out DIR [--build]
SRC files are listed in order of preference (stage B, think 8000, first); per problem: machine-backed first, then the
preferred source, then the shortest; rows with a redundant re-run (a block >= 80% the same as an earlier error-free one:
what fills code memory) rank last, or are dropped with --drop-reruns. Accepted rollout:
  correct (the fixed grader, regraded by --merge), and
  easy / easy_big / math cells: a direct answer only (no machine: the DM ablation, 88.7% vs 75.6% with the card);
  other cells: machine-backed (distill_rows.accepted; a yes / no answer may come from a -1 / 0 flag the machine
  printed last) or direct;
  the thinking closed by the model (not force-closed at the rollout's think budget: those rows end mid-thought),
  clean: distill_rows.clean against the generating prompt + rubric_v24 (text the row prompt lacks), no paraphrase of
  reference.txt's wrong operand limit, every ```forth fence
  closed and run, no full re-send after an error (resend_kind 'full').
Never: a problem of a held-out set (frozen_test, demo*, dm_frontier*, dev_codefull, dev_round, math500, s2a_eval10x2). Rows that the current spec streams differently are dropped.
Rollouts of MATH-train problems (kind math) must be graded by training/math_grade.py first (r3_data.sh does it).
Without --build: counts only (candidates per cell, resend kinds)."""
import argparse, collections, glob, hashlib, json, multiprocessing as mp, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
import distill_rows as D  # noqa: E402

F = "`" * 3
HELD = ["frozen_test", "demo*", "dm_frontier*", "dev_codefull", "dev_round", "math500", "s2a_eval10x2"]
DIRECT_KINDS = ("easy", "easy_big", "math")  # direct answers only (math: the MATH-train guard rows, never the machine)
SKIP_FAM = set()  # lcg_trace back since 6e5947e, euclid_trace since round 4 (the grader accepts both readings of the trace)
DECL = re.compile(r"\b(?:variable|constant|value|2variable)\s+(\S+)", re.I)
COLON = re.compile(r"(?:^|\s):\s+(\S+)")


def names(prog):
    return {n.lower() for n in DECL.findall(prog)}, {n.lower() for n in COLON.findall(prog)}


def resend_kind(r):
    """After each error block, what the next block re-sends: 'none' (no error followed by a block), 'trivial' (under two
    names defined so far: nothing to leave out), 'minimal' (no variable re-declared, at most one existing word
    re-defined: the habit step 2 teaches), else 'full' (re-declares variables / re-defines several words already defined:
    the habit that fills code memory). The worst one in the turn counts."""
    seen_v, seen_w, kinds = set(), set(), {"none"}
    bs = r["blocks"]
    for i, b in enumerate(bs):
        v, w = names(b["program"])
        if i and bs[i - 1]["output"].startswith("error"):
            if v & seen_v or len(w & seen_w) >= 2:
                kinds.add("full")
            else:
                kinds.add("minimal" if len(seen_v | seen_w) >= 2 else "trivial")
        seen_v |= v
        seen_w |= w
    return next(k for k in ("full", "minimal", "trivial", "none") if k in kinds)


def flag_backed(r):
    """a yes / no answer read off a -1 / 0 flag the machine printed last (is_prime): distill_rows.accepted with got =
    that flag, so its other checks (no typed outputs, closed thinking, no empty program) still apply"""
    outs = [" ".join(b["output"].split()) for b in r["blocks"]]
    return (bool(r["ok"]) and bool(outs) and outs[-1] and set(outs[-1].split()) <= {"-1", "0"}  # (Miller-Rabin: "-1 -1")
            and D.norm(r["a"]) in ("True", "False")
            and D.accepted(r | {"got": outs[-1]}))


# reference.txt's wrong limit ("* / mod take operands of up to 10 digits"), paraphrased past the 8-word quote filter
OLD_LIMIT = re.compile(r"operands (?:of )?up to 10 digits|mod take operands|take operands of up to", re.I)
def reruns(r):
    """blocks that re-run (>= 80% the same text as) an earlier block that ran without an error: verification re-runs
    and small variants of working code, what fills code memory (the card A/B, 2026-11-05)"""
    import difflib
    ok, n = [], 0
    for b in r["blocks"]:
        p = " ".join(b["program"].split())
        n += any(difflib.SequenceMatcher(None, p, q, autojunk=False).ratio() >= 0.8 for q in ok)
        if not b["output"].startswith("error"):
            ok.append(p)
    return n


def judge(args):
    """-> (accepted?, machine-backed?, resend kind, redundant re-runs) for one rollout"""
    r, kind, card, prompt, budget, fam = args
    if not r["ok"] or len(TOK(r["text"].split("</think>")[0]).input_ids) >= budget:  # thinking force-closed at the budget
        return False, False, None, 0
    if r.get("tokens", 0) > LIM["max_tokens"] or sum(b["output"].startswith("error") for b in r["blocks"]) > LIM["max_errors"]:
        return False, False, None, 0  # (round 7: concise turns, no retry chains)
    direct = D.accepted_direct(r)
    mb = (kind not in DIRECT_KINDS or fam in LIM["machine_fams"]) and (D.accepted(r) or flag_backed(r))
    fences = r["text"].count(F + "forth\n") == len(r["blocks"])
    ok = (direct or mb) and fences and D.clean(r, card, prompt) and not OLD_LIMIT.search(r["text"])
    rk = resend_kind(r)
    return ok and rk != "full", mb, rk, reruns(r)


def build(args):
    q, text = args
    try:
        ids, mask = D.row(D.prompt_text(RUBRIC, q, TOK), text, TOK)
    except ValueError:
        return None
    return ids, mask


LIM = {"max_tokens": 10**9, "max_errors": 10**9, "machine_fams": ()}


def main():
    global RUBRIC, TOK
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--pool", action="append", required=True, help="problem sets (q, family, kind); repeatable")
    a.add_argument("--src", action="append", required=True, help="ROLLOUTS.json:GENERATING_PROMPT.txt:THINK_BUDGET, preferred first")
    a.add_argument("--prompt", required=True, help="the row prompt (system)")
    a.add_argument("--caps", default="{}", help='JSON {"family/kind": cap, "*": default cap, "easy": cap for easy cells}')
    a.add_argument("--per-problem", type=int, default=2)
    a.add_argument("--val-mod", type=int, default=25, help="1 / N of problems (by hash) go to the validation split")
    a.add_argument("--max-len", type=int, default=16384)
    a.add_argument("--tokenizer", default="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16")
    a.add_argument("--workers", type=int, default=64)
    a.add_argument("--build", action="store_true")
    a.add_argument("--repeat", action="append", default=[], help="STAGE:K: train rows from that source K times (upweight)")
    a.add_argument("--drop-reruns", action="store_true", help="no row with a redundant re-run (else they rank last)")
    a.add_argument("--max-tokens", type=int, default=10**9, help="drop rollouts longer than this (the whole turn)")
    a.add_argument("--max-errors", type=int, default=10**9, help="drop rollouts with more error blocks (retry chains)")
    a.add_argument("--machine-family", action="append", default=[], help="an easy-kind family whose machine-backed rows are kept")
    a.add_argument("--only-machine", action="append", default=[], help="FAMILY/KIND: keep only machine-backed rows of this cell")
    a.add_argument("--out", required=True)
    args = a.parse_args()
    LIM.update(max_tokens=args.max_tokens, max_errors=args.max_errors, machine_fams=tuple(args.machine_family))
    from transformers import AutoTokenizer
    TOK = AutoTokenizer.from_pretrained(args.tokenizer)
    RUBRIC = {"system": open(args.prompt).read()}
    v24 = D.grams(json.load(open(ROOT / "training/data/cards/rubric_v24.json"))["system"])
    held = {p["q"] for h in HELD for f in glob.glob(str(ROOT / f"training/data/{h}.json")) for p in json.load(open(f))}
    pool = {p["q"]: p for f in args.pool for p in json.load(open(f))}
    cands, stats = collections.defaultdict(list), collections.Counter()
    with mp.Pool(args.workers) as ex:
        for si, spec in enumerate(args.src):
            path, gen, budget = spec.split(":")
            card = (v24 | D.grams(open(gen).read())) - D.grams(RUBRIC["system"])
            rs = [r for r in json.load(open(path)) if r["q"] in pool and r["q"] not in held
                  and pool[r["q"]]["family"] not in SKIP_FAM]
            for r, (ok, mb, rk, rr) in zip(rs, ex.map(judge, [(r, pool[r["q"]]["kind"], card, RUBRIC["system"], int(budget),
                                                                 pool[r["q"]]["family"]) for r in rs],
                                                     chunksize=64)):
                c = (pool[r["q"]]["family"], pool[r["q"]]["kind"])
                stats[(si, "rollouts")] += 1
                stats[(si, "correct")] += r["ok"]
                stats[(si, "resend_" + str(rk))] += r["ok"] and bool(r["blocks"])
                stats[(si, "accepted")] += ok
                stats[(si, "accepted_with_reruns")] += ok and rr > 0
                stats[(si, "machine_accepted")] += ok and mb
                stats[(si, "machine_accepted_with_reruns")] += ok and mb and rr > 0
                if ok and not mb and f"{c[0]}/{c[1]}" in args.only_machine:
                    ok = False
                if ok and not (rr and args.drop_reruns):
                    cands[r["q"]].append({"src": si, "mb": mb, "len": len(r["text"]), "text": r["text"], "cell": c, "rr": rr})
            print(f"{path}: {len(rs)} rollouts after held-out / family filters", flush=True)
    for k, v in sorted(stats.items()):
        print(k, v)
    caps = json.loads(args.caps)
    cap = lambda c: caps.get(f"{c[0]}/{c[1]}", caps.get(c[0], caps.get(c[1] if c[1] == "math" else "easy" if c[1] in DIRECT_KINDS else "*", 10**9)))
    by_cell = collections.defaultdict(list)
    for q, cs in sorted(cands.items()):
        # no redundant re-run first, then machine-backed (outside the easy cells), preferred source, shortest
        cs.sort(key=lambda x: (x["rr"] > 0, not x["mb"], x["src"], x["len"]))
        by_cell[cs[0]["cell"]].append(q)
    print("cell\tproblems\tcandidates\tmachine_backed\twith_reruns\tcap")
    for c in sorted(by_cell):
        qs = by_cell[c]
        print(f"{c[0]}/{c[1]}\t{len(qs)}\t{sum(len(cands[q]) for q in qs)}\t{sum(x['mb'] for q in qs for x in cands[q])}\t{sum(x['rr'] > 0 for q in qs for x in cands[q])}\t{cap(c)}")
    if not args.build:
        return
    # per cell: problems whose best row has no redundant re-run, a machine-backed one, the preferred source, then a fixed
    # random order (hash), every problem's first row before any second, up to the cap
    order = lambda q: hashlib.sha1(q.encode()).hexdigest()
    pick = []  # (q, candidate)
    for c, qs in sorted(by_cell.items()):
        qs, n = sorted(qs, key=lambda q: (cands[q][0]["rr"] > 0, not cands[q][0]["mb"], cands[q][0]["src"], order(q))), 0
        for k in range(args.per_problem):
            for q in qs:
                if k < len(cands[q]) and n < cap(c):
                    pick.append((q, k))
                    n += 1
    print(f"building {len(pick)} rows", flush=True)
    with mp.Pool(args.workers) as ex:
        built = ex.map(build, [(q, cands[q][k]["text"]) for q, k in pick], chunksize=8)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    out = {"train": [], "val": []}
    drop = collections.Counter()
    for (q, k), b in zip(pick, built):
        x = cands[q][k]
        if b is None or len(b[0]) > args.max_len:
            drop["differs" if b is None else "too_long"] += 1
            continue
        split = "val" if int(order(q), 16) % args.val_mod == 0 else "train"
        out[split].append({"input_ids": b[0], "train_mask": b[1], "q": q, "family": x["cell"][0], "kind": x["cell"][1],
                           "answer": pool[q]["a"], "machine": x["mb"], "reruns": x["rr"], "stage": Path(args.src[x["src"]].split(":")[0]).stem})
    for spec in args.repeat:  # upweight a source in the train split only (its val rows stay single)
        stage, k = spec.split(":")
        out["train"] += [r for r in out["train"] if r["stage"] == stage] * (int(k) - 1)
    for s, rows in out.items():
        with open(f"{args.out}/{s}.jsonl", "w") as f:
            f.write("".join(json.dumps(r) + "\n" for r in rows))
        fam = collections.Counter(f"{r['family']}/{r['kind']}" for r in rows)
        print(f"{s}: {len(rows)} rows, {len({r['q'] for r in rows})} problems, machine {sum(r['machine'] for r in rows)}, "
              f"with redundant re-runs {sum(r['reruns'] > 0 for r in rows)}, stage {dict(collections.Counter(r['stage'] for r in rows))}, trained tokens "
              f"{sum(map(sum, (r['train_mask'] for r in rows)))}, total {sum(len(r['input_ids']) for r in rows)}")
        print(" ", dict(sorted(fam.items())))
    print("dropped", dict(drop))


_e = lambda o: {"output": o}
assert resend_kind({"blocks": [_e("1") | {"program": "variable x : f 1 ;"}]}) == "none"
assert resend_kind({"blocks": [{"program": "variable x : f x @ ; : g f ;", "output": "error: x"},
                               {"program": ": f 2 ; g .", "output": "2"}]}) == "minimal"
assert resend_kind({"blocks": [{"program": "variable x : f x @ ;", "output": "error: x"},
                               {"program": "variable x : f 2 ; f .", "output": "2"}]}) == "full"
assert reruns({"blocks": [{"program": "x @ 7 * .", "output": "21"}, {"program": "x @  7 * . cr", "output": "21"}]}) == 1
assert reruns({"blocks": [{"program": "x @ 7 * .", "output": "error: y"}, {"program": "x @ 7 * .", "output": "21"}]}) == 0
assert resend_kind({"blocks": [{"program": "1 +", "output": "error: x"}, {"program": "1 1 + .", "output": "2"}]}) == "trivial"
_fb = {"ok": True, "a": "True", "got": "\\text{Yes}", "text": "x</think>" + F + "forth\n7 sprp . 11 sprp .\n" + F + "\n" + F + "output\n-1 -1\n" + F + "\ny",
       "blocks": [{"program": "7 sprp . 11 sprp .", "output": "-1 -1"}]}
assert flag_backed(_fb) and not flag_backed(_fb | {"blocks": [{"program": "n .", "output": "2793440213"}]})

if __name__ == "__main__":
    main()
