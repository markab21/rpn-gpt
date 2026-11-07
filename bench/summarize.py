"""Score every raw rollout, write bench/summary.json and the chart (bench/chart.png, bench/chart.svg).

    python bench/summarize.py [--convert]
--convert first turns the composite's e2e_vllm outputs (bench/raw/forth/SET.sampled.json, SET.greedy.json) into
bench/raw/forth/SET.RUN.jsonl, the same record shape as run_stock.py (token ids dropped, text kept).
Accuracy per (condition, set): the mean over the 4 sampled runs, with a 95% bootstrap interval over problems
(10,000 resamples, seed 0), and the greedy run on its own.
"""

import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from score import exact, gold_value  # noqa: E402

SETS = ["nupa", "mul_big", "div_big", "add_or_sub_big", "mul_div_multiple_longer"]
CONDS = ["stock", "stock_python", "forth"]
RUNS = ["s0", "s1", "s2", "s3"]
LABEL = {"stock": "Stock Lightning (no tools)", "stock_python": "Stock Lightning + Python tool",
         "forth": "rpn-gpt-lightning-Forth (kernel in the weights)"}
# qwenputer README / chart, commit cb1c969 (Qwen3.5-4B, greedy): NUPA from their CHIP-8 release (576 problems),
# the DeepMind sets from the RISC-V release (40 problems each)
QWENPUTER = {"base": [48.6, 55.0, 72.5, 87.5, 90.0], "qwenputer": [100.0, 97.5, 97.5, 97.5, 100.0],
             "python": [100.0, 100.0, 95.0, 97.5, 97.5]}
LIMIT = [("step cap", "over 196608 kernel steps"), ("overflow past 20 digits", "over 20 digits"),
         ("operand over 10 digits", "top operand over 10 digits"), ("operand over 10 digits", "dividend over 10 digits")]


def convert():
    for s in SETS:
        P = json.load(open(HERE / "sets" / f"{s}.json"))["problems"]
        for kind in ("sampled", "greedy"):
            f = HERE / "raw" / "forth" / f"{s}.{kind}.json"
            if not f.exists():
                continue
            runs = {}
            for r in json.load(open(f))["rows"]:
                p, run = P[r["idx"]], "greedy" if kind == "greedy" else f"s{r['seed'] // len(P)}"
                rec = {"id": p["id"], "set": s, "cond": "forth", "run": run, "temp": 0.0 if kind == "greedy" else 1.0,
                       "question": p["question"], "gold": p["gold"]} | exact(r["text"], p["gold"])
                rec |= {k: r[k] for k in ("text", "tokens", "visible_tokens", "hidden_tokens", "turn_cut",
                                          "forced_think_close_at", "programs", "seed", "safety_cut")}
                rec["kernel_exact"] = r["exact"]  # e2e.check: every kernel-owned position equal to text_vm
                runs.setdefault(run, []).append(json.dumps(rec) + "\n")
            for run, lines in runs.items():
                (HERE / "raw" / "forth" / f"{s}.{run}.jsonl").write_text("".join(lines))


def convert_qwenputer(src):
    """qwenputer's own evalarith records (SRC/<data>.rv1.jsonl, scored by their exact rational match) -> bench/raw/
    qwenputer/SET.greedy.jsonl, token ids dropped. Their 'exact' is kept: their transcripts have no thinking."""
    out = HERE / "raw" / "qwenputer"
    out.mkdir(parents=True, exist_ok=True)
    for s in SETS:
        rows = []
        for f in sorted(Path(src).glob(f"{'nupa_*' if s == 'nupa' else 'dmx_' + s}.rv1.jsonl")):
            for x in open(f):
                r = {k: v for k, v in json.loads(x).items() if k not in ("prompt_ids", "gen_ids")}
                rows.append(json.dumps(r | {"set": s, "run": "greedy"}) + "\n")
        (out / f"{s}.greedy.jsonl").write_text("".join(rows))


def load(cond, s, run):
    f = HERE / "raw" / cond / f"{s}.{run}.jsonl"
    rows = [json.loads(x) for x in open(f)] if f.exists() else []
    for r in rows if cond != "qwenputer" else []:  # rescore with the current scorer (qwenputer: their own score)
        r |= exact(r["text"], r["gold"])
    return {r["id"]: r for r in rows}


def bootstrap(per_problem, n=10000):
    rng, k = random.Random(0), len(per_problem)
    means = sorted(sum(rng.choice(per_problem) for _ in range(k)) / k for _ in range(n))
    return 100 * means[int(0.025 * n)], 100 * means[int(0.975 * n) - 1]


def limits(p):
    """What the problem asks beyond the machine (integers only, 20 digits, * top operand and / dividend 10 digits)."""
    q, g = p["question"], p["gold"].lstrip("-")
    nums = re.findall(r"\d+(?:\.\d+)?", q)
    ip = lambda n: len(n.split(".")[0].lstrip("0") or "0")  # noqa: E731  integer digits
    tags = set()
    if any("." in n for n in nums) or "." in g:
        tags.add("decimal")
    if any(ip(n) > 20 for n in nums + [g.split("/")[0]]):
        tags.add("overflow past 20 digits")
    if p["set"] in ("mul_big", "nupa") and re.search(r"\*|times|product|[Mm]ultiply", q) and len(nums) == 2 \
            and min(map(ip, nums)) > 10:
        tags.add("operand over 10 digits")
    if p["set"] == "div_big" and ip(nums[0]) > 10:
        tags.add("operand over 10 digits")
    return tags


def bucket(r, p):
    """Why a wrong answer is wrong: the first that applies. A machine limit counts when an output block shows its
    error or when the problem itself needs more than the machine has."""
    outs = re.findall(r"```output\n(.*?)```", r["text"], re.S)
    tags = limits(p)
    if r.get("safety_cut"):  # a block still running at e2e_vllm's 200,000-token cap per problem (the machine's own
        return "step cap"     # 196,608-step cap per block had not fired yet)
    for name, needle in LIMIT:
        if any(needle in o for o in outs) or name in tags:
            return name
    if "decimal" in tags:
        return "decimal"
    if r.get("turn_cut") or "</think>" not in r["text"]:
        return "model error: turn cap"
    if any(str(gold_value(p["gold"])) in re.sub(r"\s", "", o) for o in outs if "/" not in p["gold"]):
        return "model error: the machine printed the answer, the model wrote another"
    if not outs:
        return "model error: no machine, wrong by hand"
    return "model error: wrong program or wrong use"


def summary():
    out = {"sets": SETS, "conditions": {}, "qwenputer_published": QWENPUTER}
    for c in CONDS:
        out["conditions"][c] = {}
        for s in SETS:
            runs = {run: load(c, s, run) for run in RUNS}
            ids = sorted(set.intersection(*[set(v) for v in runs.values()])) if all(runs.values()) else []
            per = [sum(runs[run][i]["exact"] for run in RUNS) / len(RUNS) for i in ids]
            g = load(c, s, "greedy")
            P = {p["id"]: p for p in json.load(open(HERE / "sets" / f"{s}.json"))["problems"]}
            wrong = [r for v in runs.values() for r in v.values() if not r["exact"]]
            bk = {}
            for r in wrong:
                b = bucket(r, P[r["id"]])
                bk[b] = bk.get(b, 0) + 1
            out["conditions"][c][s] = {
                "n_problems": len(ids), "problems_over_a_machine_limit": {
                    t: sum(t in limits(p) for p in P.values()) for t in ("decimal", "overflow past 20 digits",
                                                                         "operand over 10 digits")}, "sampled_mean": round(100 * sum(per) / len(per), 1) if per else None,
                "sampled_ci95": [round(x, 1) for x in bootstrap(per)] if per else None,
                "sampled_runs": {run: round(100 * sum(r["exact"] for r in v.values()) / len(v), 1) if v else None
                                 for run, v in runs.items()},
                "greedy": round(100 * sum(r["exact"] for r in g.values()) / len(g), 1) if g else None,
                "greedy_n": len(g), "wrong_sampled": len(wrong), "buckets_sampled": bk,
                "median_tokens": sorted(r["tokens"] for v in runs.values() for r in v.values())[
                    sum(map(len, runs.values())) // 2] if per else None}
            if c == "forth":
                allr = [r for v in [*runs.values(), g] for r in v.values()]
                out["conditions"][c][s]["kernel_exact"] = f"{sum(r.get('kernel_exact', False) for r in allr)}/{len(allr)}"
    out["paired_forth_minus"] = {}  # points, per problem over the 4 sampled runs, 95% bootstrap over problems
    for b in ("stock", "stock_python"):
        out["paired_forth_minus"][b] = {}
        for s in SETS:
            ra, rb = ({run: load(c, s, run) for run in RUNS} for c in ("forth", b))
            ids = sorted(set(ra["s0"]) & set(rb["s0"]))
            d = [sum(ra[run][i]["exact"] - rb[run][i]["exact"] for run in RUNS) / len(RUNS) for i in ids]
            if d:
                lo, hi = bootstrap(d)
                out["paired_forth_minus"][b][s] = [round(100 * sum(d) / len(d), 1), round(lo, 1), round(hi, 1)]
    qp = {s: load("qwenputer", s, "greedy") for s in SETS}
    out["qwenputer_reproduced"] = {s: {"greedy": round(100 * sum(r["exact"] for r in v.values()) / len(v), 1),
                                       "n": len(v)} if v else None for s, v in qp.items()}
    json.dump(out, open(HERE / "summary.json", "w"), indent=1)
    return out


def chart(S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    bg, fg, dim, grid = "#0b0b0b", "#f2f2f2", "#a0a0a0", "#262626"
    cols = {"stock": "#5b6474", "stock_python": "#ecb24a", "forth": "#8a7cf4", "qp": "#3f3a6b"}
    names = ["NUPA", "mul_big", "div_big", "add_or_sub_big", "mul_div_longer"]
    subs = ["long integers", "DeepMind", "DeepMind", "DeepMind", "DeepMind"]
    series = [("stock", "Stock Lightning"), ("stock_python", "Stock Lightning + Python tool"),
              ("forth", "rpn-gpt (kernel in the weights)")]
    fig, ax = plt.subplots(figsize=(16, 8.2), facecolor=bg)
    ax.set_facecolor(bg)
    x, w = np.arange(len(SETS)), 0.2
    for j, (c, lab) in enumerate(series):
        v = [S["conditions"][c][s]["sampled_mean"] or 0 for s in SETS]
        ci = [S["conditions"][c][s]["sampled_ci95"] or [0, 0] for s in SETS]
        err = np.array([[m - lo for m, (lo, hi) in zip(v, ci)], [hi - m for m, (lo, hi) in zip(v, ci)]])
        xs = x + (j - 1.5) * w
        ax.bar(xs, v, w * 0.86, color=cols[c], label=lab, zorder=3)
        ax.errorbar(xs, v, yerr=err, fmt="none", ecolor="#d9d9d9", elinewidth=1.2, capsize=3, zorder=4)
        for xi, vi, (lo, hi) in zip(xs, v, ci):
            ax.text(xi, hi + 1.2, f"{vi:g}", ha="center", va="bottom", color=cols[c] if c != "stock" else "#c8ccd4",
                    fontsize=12.5, fontweight="bold", family="monospace")
    xs = x + 1.5 * w
    qp = [S["qwenputer_reproduced"][s]["greedy"] for s in SETS]
    ax.bar(xs, qp, w * 0.86, color="none", edgecolor="#8a7cf4", hatch="///", linewidth=1.2,
           label="Qwenputer, our rerun (Qwen3.5-4B, greedy)", zorder=3)
    for xi, vi in zip(xs, qp):
        ax.text(xi, vi + 1.2, f"{vi:g}", ha="center", va="bottom", color=dim, fontsize=11, family="monospace")
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(colors=dim, labelsize=12, length=0)
    ax.yaxis.grid(True, color=grid, zorder=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axhline(0, color="#444", lw=1)
    ax.set_xticks(x, [""] * len(x))
    for xi, n, sub in zip(x, names, subs):
        ax.text(xi, -6, n, ha="center", va="top", color=fg, fontsize=14, fontweight="bold")
        ax.text(xi, -10.5, sub, ha="center", va="top", color=dim, fontsize=11.5)
    fig.text(0.06, 0.95, "rpn-gpt vs stock Lightning on Qwenputer's arithmetic sets", color=fg, fontsize=21,
             fontweight="bold", ha="left")
    fig.text(0.06, 0.91, "Accuracy (%) on 40 problems per set. Lightning: mean of 4 sampled runs, 95% bootstrap "
             "interval. Qwenputer: greedy, rerun by us on the same problems.", color=dim, fontsize=12, ha="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=4, frameon=False, labelcolor=fg, fontsize=12,
              handlelength=1.2)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.83, bottom=0.12)
    for ext in ("png", "svg"):
        fig.savefig(HERE / f"chart.{ext}", dpi=160, facecolor=bg)


if __name__ == "__main__":
    if "--convert" in sys.argv:
        convert()
    if "--qwenputer" in sys.argv:
        convert_qwenputer(sys.argv[sys.argv.index("--qwenputer") + 1])
    S = summary()
    for c in CONDS:
        for s in SETS:
            d = S["conditions"][c][s]
            print(f"{c:13s} {s:24s} mean {d['sampled_mean']} {d['sampled_ci95']} greedy {d['greedy']} "
                  f"runs {d['sampled_runs']} buckets {d['buckets_sampled']}")
    chart(S)
