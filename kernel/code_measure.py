"""Code-memory freeing variants, measured on real turns before any kernel build (as r_measure.py did for R). cwd nemotron/
  python kernel/code_measure.py VARIANT OUT.jsonl FILES...    a turn per line: its block outputs under the variant
  python kernel/code_measure.py compare BASE.jsonl VAR.jsonl  code-full turns avoided, region-full turns, changes
VARIANT: base (the spec), a1 / a2 (a block's end-phase top-level hidden definition is freed after its call when nothing
was compiled after it: a1 from the first whole code head it starts in, a2 with its start padded to a head), d32 / d48
(top-level hidden definitions compile into their own region of 32 / 48 cells, reused by the next one; main memory is
192 minus that), n256 / n384 (a bigger code memory); parts combine (a1+n256); cells (the spec; at a turn's first code
memory full: its cells by owner, top-level hidden code / older copies of re-defined words / live words / hidden shared
words; summed by: python kernel/code_measure.py cellsum OUT.jsonl). FILES: loop_sim results (.json list or .jsonl), records with 'blocks' [{program}] and 'a'."""
import json
import os
import sys
from multiprocessing import Pool

VAR = sys.argv[1]
PART = {p[:1]: p for p in VAR.split("+")}  # parts combine: a1+n256
A, NN = PART.get("a", ""), PART.get("n", "")[1:]
REG = int(PART["d"][1:]) if "d" in PART else 0
if REG or NN:  # nN: a bigger code memory (N cells)
    os.environ["SP_NADDR"] = "1000000" if REG else NN  # (d: one list holds both regions; the checks below are the kernel's)
sys.path[:0] = ["kernel", "."]
import text_vm as T  # noqa: E402
from forth import sp  # noqa: E402

V, STATS = T.V, {"region": 0, "topmax": 0, "segs": [], "first": None}
op, eb, rs = T.TextMachine.opener, T.TextMachine.end_block, T.TextMachine.reset


def opener(self, lw):
    top = not self.ctx
    if top and A == "a2":
        self.m.code += [{"op": V.EXIT}] * (-len(self.m.code) % 16)
    start = len(self.m.code)
    op(self, lw)
    if top:
        self.m.intop, self.m.topstart = True, start


def end_block(self):
    top, flushed = bool(self.ctx) and self.ctx[0]["kind"] == "top", bool(self.flush)
    eb(self)
    if A and top and not flushed and not self.error:
        cut = self.m.topstart + (-self.m.topstart % 16)
        if cut < len(self.m.code):
            del self.m.code[cut:]
            self.commit()


def reset(self):
    old = self.m
    rs(self)
    n = len(self.m.code)
    self.m.ranges = tuple((s, min(e, n)) for s, e in getattr(old, "ranges", ()) if s < n)


class DExec(T.Exec):
    intop, topstart, ranges = False, 0, ()

    def compile_tok(self, t):
        tt = V.DIGITS[V.MD.index(t)] if t in V.MD else V.MINUS if t == V.MMINUS else t
        if tt in V.CELLOPS or ((tt in V.DIGITS or tt == V.MINUS) and self.inlit):
            n = len(self.code)
            if self.intop and n - self.topstart >= REG - 1:  # (the ; always fits, in either region)
                STATS["region"] += 1
                raise T.ForthError("code memory full")
            if not self.intop and n - sum(e - s for s, e in self.ranges) >= 192 - REG - 1:
                raise T.ForthError("code memory full")
        super().compile_tok(t)
        if t == V.SEMI and self.intop:
            self.ranges, self.intop = self.ranges + ((self.topstart, len(self.code)),), False
            STATS["topmax"] = max(STATS["topmax"], len(self.code) - self.topstart)


if A or REG:
    T.TextMachine.opener, T.TextMachine.end_block = opener, end_block
if REG:
    T.Exec, T.TextMachine.reset = DExec, reset
df, ctok = T.TextMachine.define, T.Exec.compile_tok


def define(self, name, kind):  # cells: where each definition starts
    if kind == "def":
        STATS["segs"].append((len(self.m.code), name))
    return df(self, name, kind)


def compile_tok(self, t):
    try:
        ctok(self, t)
    except T.ForthError as e:
        if "code memory full" in str(e) and STATS["first"] is None:  # the turn's first: cells by owner
            segs, n, c = sorted(STATS["segs"]), len(self.code), {"top": 0, "old": 0, "live": 0, "shared": 0, "other": 0}
            c["other"] = min([s for s, _ in segs] + [n])
            for i, (s, name) in enumerate(segs):
                e_ = min(segs[i + 1][0] if i + 1 < len(segs) else n, n)
                kind = "top" if name == "\x01top" else "shared" if name[:1] == "\x01" else \
                    "old" if any(nm == name for _, nm in segs[i + 1:]) else "live"
                c[kind] += max(0, e_ - s)
            STATS["first"] = c
        raise


if VAR == "cells":
    T.TextMachine.define, T.Exec.compile_tok = define, compile_tok
if REG or NN:
    V.RBH[:] = V.RBH + [V.RBH[-1]] * 70000  # (emulation addresses pass 191: a rebind's hidden tokens only)
tok = sp.tokenizer()


def run(job):
    STATS["region"] = STATS["topmax"] = 0
    STATS["segs"], STATS["first"] = [], None
    try:
        outs = T.blocks(job["blocks"], tok)
    except Exception as e:  # (counted, never silent)
        outs = [f"CRASH {type(e).__name__}: {e}"]
    return {"k": job["k"], "a": job["a"], "outs": outs, "region": STATS["region"], "topmax": STATS["topmax"],
            "first_full": STATS["first"]}


def load(files):
    for f in files:
        rows = [json.loads(x) for x in open(f)] if f.endswith(".jsonl") else json.load(open(f))
        for i, r in enumerate(rows):
            bl = [b["program"] for b in r.get("blocks", []) if b["program"]]
            if bl:
                yield {"k": f"{os.path.basename(f)}:{i}", "a": str(r["a"]), "blocks": bl}


def compare(base, var):
    B = {r["k"]: r for r in map(json.loads, open(base))}
    full = lambda r: any("code memory full" in o for o in r["outs"])
    ans = lambda o, a: " ".join(o.replace(",", " ").split()) == " ".join(a.split())
    nf = avoided = region = blocks = changed = lost = gained = crash = 0
    for r in map(json.loads, open(var)):
        b = B[r["k"]]
        nf += full(b)
        avoided += full(b) and not full(r)
        region += r["region"] > 0
        crash += r["outs"][:1] != [] and r["outs"][0].startswith("CRASH")
        blocks += len(b["outs"])
        for p, q in zip(b["outs"], r["outs"]):
            changed += p != q
            lost += ans(p, r["a"]) and not ans(q, r["a"])
            gained += ans(q, r["a"]) and not ans(p, r["a"])
    print(f"{var}: {len(B)} turns, {blocks} blocks; code-full turns (base) {nf}, avoided {avoided} "
          f"({100 * avoided / max(nf, 1):.1f}%); turns with a region-full error {region} ({100 * region / len(B):.2f}%); "
          f"outputs changed {changed}; answers lost {lost}, gained {gained}; crashed turns {crash}")


if __name__ == "__main__":
    if VAR == "compare":
        compare(sys.argv[2], sys.argv[3])
    elif VAR == "cellsum":  # share of the cells at the first full, summed over the turns that filled code memory
        F = [r["first_full"] for r in map(json.loads, open(sys.argv[2])) if r.get("first_full")]
        tot = {k: sum(f[k] for f in F) for k in F[0]} if F else {}
        print(f"{len(F)} turns filled code memory; cells by owner:",
              {k: f"{100 * v / max(sum(tot.values()), 1):.1f}%" for k, v in tot.items()})
    else:
        with Pool(int(os.environ.get("PROCS", 32))) as pool, open(sys.argv[2], "w") as out:
            for r in pool.imap(run, load(sys.argv[3:]), chunksize=4):
                out.write(json.dumps(r) + "\n")
