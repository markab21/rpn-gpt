"""Tests for the stored-program kernel: build, save, reload with from_pretrained, then check every kernel-owned
position (argmax) against the reference token machine (sp_vm.py) in one-shot prefill, token-by-token cached decode,
and plain generate() per kernel run.
  python sp_test.py OUT_DIR block [N_STREAMS] [DEPTH]     (stage 1: row store / copies / print / clears / carries)
  python sp_test.py OUT_DIR handoff     model-written tokens one decode step at a time; the kernel claims only its own
                                         positions (forth/sp.py streams of Hanoi 1-5, blackjack); unset variable reads
"""

import json
import os
import random
import sys
import time

import torch
from transformers import AutoModelForCausalLM

import sp_kernel as K
import sp_vm as V

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # forth/ (sp.py, ref.py)


def flatten(runs):
    ids, want = [], {}
    for user, model in runs:
        ids += user
        for t in model:
            want[len(ids) - 1] = t
            ids.append(t)
    return ids, want


def lit(v):
    return [V.LIT] + [V.MINUS] * (v < 0) + [V.DIGITS[int(c)] for c in str(abs(v))]


def block_stream(rng, depth, n_ops):
    """Random user ops on the data stack: literals, DUP OVER DROP PICK '.', climbing to `depth` and back."""
    m, runs = V.Machine(), []
    ops = 0
    up = True
    while ops < n_ops:
        s = m.stack
        if len(s) >= depth:
            up = False
        if len(s) <= 2:
            up = True
        u = rng.random()
        if not s or (u < 0.25 and m.lo < 150 and up):
            nd = rng.randint(1, 20 if depth <= 40 else 2)  # deep streams: short values (token budget)
            v = rng.randint(10 ** (nd - 1), 10**nd - 1) * rng.choice([1, 1, -1]) if rng.random() > 0.05 else 0
            user = lit(v)
        elif u < 0.6 and up and len(s) >= 2:
            user = [rng.choice([V.DUP, V.OVER])]
        elif u < 0.6 and up:
            user = [V.DUP]
        elif u < 0.7 and 3 <= len(s) < depth - 1 and m.lo < 150:  # (the index literal is a push too)
            k = rng.randint(0, min(9, len(s) - 2))
            user = [V.LIT, V.DIGITS[k], V.PICK]
        elif u < 0.85 or not up:
            user = [rng.choice([V.DROP, V.PR])]
        else:
            user = [V.DUP]
        runs += V.run(m, [user])
        ops += 1
    return runs, m


CHECKS = os.environ.get("SP_CHECKS", "build,prefill,cached,generate").split(",")


@torch.no_grad()
def check(m, runs, gen=True):
    ids, want = flatten(runs)
    x = torch.tensor([ids], device="cuda")
    pos = torch.tensor(sorted(want), device="cuda")
    tgt = torch.tensor([want[p] for p in sorted(want)], device="cuda")
    r = {"tokens": len(ids), "kernel_tokens": len(want)}
    t0 = time.time()
    if "prefill" in CHECKS and os.environ.get("SP_SKIP") != "prefill":
        lg = m(x).logits[0, pos].float()
        r["prefill_wrong"] = int((lg.argmax(-1) != tgt).sum())
        top2 = lg.topk(2, -1).values
        r["prefill_min_margin"] = float((top2[:, 0] - top2[:, 1]).min()) if len(pos) else None
        if r["prefill_wrong"]:
            wrong = [(int(p), int(t), int(a)) for p, t, a in zip(pos, tgt, lg.argmax(-1)) if t != a][:5]
            r["prefill_first_wrong (pos, want, got)"] = wrong
    if "cached" in CHECKS:
        r |= cached(m, x, want)
    if gen and "generate" in CHECKS:
        r |= generate(m, runs)
    r["s"] = round(time.time() - t0, 1)
    return r


@torch.no_grad()
def cached(m, x, want):
    r = {}
    out = m(x[:, :1], use_cache=True)
    am = [int(out.logits[0, -1].argmax())]
    for t in range(1, x.shape[1]):
        out = m(x[:, t:t + 1], past_key_values=out.past_key_values, use_cache=True)
        am.append(int(out.logits[0, -1].argmax()))
    bad = [p for p in sorted(want) if am[p] != want[p]]
    r["cached_wrong"] = len(bad)
    if bad:
        r["cached_first_wrong (pos, want, got)"] = [(p, want[p], am[p]) for p in bad[:5]]
    return r


@torch.no_grad()
def generate(m, runs):
    r = {}
    if True:
        ctx, nbad = [], 0
        for user, model in runs:
            ctx += user
            if model:
                o = m.generate(torch.tensor([ctx], device="cuda"), max_new_tokens=len(model), do_sample=False,
                               min_new_tokens=len(model), eos_token_id=None, pad_token_id=0)[0, len(ctx):].tolist()
                nbad += o != model
                ctx += model
        r["generate_runs"] = sum(1 for _, b in runs if b)
        r["generate_runs_wrong"] = nbad
    return r


def arith_stream(rng, n_ops, mulmod=False, words=False):
    """Random signed arithmetic / compare / copy ops (stage 2), incl. long carries, borrows, equal prefixes.
    words: also / NEGATE SWAP, literals <= 10 digits (/'s limit), often exactly 10."""
    m, runs = V.Machine(), []
    big = lambda v: len(str(abs(v))) > 20
    while len(runs) < n_ops:
        s, u = m.stack, rng.random()
        if len(s) < 2 or (u < 0.3 and len(s) < 12):
            if s and rng.random() < 0.4:
                x = abs(s[-1])
                v = rng.choice([x, x + 1, max(0, x - 1), 10 ** len(str(x)) - x, int("9" * len(str(x))), 1, 0])
            else:
                nd = rng.choice(list(range(1, 11)) + [10] * 4) if words else rng.randint(1, 19)
                v = rng.randint(10 ** (nd - 1), 10**nd - 1) if rng.random() > 0.05 else 0
            v = v * rng.choice([1, -1]) if v else 0
            if big(v):
                continue
            user = lit(v)
        else:
            op = rng.choice([V.ADD, V.SUB, V.LT, V.GT, V.INC, V.DECR, V.ZEQ, V.DUP, V.OVER, V.DROP, V.PR] +
                            [V.MUL, V.MOD] * (2 if mulmod else 0) + [V.DIV] * 4 * words + [V.NEG, V.SWAP] * 2 * words)
            x, y = s[-2], s[-1]
            if big(x + y) or big(x - y) or big(y + 1) or big(y - 1):
                continue
            if op in (V.MUL, V.MOD) and (len(str(abs(y))) > 10 or big(x * y) or y == 0):
                continue
            if op == V.DIV and (len(str(abs(x))) > 10 or len(str(abs(y))) > 10 or y == 0):
                continue
            user = [op]
        runs += V.run(m, [user])
    return runs


def forth(what, args):
    """ref.py programs as kernel runs: {name: (runs, expected printed text per command)}."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../forth"))
    from ref import PROGRAMS, Forth
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(K.LIGHTNING)
    src, case = PROGRAMS.get(what.split("_")[0], GCD), {"hanoi": str.upper}.get(what, str.lower)
    if what.startswith("hanoi"):  # ' HANOI' = 4 tokens, ' hanoi' = 3: names keyed by their first token
        cases = {f"{what}{n}": [f"{n} 1 3 HANOI"] for n in args}
    elif what == "gcd":  # ' gcd' = 2 tokens; SWAP NEGATE / inside definitions and at top level
        cases = {f"gcd{a}_{b}": ["ACC @ .", f"{a} {b} GCD .", f"{a} {b} MID .", f"{a} {b} FLIP . .",
                                 f"{a} {b} SWAP / .", f"ACC @ {a} + ACC ! ACC @ ."]  # ACC: read before any store
                 for a, b in (x.split(",") for x in args)}
    else:  # lowercase: uppercase ' DEAL' / ' DEALER' share their first token ' DE'
        cases = {f"blackjack{s}": [f"{s} SEED !", "DEAL", "HIT", "STAND"] for s in args}
    out = {}
    for name, cmds in cases.items():
        f = Forth()
        f.run(src)
        want = []
        for c in cmds:
            n0 = len(f.out)
            f.run(c)
            want.append(" ".join(f.out[n0:]))
        m, runs, got = V.Machine(), [], []
        for i, c in enumerate([src] + cmds):  # one user token per run: the kernel speaks right after any token
            r = V.run(m, [[t] for t in V.source_tokens(c, tok, case)])
            runs += r
            got += [tok.decode([t for _, mm in r for t in mm], skip_special_tokens=True).strip()] * bool(i)
        assert got == want, (name, got, want)
        out[name] = (runs, want)
    return out


def overflow_stream():
    """Fill all 1024 data positions with DUP, then one more DUP: the kernel must emit the overflow token."""
    m = V.Machine()
    runs = V.run(m, [lit(7)] + [[V.DUP]] * (V.NPOS - 1))
    assert len(m.stack) == V.NPOS
    return runs + [([V.DUP], [V.OVF])]


def value_overflow_streams():
    """Values over 20 digits (sp_vm: OVF from the token that made one, at every position after it): +, -, 1+, 1-,
    a literal's 21st digit, * (carry out of the top digit, and T = 10 r past 20 digits), a factorial loop; each
    preceded by its largest non-overflowing neighbour and followed by '.' (still OVF)."""
    from transformers import AutoTokenizer
    n20, ADD, SUB, INC, DECR, MUL, PR = 10**20 - 1, [V.ADD], [V.SUB], [V.INC], [V.DECR], [V.MUL], [V.PR]
    cases = {"add": [n20 - 1, 1, ADD, PR, 12345678901234567890, 98765432109876543210, ADD],
             "sub": [-5 * 10**19, 49999999999999999999, SUB, PR, -5 * 10**19, 5 * 10**19, SUB],
             "inc": [n20 - 1, INC, PR, n20, INC], "dec": [-n20 + 1, DECR, PR, -n20, DECR],
             "mul11x10": [9999999999, 9999999999, MUL, PR, 12345678901, 9876543210, MUL],  # T past 20 digits
             "mulcarry": [10000000001, 9999999999, MUL, PR, 10000000002, 9999999999, MUL],  # top-digit carry
             "lit": [n20, PR, int("1" * 21)]}
    out = {}
    for name, ops in cases.items():
        m, runs = V.Machine(), []
        for o in ops:  # literals one token per run: the 21st digit must be the one answered with OVF
            runs += V.run(m, [o] if isinstance(o, list) else [[t] for t in lit(o)])
        runs += V.run(m, [[V.PR]])
        out[f"ovf_{name}"] = runs
    tok = AutoTokenizer.from_pretrained(K.LIGHTNING)
    m = V.Machine()
    src = ": FACT 1 SWAP BEGIN DUP WHILE SWAP OVER * SWAP 1- REPEAT DROP ; 21 FACT . 30 FACT . 5 ."
    runs = V.run(m, [[t] for t in V.source_tokens(src, tok, str.lower)])
    assert tok.decode([t for _, k in runs for t in k], skip_special_tokens=True).strip() == "51090942171709440000"
    out["ovf_fact30"] = runs
    for name, runs in out.items():
        ovf = [i for i, (_, k) in enumerate(runs) if V.OVF in k]
        assert ovf and runs[ovf[0]][1][-1] == V.OVF and all(k == [V.OVF] for _, k in runs[ovf[0] + 1:]), name
    return out


GCD = "VARIABLE ACC : GCD BEGIN DUP WHILE SWAP OVER MOD REPEAT DROP ; : MID OVER - 2 / + ; : FLIP SWAP NEGATE ;"
ALL = {"hanoi": ["1", "2", "3", "4", "5"], "hanoi_lower": ["1", "2", "3", "4", "5"],
       "blackjack": ["1", "42", "7", "12345", "2024"], "gcd": ["48,18", "1071,-462", "-999999999,1234567890", "17,0"],
       "arith": ["2", "150"], "mulmod": ["3", "150"], "words": ["4", "200"], "block": ["2", "40"],
       "block250": ["1", "250"], "block1020": ["1", "1020"]}
FORTH = ("hanoi", "hanoi_lower", "blackjack", "gcd")


def programs(what, args):
    """Named token streams (user runs + reference kernel runs)."""
    if what == "block":
        n, depth = int(args[0]), int(args[1])
        return {f"stream{s}": block_stream(random.Random(s), depth, 60 if depth <= 40 else 3 * depth)[0]
                for s in range(n)}
    if what in FORTH:
        return {k: v[0] for k, v in forth(what, args).items()}
    if what.startswith("block") and what != "block":
        return {f"{what}_{k}": v for k, v in programs("block", args).items()}
    if what == "overflow":
        return {"overflow1025": overflow_stream()}
    if what == "overflow20":
        return value_overflow_streams()
    if what == "all":
        return {k: v for w, a in list(ALL.items()) + [("overflow", []), ("overflow20", [])]
                for k, v in programs(w, a).items()}
    if what in ("arith", "mulmod", "words"):
        n, ops = int(args[0]), int(args[1])
        return {f"{what}{s}": arith_stream(random.Random(s), ops, what != "arith", what == "words") for s in range(n)}
    raise ValueError(what)


def sp_cases(tok):
    """Prize programs as training data writes them (forth/sp.py: sp-ready source, single-token names) and their true
    interleaving (user tokens one at a time, the kernel right after any user token): {name: (ids, owned offsets)}."""
    from forth import sp
    from forth.ref import PROGRAMS, Forth
    srcs = {f"hanoi{n}": PROGRAMS["hanoi"] + f" {n} 1 3 HANOI" for n in range(1, 6)}
    srcs |= {f"blackjack{s}": PROGRAMS["blackjack"] + f" {s} SEED ! DEAL HIT STAND" for s in ALL["blackjack"]}
    srcs["gcd"] = GCD + " ACC @ . 1071 -462 GCD . 48 18 MID . 7 -100 FLIP . . 1234567890 -999999999 SWAP / . 5 NEGATE ."
    out = {}
    for name, src in srcs.items():
        p, ids, owned = sp.problems(sp.sp_ready(src, tok), Forth().run(src), tok)
        assert p is None, (name, p)
        out[name] = (ids, owned)
    return out


@torch.no_grad()
def drive(m, user, prefix=(), mine=0.5, limit=30000):
    """Real generation: model-written tokens arrive one at a time (one decode step each); after each, the model's greedy
    tokens are kept while the kernel claims the position (max logit > mine: standalone, an emitted token's logit is >= 1
    and every other is 0; merged, 1024 x that). -> (ids after prefix, owned offsets, max logit where the model writes,
    min max-logit where the kernel writes)."""
    cache, ids, owned, free, held = None, list(prefix), [], [], []

    def step(x):
        nonlocal cache
        o = m(torch.tensor([x], device="cuda"), past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        return o.logits[0, -1].float()

    lg = step(ids) if ids else None
    for t in user:
        if lg is not None:
            free.append(float(lg.max()))
        ids.append(t)
        lg = step([t])
        while float(lg.max()) > mine and len(owned) < limit:
            held.append(float(lg.max()))
            owned.append(len(ids) - len(prefix))
            ids.append(int(lg.argmax()))
            lg = step(ids[-1:])
    return ids[len(prefix):], owned, max(free, default=None), min(held, default=None)


def handoff(m, tok, prefix=(), mine=0.5):
    """Token-by-token handoff: drive() each prize program's user tokens; the kernel must claim exactly sp_vm's
    positions with sp_vm's tokens, and stay silent at every model-written position. Plus reads of never-stored
    variables (sp_vm has no answer: its cells start undefined; ref.py zero-fills)."""
    from forth import sp
    R = {}
    for name, (ids, owned) in sp_cases(tok).items():
        user = [t for i, t in enumerate(ids) if i not in set(owned)]
        t0 = time.time()
        got, gown, free, held = drive(m, user, prefix, mine)
        bad = next((i for i, (a, b) in enumerate(zip(got, ids)) if a != b), None)
        R[name] = {"tokens": len(ids), "owned": len(owned), "exact": got == ids and gown == owned,
                   "first_diff": bad, "model_positions": len(user), "max_logit_where_model_writes": free,
                   "min_logit_where_kernel_writes": held, "s": round(time.time() - t0, 1)}
        print("handoff", name, R[name], flush=True)
    for src in ("VARIABLE x x @ .", "VARIABLE x x @ 5 + .", "VARIABLE x VARIABLE y 7 y ! x @ . y @ .",
                "VARIABLE x 0 x ! x @ ."):
        user = sp.tokens(src, tok)[0]
        got, gown, _, _ = drive(m, user, prefix, mine, limit=200)
        R[f"unset: {src}"] = {"hit_200_token_limit": len(gown) >= 200,
                              "kernel": [tok.convert_ids_to_tokens(got[i]) for i in gown],
                              "printed": tok.decode([got[i] for i in gown], skip_special_tokens=True)}
        print(src, R[f"unset: {src}"], flush=True)
    return R


TEXT_2A = ["3 4 + .", "-17 3 * . 20 3 MOD .", "1 2 3 . . .", "12345678901234567890 .", "-12345678901234567890 .",
           " 12 34 * dup . 5 + .", "7 dup * .\n100 7 / . 100 7 mod .", "5 -3 - . -5 3 - .", "1 2 swap . . 3 4 OVER . . .",
           "10 20 30 drop + .", "0 . -0 . 007 .", "1 2 < . 2 1 < . 1 2 > .", "9 0 swap - . -9 0 swap - .",
           "4 3 2 1 0 pick . 2 pick . . . .", "3\n4\n+\n.", "5 Dup . .", "2 dup * Dup * dup . 1 swap . .", "1 2 +",
           "10 -5\n-\n.\n6  -2 * .", " ".join(f"{10007 * i} ." for i in range(1, 61)),
           "123456789 987654321 * . 9999999999 7 / . -100 7 MOD . 100 -7 / .", "1 . 2 . 3 .\n4 . 5 .", "-0 3 + . -000 . -0 -7 * .", "007 5 * . 0005 . 1 2 + . -0010 -020 * .",
           "7" + " dup" * 199 + " ." * 12 + " 1 2 + .",  # borrow due at a hand-back (after a print)
           "7" + " dup" * 190 + " 5 6 + . . .",  # carry due after a literal
           " ".join(f"{i} ." for i in range(100))]
TEXT_STREAMS = ["Let me compute it.\n```forth\n3 4 + dup .\n```\nThat is 7. Doubling:\n```forth\n2 * .\n```\nDone, 14.",
                "```forth\n```\nNothing ran. ```python\nx = 1\n```\n```forth\n5 . 6\n```\n```forth\n.\n```\nok",
                "Prose with 12 34 + . digits outside.\n```forth\n1 2 + .\n```\n```forth\n" +
                " ".join(["9"] * 60) + " " + " ".join(["+"] * 59) + " .\n```"]


TEXT_2B = [  # piece 2b: multi-token and more built-in words, names, definitions, variables, comments
    "1 1+ . 5 1- . 0 0= . 3 0= . 4 4 = . 4 5 = .",
    "3 4 <> . 3 3 <> . 3 4 <= . 5 4 <= . 3 4 >= . 4 4 >= .",
    "5 0<> . 0 0<> . 5 0> . -5 0> . -5 0< . 0 0< .",
    "0 not . 7 not . 5 invert . true . false . -1 0 and . -1 -1 and . 0 0 or . 0 -3 or .",
    "1 2 2dup . . . . 1 2 nip . 3 4 tuck . . .",
    "7 2* . 7 2/ . -9 abs . 9 abs . 1 2 3 2drop . 5 negate .",
    "1 2 depth . . . 1 . cr 2 . space 3 . cr",
    "1 DUP + . 3 4 SWAP - . 10 NEGATE . 2 3 OVER * * .",
    ": sq dup * ; 5 sq . : cube dup sq * ; 3 cube .",
    ": square-sum dup * swap dup * + ; 3 4 square-sum .",
    "variable x 5 x ! x @ . 7 x ! x @ 1+ .",
    "variable n 6 n ! n 7 * . n .",
    "variable v v .",
    "variable d 4 d ! d @ . d 2 + .",
    "10 value v v . 20 to v v . v 1+ .",
    "5 constant k k 2 * . k k + .",
    "variable cnt 0 cnt ! : inc cnt @ 1+ cnt ! ; inc inc inc cnt @ .",
    "variable total 3 total ! 4 total +! total @ .",
    ": dup 7 ; 1 dup . .",
    "( a b -- c ) 1 2 + . \\ add\n3 . ( done )",
    "variable x 3 x ! x ( fetch ) . 9 .",
    "variable MyVar 12 MyVar ! myvar @ . MYVAR .",
    ": is-big? 100 > ; 150 is-big? . 50 is-big? .",
    "1 2 +",
    "variable q",
    "2 value a 3 value b a b * . 10 to a a b - .",
    "variable a variable b 1 a ! 2 b ! a @ b @ + . b a ! a @ .",
    ": f 1 2 ; : g f + ; g . : h g g * ; h .",
    "\\ only a comment\n7 . \\ x\n8 .",
    "1 . 2 . 3 . cr cr 4 .",
    "space 5 . space",
    ": tw ( n -- 2n ) 2 * ; ( x ) 21 tw .",
    "variable xs 1 xs ! xs @ xs @ + xs ! xs @ .",
]
TEXT_2B_STREAMS = [
    "First:\n```forth\nvariable x 5 x ! : dbl 2 * ; x @ .\n```\nThen:\n```forth\nx @ dbl .\n```\nAnd ```forth\n3 dbl dbl .\n```",
    "```forth\n10 value v\n```\nok\n```forth\nv 1+ to v v .\n```",
]


TEXT_2C = [  # piece 2c: one-level control flow in place, hidden cells, hidden top-level / shared words
    ": f 5 0 do i . loop ; f", "10 0 do i . 3 +loop", "0 3 do i . -1 +loop", "3 3 do i . loop 7 .",
    ": sq 0 swap 1+ 1 do i dup * + loop ; 4 sq .", "10 0 do i 3 mod . loop 4 0 ?do i i * . loop",
    ": grow ( p y -- p ) BEGIN DUP WHILE SWAP DUP 7 * 100 / + SWAP 1- REPEAT DROP ;\n40352 8 grow .",
    ": f 0 begin 1+ dup 5 = until ; f .", "0 begin 1+ dup 3 > until .", ": f 0 begin 1+ dup 4 = until ; f .",
    "variable n 3 n ! begin n @ while n @ . n @ 1- n ! repeat n @ .",
    "1 if 2 . then 3 .", "1 . 0 if 2 . else 4 . then 5 .", ": f dup 0 > if 1 else -1 then ; 5 f . -3 f .",
    "1 if 2 . then : f 9 ; f .", "1 if 2 . then variable v 4 v ! v .", "variable x 1 x ! x if 7 . then",
    "48 18 gcd . 4 6 lcm . -4 6 gcd . 8360 33782 lcm . 0 5 gcd .", ": g gcd ; 12 18 g . 7 5 lcm .",
    ": h lcm gcd ; 4 6 10 h . 9 12 gcd .", "5 ?dup . . 0 ?dup .", ": f ?dup if . then ; 7 f 0 f 9 .",
    "variable a 12 a ! a 18 gcd .", "variable x 4 x ! : f x gcd ; 6 f .", "1 if 4 6 gcd . then variable q 3 q ! q .",
    "17 5 /mod . . -17 5 /mod . . 6 7 3 */ .", "1 2 3 rot . . . 1 2 3 -rot . . .",
    "3 5 min . 3 5 max . -2 -9 min . -2 -9 max .", ": gcd 1 ; 8 12 gcd . .", ": f 1 if exit then 5 . ; f 6 .",
    ": run ( x n -- x ) BEGIN DUP WHILE SWAP 31265 * 65723 + 87171 MOD SWAP 1- REPEAT DROP ;\n32359 9 run .",
    "variable s 0 s ! 11 1 do i s @ + s ! loop s @ .", ": t 1 2 3 rot rot + * ; t . 2 3 4 */ .",
    "5 0 do i . loop cr 3 0 do i 10 * . loop",
    ": pow>max2?buzz 1 ; 5\tpow>max2?buzz . .", ": kk=res<fizz 3 ; kk=res<fizz .", ": barkk9world/n! 4 ; barkk9world/n! .",  # fuzz D3
    ": a$b 1 ; : c$d 2 ; a$b . c$d .", ": a:b 1 ; : c:d 2 ; a:b . c:d .", "variable data]buf 5 data]buf ! data]buf @ .",  # D5
    "space", "( x ) space",  # fuzz D4
]
TEXT_2C_STREAMS = [
    "Use a ```forth block, then ```forth run.\n```forth\n3 0 do i . loop\n```\nok",
    "```forth\n2 ( x\n```\n```forth\n2 3 + .\n```\n```forth\n7 \\ y\n```\n```forth\n1 .\n```",
    "```forth\n: g 6 4 gcd ; 1 .\n```\nok\n```forth\ng . 4 6 gcd .\n```",
]
TEXT_2D = [  # piece 2d-minimal: the kernel stops a block at an error (its two fixed lines) and resets at the next
    ["drop", "64985401 41 mod ."], ["1 2 frob", "depth ."], ["5 0 /", "1 ."], [": f drop ; 3 . f 4 .", "7 ."],  # word;
    [": g 1 frob 2 ;", "g . depth ."], ["1 2 3 0 pick 9 pick", "depth ."], ["3 0 do i . loop frob", "5 ."],  # reset
    ["1 if 2 drop drop then", "7 ."], ["1 2 + . swap", "4 ."], ["1-", "6 ."], ["1 . 2 . over", "3 ."],  # at the next
    ["5 0 mod", "8 3 mod ."], ["variable x x @ 1+ x ! drop drop", "x @ ."], ["frob", "1 2 + ."],  # block
    [": h dup * ; 3 h . 4 h h . + cr", "2 h ."], ["1 2 3 4 5 frob", "depth . 1 ."],
    [": f 1 +", "2 ."], ["1 if 2 .", "3 ."], ['5 ." hi"', "1 ."], ["1 2 ;", "3 ."], ["1 if 2 . then ; 4 .", "5 ."],
    [": f 1 if 2 ; 3 .", "4 ."], ["then 3 .", "1 ."], [": f 1 loop ;", "2 ."], [": f 1 if loop ;", "2 ."],
    ["1 if 2 . then repeat", "3 ."], ["i .", "1 ."], [": f : g ;", "1 ."], ["5 @ .", "1 ."], ["3 0 do 5 ! loop", "1 ."],
    ["1 if 7 . then frob", "2 ."], ["2 type", "3 ."],
    ["3 0 do i . loop", "2 0 do i . loop", "1 if 5 . then : f 9 ; 2 0 do i f + . loop"],  # top-level definitions again
    ["\\ a value here\n392 7 * .", "1 ."],  # a declaring word in a comment
    ["5 constant k 6 to k", "k ."], ["1 2 . 5 to y 3 .", "4 ."], [": f 1 ; 5 to f", "f ."], ["5 to dup", "1 ."],  # toval
    ["variable v 7 to v v @ .", "2 ."], [": g 5 to kk ;", "1 ."], ["1 if 5 to zz then", "2 ."],
    ["3 . 1 if 2 . then 9 to zz", "5 ."], ["4 value w : h w 1+ to w ; h h w .", "w ."],
    ["2 12345678901 * .", "1 ."], ["12345678901 3 / .", "1 ."], ["12345678901 0 / .", "1 ."],  # mulop / divop
    ["5 12345678901 mod .", "1 ."], ["1 2 3 12345678901 12345678901 / .", "depth ."], ["12345678901 2 * . 7 .", "1 ."],
    [": f 12345678901 7 / ; 1 . f . 2 .", "3 ."], ["1 0 do 12345678901 5 / . loop", "4 ."], ["9999999999 7 / .", "2 ."],
    ["1 2 3 .s", ".s 9 .", ": f .s ; 1 .", "3 spaces 4 .", "7 >r 8 r> . .", ": g r@ ; 5 .", "2 ."],  # not words (dda501f:
    # .s spaces >r r> r@ dropped from the spec: unknown words)
    [": a " + "1 + " * 63 + "; 1 ."] + [f": g{k} 1 ;" for k in range(75)] + ["7 .", ": h 2 ;", "8 ."],  # code memory
    # full past 75 resets: each reset's ; adds a cell past the end (WA 192 + 75), the check stays on
    [": a " + "1 + " * 57 + "dup ; 1 .", "3 0 do i . loop 7 .", "8 ."],  # code full at a top-level cell: the error
    # at the cell (the finished loop before it is not run first)
    [f"{100 + k} constant a a ." for k in range(33)] + ["a 1+ .", "variable b"],  # a 33rd variable cell: 'vars' at the
    # name (the old binding stays), and at every declaration after
    [": a " + "1 + " * 63 + "; 1 .", ": g 1 ;", ": g 2 ;", ": a2 3 ;", "0 a . 1 ."],  # a ':' name with no free cell: code
]  # full at the name (g is never bound past the end, so declaring it again needs no rebind)
TEXT_A1 = [  # a1: the last block's dead top-level code is freed at the next block's first token (<free_h>)
    [": a " + "1 + " * 20 + "; 1 .", "3 0 do i . loop", "2 0 do i . loop", ": b 5 ; b ."],  # freed from 64, reused
    ["3 0 do i . loop", "4 .", "1 if 5 . then"],  # top at 0: h 0 (all of code memory), then a block with no code
    ["3 0 do i ?dup drop loop 7 .", "4 ."],  # a shared word defined after it (the end phase): not freed (gcd / lcm
    # of 2-digit values inside top-level code are wrong on the kernel before a1 too: '3 0 do i . loop 12 8 gcd .')
    ["3 0 do i . loop frob", "4 .", "1 if 2 . then"],  # an error: not freed
    ["1 if 2 . then variable x 5 x ! x @ .", "4 ."],  # closed mid-block by variable, nothing compiled after: freed
    ["1 if 2 . then : f 3 ; f .", "4 ."],  # a definition after it: not freed
    [": a 1 2 + ; 1 if 2 . then", "5 .", "a .", "1 if 3 . then"],  # cut 16 >= its end: nothing to free (kept on)
    [": a " + "1 + " * 40 + "; 1 ."] + ["3 0 do i . loop 7 ."] * 6 + ["8 ."],  # filled memory without a1
    ["3 0 do i . loop", ": f 1 2 + ; frob", "f .", "2 0 do i . loop", "f ."],  # an error after a free: reset keeps it
    ["1 if 3 4 gcd . then variable q", "5 ."],  # closed mid-block by variable with a shared word pending: not freed
]
TEXT_EP = [  # the end phase compiles machine digits (FON off after the closing ```): gcd / lcm first used in top-level
    # code closed by the block's end (abs's 0); a definition's last word a number (closed before the fence: controls)
    ["1 if 3 4 gcd . then", "4 ."], ["3 0 do i . loop 12 8 gcd .", "4 ."], ["2 0 do 12 8 lcm . loop", "4 ."],
    ["1 if 12 8 gcd . then", "4 ."], ["1 if -6 4 lcm . then", "2 ."], ["1 if 5 then 12", ". . 3 ."], ["3 0 do i . loop -7", ". 1 ."],
    [": f 1 2", "3 ."], ["1 if 3 4 gcd . then variable q", "5 ."], ["12 8 gcd drop 1 if 12 8 gcd . then", "4 ."],  # controls
    ["1 if 5 abs . then", "4 ."], ["3 0 do i ?dup drop loop 7 .", "4 ."], [": g 12 8 gcd ; g .", "1 ."],
]
TEXT_NM = [  # a control word declared as a name: 'collide' at the name (text_vm.CTLNAMES); recurse: an unknown word
    ["variable i 5 i ! i @ .", "3 0 do i . loop"], ["7 value i i .", "1 ."], ["3 constant leave", "2 ."], [": loop 1 ;", "4 ."],
    [": f 3 0 do i . loop ; variable i f", "f"], ["1 if 2 . then variable do", "5 ."], [": exit 1 ;", ": g 2 ; g ."],
    [": g dup 0 > if dup . 1- recurse then ; 3 g drop", "1 ."], [": recurse 7 ; recurse .", "1 ."],
    [": fact dup 1 > if dup 1- fact * then ; 5 fact .", "1 ."], ["variable j 5 j ! j @ .", "2 to i"],
    [": f 12 8 gcd frob ;", "84 36 gcd ."], [": g 5 ?dup frob ;", ": h ?dup ; 3 h . ."],  # a shared word first used in a
    # definition that failed is not defined (the kernel's DECL): the next use defines it
]
TEXT_LONG = [  # one block of ~170k kernel tokens (a counter that grew per token broke the decodes at ~80k: NOTEJ)
    ["variable s 1400 0 do i s +! loop s @ .", "7 ."],
]
TEXT_2F = [  # item 2: value limits, caps (text_vm: digits, code, prints, full, rfull)
    ["99999999999999999999 1 + .", "7 ."], ["99999999999999999999 1+ .", "2 ."], ["123456789012345678901 .", "3 ."],
    [": f 123456789012345678901 ; 1 .", "4 ."], ["12345678901 1234567890 * .", "5 ."], ["-99999999999999999999 1 - .", "6 ."],
    ["99999999999999999999 1- 1+ 1+ .", "8 ."], ["1 . 99999999999999999999 99999999999999999999 + .", "depth ."],
    ["1234567890 1234567890 * .", "9 ."], ["3 0 do 99999999999999999998 i + . loop", "1 ."],
    [": a " + "1 + " * 30 + "; 1 .", ": b " + "1 + " * 30 + "; 2 .", ": c 1 " + "1 + " * 10 + "; 5 .", "3 . 2 a ."],  # code
    ["129 0 do i . loop", "1 ."], ["65 0 do i . cr loop", "2 ."], ["1 . " * 127 + "cr space 5 .", "3 ."],  # prints
    ["128 0 do i . loop", "4 ."], ["1 . " * 128 + "drop", "6 ."],
    ["1 begin dup again", "depth ."], [": f 1 begin 2 again ; f", "1 ."], ["1" + " dup" * 1021 + " 2 *", "1 ."],  # full (<depth> covers depths < 192)
    ["1" + " dup" * 1020 + " 2 * .", "5 ."], ["1" + " dup" * 1022 + " 2 swap", "3 ."],
    [": f f ; f", "1 ."], [": g 1 g ; g", "depth ."], [": h 1+ h ; 0 h", "7 ."],  # rfull
    ["3 0 do i . loop " * 40, "1 ."],  # many machine tokens: <tick> / <tick2> placement
]
# the token cap: training/data/kernel_turns_cap.json (~200k tokens each: prefill only)
TEXT_2E = [  # in-place rebind: a name declared again replaces its dictionary entry (text_vm.rebind)
    ": f 1 ; : f 2 ; f .", ": foo 5 ; : bar foo ; : foo 6 ; bar . foo .", ": f 1 ; variable f 3 f ! f @ .",
    ": f 1 ; : g f ; variable f 2 f ! g . f .", "variable x 5 x ! variable x x @ . 7 x ! x @ .",
    "5 value v 7 value v v . 3 to v v .", "5 constant c 6 constant c c .", "5 constant c variable c 4 c ! c @ .",
    "variable q : q 9 ; q .", ": f 1 ; : f 2 ; : f 3 ; f .", "2 constant k 5 value k k . 8 to k k .",
    ": a 1 2 3 4 5 6 7 8 9 10 + + + + + + + + + ; : b a 1 + ; : c b b + ; : b a 2 + ; b . c .",
    ": w0 1 2 3 4 5 6 7 8 9 + + + + + + + + ; : w1 w0 w0 + ; : w2 w1 w1 + ; : w3 w2 w2 + ; : w1 w0 1+ ; w3 . : w0 7 ; w3 .",
    ": sq dup * ; 4 sq . : sq dup dup * * ; 4 sq .", "variable a variable b 1 a ! 2 b ! variable a 3 a ! a @ b @ + .",
]
NVARS = 32  # the kernel's named variable cells (2b-2d: 8)
TEXT_2G = [  # 32 variable cells: <@x> / <!x> for cells 8..31
    " ".join(f"variable v{i}" for i in range(12)) + " " + " ".join(f"{i * 7} v{i} !" for i in range(12)) +
    " v11 @ v3 @ + . v8 @ . 5 v9 +! v9 @ .",
    " ".join(f"{i} value w{i}" for i in range(32)) + " w31 . w8 w30 + . 99 to w20 w20 .",
    " ".join(f"variable a{i}" for i in range(20)) + " : f a19 @ 1+ a19 ! a10 @ a19 @ + ; 3 a10 ! f . f . a19 .",
    " ".join(f"{i} constant c{i}" for i in range(16)) + " c15 c9 * .",
]
TEXT_2H = ["5 .", "drop", "1 2 +", "variable v", "1 if 2", "3 0 do i . loop", "1 . frob", "1 . cr"]  # fence endings
TEXT_2N = [  # nesting in place (a control stack of 4 levels), leave, loops by loop depth
    ": f 1 if 2 if 3 . then then ; f", ": f dup 0 > if dup 2 mod if 1 . else 2 . then then drop ; 5 f 4 f -1 f",
    ": f 3 0 do i 2 mod if i . then loop ; f", "3 0 do 2 0 do i . loop loop", "4 0 do i 3 0 do dup . loop drop loop",
    ": f 10 0 do i . i 3 = if leave then loop 99 . ; f", "10 0 do i 2 = if leave then i . loop 7 .",
    ": f 0 begin dup . 1+ dup 3 = if leave then again . ; f", "0 begin 1+ dup 5 > if leave then dup 7 < while repeat .",
    ": f 10 0 do i 3 = if leave then i 5 = if leave then i . loop ; f", "3 0 do 5 0 do i 2 = if leave then i . loop cr loop",
    "1 if 0 begin dup 3 < while 1+ repeat . then", "3 0 do 0 begin 1+ dup i > until . loop",
    ": f 3 0 do i 1 = if exit then i . loop 7 . ; f 9 .", ": f 1 if 1 if 1 if 1 if 5 . then then then then ; f",
    ": f 0 begin 1+ dup 4 = if exit then repeat ; f .", ": f 2 0 do 2 0 do 2 0 do 1 . loop loop loop ; f",
    "2 0 do i . 2 0 do i . loop loop", "5 0 do i 2 mod 0= if i . then loop", ": g 3 0 do i 12 gcd . loop ; g",
    ": f 0 swap 0 ?do i 2 mod if i + else 1+ then loop ; 10 f .", "variable n 0 n ! 5 0 do 3 0 do 1 n +! loop loop n @ .",
    ": f begin dup 0 > while dup 2 mod if 1- else 2 / then dup . repeat drop ; 9 f",
    ": f 5 0 do i 0 = if 1 . else i 4 = if leave else i . then then loop 8 . ; f",
]
TEXT_2N_TURNS = [[": g 1 if 1 if 1 if 1 if 1 if 5 . then then then then then ; g", "1 ."], ["1 if 2 . else 3 . 4", "5 ."],
                 [": f begin 1 if then ;", "1 ."], ["leave", "2 ."], [": f 1 if leave then ;", "3 ."],
                 ["3 0 do i if then then loop", "1 ."], ["1 2 3 4 1 if begin 1 if 1 if 0 while", "depth ."],
                 [": h 2 0 do 2 0 do i . loop loop ; h", "h"], [": f 2 0 do 2 0 do 2 0 do 2 0 do 1 . loop loop loop loop ;", "2 ."]]
TEXT_2V = [  # D2: a variable / value named like a built-in word (its own dictionary entry: KV)
    "variable mod 5 mod ! mod @ .", "variable mod 930884 mod ! 17 3 mod . mod @ .", "variable max 4 max ! max .",
    "7 value abs abs 1+ . -3 abs .", "variable dup 3 dup ! dup @ 2 * . 5 dup . .", "variable sum 0 sum ! 4 sum +! sum @ .",
]
TEXT_2E_TURNS = [["variable a : sq dup * ; 1 .", "3 sq .", ": sq dup dup * * ; 3 sq ."],
                 ["variable z 5 value n", "n . 9 value n n .", ": m 1 ; : n 4 ; n ."], ["1 . : f 2 ;", "3 . : g f f + ; : f 10 ; g ."]]
NEW_2C = set(K.CTLW + K.SCRW + K.SHWC + ["?do"])  # words piece 2c adds


def in_2c(p, tok):
    """A program piece 2c covers: 2b's words plus one level of control (no hoisted helper, no begin ... repeat without
    while), do loops with i inside them, the scratch words, gcd lcm ?dup; no control word as a name, no variable named
    like a 2c word (a definition may shadow the scratch / shared words)."""
    import text_vm as T
    tm, words, nested = T.TextMachine(tok), [], []
    tr, op, ctl = tm.translate, tm.opener, tm.control
    tm.translate = lambda w: (words.append((w.lower(), tm.comment)), tr(w))[1]
    out, owned = [], []
    for t in T.text_ids(p, tok):
        out.append(t)
        for k in tm.feed(t):
            owned.append(len(out))
            out.append(k)
    vis = T.visible(out, owned, tok)
    decl = [(w, words[i + 1][0]) for i, (w, c) in enumerate(words[:-1])
            if not c and w in (":", "variable", "value", "constant")]
    names = [n for _, n in decl]
    bad = {n for w, n in decl if w != ":" and n in K.SHWC}  # (a control word as a name: 'collide'; D2: a variable named like a
    # built-in is its own entry; shortcut: a variable / constant named gcd lcm ?dup is not checked)
    forbid = FORBID_2B - NEW_2C
    return ("error" not in vis and
            not bad and not any(nested) and
            sum(w in ("variable", "value", "constant") and not c for w, c in words) <= NVARS and
            not any(w in forbid and not c for w, c in words))


def in_2d(blocks, tok):
    """A turn (its ```forth blocks, state carried) piece 2d-minimal covers: in_2c's words and structures, its errors only
    of the classes the kernel detects (sp_kernel.KCLS)."""
    import text_vm as T
    tm, words = T.TextMachine(tok), []
    tr, op, ctl = tm.translate, tm.opener, tm.control
    tm.translate = lambda w: (words.append((w.lower(), tm.comment, bool(tm.error))), tr(w))[1]
    out, owned = [], []
    for t in tok("".join(f"```forth\n{b}\n```\nok\n" for b in blocks), add_special_tokens=False).input_ids:
        out.append(t)
        for k in tm.feed(t):
            owned.append(len(out))
            out.append(k)
    vis = tok.decode([out[o] for o in owned], skip_special_tokens=True)
    errs = [ln for ln in vis.splitlines() if ln.startswith("error")]
    live = [(w, c) for w, c, e in words if not e]  # (words after an error are not run)
    decl = [(w, live[i + 1][0]) for i, (w, c) in enumerate(live[:-1]) if not c and w in (":", "variable", "value", "constant")]
    bad = {n for w, n in decl if w != ":" and n in K.SHWC}  # (a control word as a name: 'collide'; D2: a variable named like a
    # built-in is its own entry; shortcut: a variable / constant named gcd lcm ?dup is not checked)
    return (all(e[len("error: "):] in [T.ERRS[c] for c in K.KCLS] for e in errs) and not bad and
            not any(w in FORBID_2B - NEW_2C and not c for w, c in live))


def text_words(ids, tok):
    """text_vm's word split (fence-aware): [[token texts] per closed word] (only words inside fences)."""
    words, fence, after, cur, ws = [], False, False, [], False
    for t in ids:
        s = tok.decode([t])
        if not fence:
            if not (after == "forth" and s[:1] == "\n"):  # (text_vm: a fence opens at the line start after ```forth)
                after, ws = "forth" if after is True and s.strip().lower() == "forth" else s.strip() == "```", False
                continue
            fence, after = True, False
        if s.strip() == "```":
            words += [cur] * bool(cur)
            fence, cur = False, []
            continue
        if s[:1].isspace() or ws:
            words += [cur] * bool(cur)
            cur = []
        if s.strip():
            cur.append(s.strip())
        ws = s[-1:].isspace()
    return words


FRONT_OPS = [V.ADD, V.SUB, V.MUL, V.DIV, V.MOD, V.PR, V.DUP, V.DROP, V.OVER, V.PICK, V.SWAP, V.NEG, V.LT, V.GT]  # 2a's words


def in_2a(p, tok):
    """A program text_vm runs without error using only numbers and single-token built-in words."""
    import re as _re
    import text_vm as T
    ids = T.text_ids(p, tok)
    ok = all(_re.fullmatch(r"-?[0-9]+", "".join(w)) or (len(w) == 1 and w[0].lower() in
             {k.lower() for k, v in V.WORDS.items() if v in FRONT_OPS}) for w in text_words(ids, tok))
    out, owned, _ = T.stream(ids, tok)
    return ok and "error" not in T.visible(out, owned, tok)


FORBID_2B = {"if", "else", "then", "begin", "while", "repeat", "until", "again", "do", "?do", "loop", "+loop", "i", "exit",
             "leave", "gcd", "lcm", "?dup", "rot", "-rot", "min", "max", "/mod", "*/", "emit"}  # later pieces (2c: control flow; hidden shared / scratch words)


def in_2b(p, tok):
    """A program text_vm runs without error that piece 2b covers: numbers, the built-in words of l_f_cls, names (one
    definition each), at most 8 variables (the kernel's cells), comments."""
    import text_vm as T
    tm, words = T.TextMachine(tok), []
    tr = tm.translate
    tm.translate = lambda w: (words.append((w.lower(), tm.comment)), tr(w))[1]
    out, owned = [], []
    for t in T.text_ids(p, tok):
        out.append(t)
        for k in tm.feed(t):
            owned.append(len(out))
            out.append(k)
    vis = T.visible(out, owned, tok)
    names = [words[i + 1][0] for i, (w, c) in enumerate(words[:-1])
             if not c and w in (":", "variable", "value", "constant")]
    return ("error" not in vis and len(names) == len(set(names)) and
            sum(w in ("variable", "value", "constant") and not c for w, c in words) <= NVARS and
            not any(w in FORBID_2B and not c for w, c in words))


def text_cases(tok, extra=None):
    """{name: (ids, owned)}: piece-2a / 2b / 2c programs, multi-fence streams with prose, and the real programs in 2c's
    scope."""
    import json
    import text_vm as T
    st = lambda ids: T.stream(ids, tok)[:2]
    cases = {f"t2a_{i}": st(T.text_ids(p, tok)) for i, p in enumerate(TEXT_2A)}
    for p in TEXT_2A:
        assert in_2a(p, tok), p
    cases |= {f"tstream_{i}": st(tok(s, add_special_tokens=False).input_ids) for i, s in enumerate(TEXT_STREAMS)}
    for p in TEXT_2B:
        assert in_2b(p, tok), p
    cases |= {f"t2b_{i}": st(T.text_ids(p, tok)) for i, p in enumerate(TEXT_2B)}
    cases |= {f"t2bs_{i}": st(tok(s, add_special_tokens=False).input_ids) for i, s in enumerate(TEXT_2B_STREAMS)}
    for p in TEXT_2C:
        assert in_2c(p, tok), p
    cases |= {f"t2c_{i}": st(T.text_ids(p, tok)) for i, p in enumerate(TEXT_2C)}
    cases |= {f"t2cs_{i}": st(tok(s, add_special_tokens=False).input_ids) for i, s in enumerate(TEXT_2C_STREAMS)}
    turn = lambda bs: tok("".join(f"```forth\n{b}\n```\nok\n" for b in bs), add_special_tokens=False).input_ids
    cases |= {f"t2d_{i}": st(turn(bs)) for i, bs in enumerate(TEXT_2D)}
    for bs in TEXT_2F:
        assert in_2d(bs, tok), bs
    cases |= {f"t2f_{i}": st(turn(bs)) for i, bs in enumerate(TEXT_2F)}
    for bs in TEXT_A1:
        assert in_2d(bs, tok), bs
    cases |= {f"ta1_{i}": st(turn(bs)) for i, bs in enumerate(TEXT_A1)}
    for bs in TEXT_EP:
        assert in_2d(bs, tok), bs
    cases |= {f"tep_{i}": st(turn(bs)) for i, bs in enumerate(TEXT_EP)}
    for bs in TEXT_NM:
        assert in_2d(bs, tok), bs
    cases |= {f"tnm_{i}": st(turn(bs)) for i, bs in enumerate(TEXT_NM)}
    cases |= {f"tlong_{i}": st(turn(bs)) for i, bs in enumerate(TEXT_LONG)}
    prose = " the quick brown fox jumps over the lazy dog and then" * 300  # 3300 digit-free text tokens on one line:
    # NHA (the protocol name hash) sums ~54k there, the largest residual a real turn can plausibly build
    cases["tlong_1"] = st(tok("```forth\n: f 3 0 do i . loop ; f\n```\n" + prose + "\n```forth\nf 2 0 do i . loop\n```\n",
                              add_special_tokens=False).input_ids)
    for p in TEXT_2E:
        assert in_2c(p, tok), p
    cases |= {f"t2e_{i}": st(T.text_ids(p, tok)) for i, p in enumerate(TEXT_2E)}
    for i, b in enumerate(TEXT_2H):  # the closing fence as '```' or '```\n' (then no leading '\n' before '```output')
        for close in (1975, 2640):
            body = tok(f"```forth\n{b}\n", add_special_tokens=False).input_ids + [close]
            more = tok("ok then\n```forth\n7 .\n```\ndone", add_special_tokens=False).input_ids
            cases[f"t2h_{i}_{close}"] = st(body + more)
    for p in TEXT_2V:
        assert in_2c(p, tok), p
    cases |= {f"t2v_{i}": st(T.text_ids(p, tok)) for i, p in enumerate(TEXT_2V)}
    for p in TEXT_2N:
        assert in_2c(p, tok), p
    cases |= {f"t2n_{i}": st(T.text_ids(p, tok)) for i, p in enumerate(TEXT_2N)}
    for bs in TEXT_2N_TURNS:
        assert in_2d(bs, tok), bs
    cases |= {f"t2nt_{i}": st(turn(bs)) for i, bs in enumerate(TEXT_2N_TURNS)}
    for p in TEXT_2G:
        assert in_2c(p, tok), p
    cases |= {f"t2g_{i}": st(T.text_ids(p, tok)) for i, p in enumerate(TEXT_2G)}
    for bs in TEXT_2E_TURNS:
        assert in_2d(bs, tok), bs
    cases |= {f"t2et_{i}": st(turn(bs)) for i, bs in enumerate(TEXT_2E_TURNS)}
    if extra:  # programs (in_2c), or turns: lists of blocks (in_2d)
        real = [p for p in json.load(open(extra)) if (in_2d if isinstance(p, list) else in_2c)(p, tok)]
        cases |= {f"real_{i}": st(turn(p) if isinstance(p, list) else T.text_ids(p, tok)) for i, p in enumerate(real)}
    return cases


MIN_MARGIN = 0.5  # kernel-owned positions: top-1 minus top-2 logit (the kernel's own tokens are at 1 and up)


@torch.no_grad()
def text_check(m, ids, owned, mine=0.5):
    """Every position: the kernel's argmax where text_vm says the kernel writes, max logit < mine where the model does
    (incl. after the last token). Prefill (if <= 7000 tokens) and token-by-token cached decode."""
    own = set(owned)
    kern = [p - 1 for p in owned]
    free = [p for p in range(len(ids)) if p + 1 not in own]
    tgt = torch.tensor([ids[p + 1] for p in kern], device="cuda")
    r = {"tokens": len(ids), "kernel_tokens": len(owned)}

    def score(lg, tag):
        """lg: logits [positions, vocab], or the final hidden states [positions, hidden] (lm_head applied in chunks)."""
        if lg.shape[-1] != m.lm_head.weight.shape[0]:
            parts = [m.lm_head(lg[i:i + 2048]).float() for i in range(0, len(lg), 2048)]
            am = torch.cat([p.argmax(-1) for p in parts])
            mx = torch.cat([p.max(-1).values for p in parts])
            t2 = torch.cat([p.topk(2, -1).values for p in parts])
        else:
            am, mx, t2 = lg.argmax(-1), lg.max(-1).values, lg.topk(2, -1).values
        a = am[kern]
        bad = (a != tgt).nonzero().flatten().tolist()
        r[f"{tag}_wrong"] = len(bad)
        if bad:
            r[f"{tag}_first_wrong (pos, want, got)"] = [(kern[i], int(tgt[i]), int(a[i])) for i in bad[:5]]
        loud = [p for p in free if float(mx[p]) >= mine]
        r[f"{tag}_loud_model_positions"] = len(loud)
        if loud:
            r[f"{tag}_first_loud (pos, got)"] = [(p, int(am[p])) for p in loud[:5]]
        if kern:
            margin = t2[kern, 0] - t2[kern, 1]
            r[f"{tag}_min_margin"] = float(margin.min())
            low = (margin < MIN_MARGIN).nonzero().flatten().tolist()  # a tie the argmax hides: a sampler would split it
            r[f"{tag}_low_margin_wrong"] = len(low)
            if low:
                r[f"{tag}_first_low_margin (pos, margin)"] = [(kern[i], float(margin[i])) for i in low[:5]]

    x = torch.tensor([ids], device="cuda")
    if len(ids) <= int(os.environ.get("SP_PREFILL_MAX", 7000)):
        score(m.model(x).last_hidden_state[0], "prefill")
    if len(ids) > int(os.environ.get("SP_CACHED_MAX", 10**9)):  # (long programs: one-shot prefill only)
        return r
    out = m(x[:, :1], use_cache=True)
    lgs = [out.logits[0, -1].float()]
    for t in range(1, len(ids)):
        out = m(x[:, t:t + 1], past_key_values=out.past_key_values, use_cache=True)
        lgs.append(out.logits[0, -1].float())
    score(torch.stack(lgs), "cached")
    return r


def load(out, plan):
    if "build" in CHECKS:
        K.save(K.build(plan), out)
    m = AutoModelForCausalLM.from_pretrained(out, dtype=torch.bfloat16).cuda().eval()
    return m


if __name__ == "__main__":
    out, what = sys.argv[1], sys.argv[2]
    R = {}
    if what in FORTH:
        m = load(out, K.PLAN)
        R["layers"] = m.config.hybrid_override_pattern
        R["residual_dims"] = K._n[0]
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(K.LIGHTNING)
        for name, (runs, want) in forth(what, sys.argv[3:]).items():
            r = check(m, runs, gen=True)
            r["expected_output"] = want
            if "generate" in CHECKS:  # plain generate(): prompt = everything before the last command's kernel run
                ctx = flatten(runs[:-1])[0] + runs[-1][0]
                g = m.generate(torch.tensor([ctx], device="cuda"), max_new_tokens=len(runs[-1][1]) + 8,
                               do_sample=False)[0, len(ctx):].tolist()
                r["plain_generate_last_command"] = {"tokens": len(g), "exact": g[:len(runs[-1][1])] == runs[-1][1],
                                                    "printed": tok.decode(g, skip_special_tokens=True).strip()}
            R[name] = r
            print(name, r, flush=True)
    if what == "text":  # text mode (text_vm.py) vs the compiled front end; optional real programs json
        from transformers import AutoTokenizer
        m = load(out, K.PLAN)
        R["layers"] = m.config.hybrid_override_pattern
        R["residual_dims"] = K._n[0]
        for name, (ids, owned) in text_cases(AutoTokenizer.from_pretrained(K.LIGHTNING),
                                             sys.argv[3] if len(sys.argv) > 3 else None).items():
            if os.environ.get("SP_ONLY") and not name.startswith(tuple(os.environ["SP_ONLY"].split(","))):
                continue  # SP_ONLY=t2d_59,real_: only cases with these name prefixes (a small run first)
            R[name] = text_check(m, ids, owned)
            print(name, R[name], flush=True)
    if what == "handoff":
        from transformers import AutoTokenizer
        m = load(out, K.PLAN)
        R |= handoff(m, AutoTokenizer.from_pretrained(K.LIGHTNING))
    if what == "audit":  # max |dt * A| per Mamba layer over a real program (vLLM chunk-cumsum rule: <= ~173)
        m = load(out, K.PLAN)
        runs = programs("blackjack", ["1"])["blackjack1"]
        x = torch.tensor([flatten(runs)[0]], device="cuda")
        worst = {}
        for i, blk in enumerate(m.model.layers):
            if blk.block_type == K.MA:
                mx = blk.mixer
                hook = lambda mod, inp, o, i=i, mx=mx: worst.__setitem__(i, float(
                    (torch.nn.functional.softplus(o[0, :, -mx.num_heads:].float() + mx.dt_bias.float())
                     * torch.exp(mx.A_log.float())).max()))
                mx.in_proj.register_forward_hook(hook)
        with torch.no_grad():
            m(x)
        R["max_abs_dtA_per_layer"] = worst
        R["max_abs_dtA"] = max(worst.values())
        R["A_log"] = sorted({float(v) for b in m.model.layers if b.block_type == K.MA for v in b.mixer.A_log})
        print(R, flush=True)
    if what in ("all", "some"):  # final suite; one-shot prefill and per-run generate() skipped above 7000 tokens
        m = load(out, K.PLAN)
        R["layers"] = m.config.hybrid_override_pattern
        R["residual_dims"] = K._n[0]
        groups = sys.argv[3:] if what == "some" else list(ALL) + ["overflow", "overflow20"]
        progs = {k: v for g in groups for k, v in programs(g, ALL.get(g, [])).items()}
        for name, runs in progs.items():
            n = len(flatten(runs)[0])
            os.environ["SP_SKIP"] = "prefill" if n > 7000 else ""
            r = check(m, runs, gen=n <= 7000 and (not name.startswith("block") or name.endswith("0")))
            r["kernel_runs"] = sum(1 for _, b in runs if b)
            R[name] = r
            print(name, r, flush=True)
    if what in ("arith", "mulmod", "words"):
        m = load(out, K.STAGE1)
        for name, runs in programs(what, sys.argv[3:]).items():
            R[name] = check(m, runs, gen=name.endswith("0"))
            print(name, R[name], flush=True)
    if what == "block":
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 2
        depth = int(sys.argv[4]) if len(sys.argv) > 4 else 40
        m = load(out, K.STAGE1)
        R["layers"] = m.config.hybrid_override_pattern
        R["residual_dims"] = K._n[0]
        for s in range(n):
            runs, vm = block_stream(random.Random(s), depth, 60 if depth <= 40 else 3 * depth)
            r = check(m, runs, gen=s == 0)
            r["peak_depth"] = vm.peak["data"]
            R[f"stream{s}"] = r
            print(s, r, flush=True)
    R["checks"] = CHECKS
    import transformers
    R["transformers"] = transformers.__version__
    tag = "_".join([what] + (sys.argv[3:] if what == "some" else []))
    json.dump(R, open(f"{out}/sp_{tag}_{transformers.__version__}.json", "w"), indent=1)
