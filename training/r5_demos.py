"""Round-5 demonstrations (SFT, no RL). s2a_r3 fails its weakest cells (recurrence, is_prime, nearest_integer_root,
lcg_trace, euclid_trace) by computing by hand or by fighting the stack (`to` on a variable, a missing dup, x without @),
then running out of thinking. A demonstration row shows the habit: a short plan in the thinking (no long hand
arithmetic), ONE compact working program, the machine's output, the answer read off it.
Problems are fresh draws from the generators (curriculum.L6 / L6_SMALL / L6_EASY, demo_gen.dmbig; numbers.is_prime:
DeepMind's phrasings and sizes, drawn here), never a problem of any data/loop set. Each problem's expected answer comes
from the generator (its own reference program on the machine, or Python); the demo program is a different program, so
an agreement is a real check. Every row: graded correct by loop_sim.same, composite-exact under THIS tree's text_vm
(distill_rows.row), clean (distill_rows.clean) under the row prompt (card v4).
  LOOP_TOKENIZER=<Lightning tokenizer> python training/r5_demos.py --prompt training/data/cards/reference_v4.txt --out DIR \
      [--n recurrence/hard=200 ...] [--show 10]
Programs per family (the step limit is 196608 kernel steps a block: odd trial division reaches n ~ 3e6, so numbers.is_prime
<= 10 digits uses Miller-Rabin (products of two 10-digit numbers fit 20 digits); 11-12 digits only when a factor < 1500
settles it: mod by an 11-digit n is past the machine's limit)."""
import argparse, collections, glob, hashlib, json, math, os, random, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training")]
from forth.curriculum import L6, L6_SMALL, L6_EASY  # noqa: E402
from demo_gen import dmbig  # noqa: E402
import distill_rows as D  # noqa: E402
import loop_sim  # noqa: E402

F = "`" * 3
pick = lambda R, xs: R.choice(xs)
L6_ARGS = {"hard": lambda f: L6[f][0], "small": lambda f: L6_SMALL[f], "variant": lambda f: L6_EASY[f]}
DEFAULT_N = {"L6.recurrence/hard": 220, "L6.recurrence/small": 40, "L6.lcg_trace/hard": 110, "L6.lcg_trace/small": 50,
             "L6.euclid_trace/hard": 70, "L6.euclid_trace/small": 40, "L6.euclid_trace/variant": 40,
             "DMbig.is_prime/hard": 170, "DMbig.is_prime/variant": 40, "numbers.is_prime/easy": 220,
             "DMbig.nearest_integer_root/hard": 200, "DMbig.nearest_integer_root/variant": 30}


def run(progs):
    """-> [(prog, its output)] for the blocks of one turn (state carries)"""
    return [(p, loop_sim.kernel_output(progs[:i + 1], TOK)) for i, p in enumerate(progs)]


def one(think, prog, fin):
    """a one-block demo -> (think, [(lead, prog, out)], words, boxed)"""
    out = loop_sim.kernel_output([prog], TOK)
    return (think, [("", prog, out)], *fin(out)) if not out.startswith(("error", "(")) else (think, [("", prog, out)], "", "")


def recurrence(R, a0, a1, k, m, n):
    a2 = (a1 + k * a0) % m
    check = R.random() < 0.5
    t = [pick(R, [f"We need a_{n} where a_0 = {a0}, a_1 = {a1} and a_(n+2) = (a_(n+1) + {k} * a_n) mod {m}.",
                  f"The sequence starts {a0}, {a1}; each next term is (previous + {k} times the one before it) mod {m}, and we want term {n}.",
                  f"We want term {n} of the sequence {a0}, {a1}, ... with next = (previous + {k} * the one before) mod {m}."]),
         pick(R, [f"That is {n - 1} steps of modular arithmetic, so a short loop on the machine is the reliable way.",
                  f"Doing {n - 1} steps by hand would be long and error-prone; a loop does them exactly.",
                  f"{n - 1} steps with numbers up to {m}: a job for a loop, not for hand arithmetic."]),
         pick(R, ["Keep the last two terms in variables a and b (a = a_j, b = a_(j+1)).",
                  "Two variables hold the last two terms: a = a_j and b = a_(j+1)."]),
         f"One step: new = (b + {k}*a) mod {m}, then a gets the old b and b gets new. On the stack: b @ dup a @ {k} * + "
         f"{m} mod b ! a ! (the dup keeps the old b, which goes into a).",
         f"After j steps a holds a_j, so {n} steps leave a_{n} in a."]
    if check:
        t.append(f"To check the indexing I'll print a_2 first: ({a1} + {k}*{a0}) mod {m} = {a2}.")
        prog = (f"variable a variable b\n: term ( j -- a_j ) {a0} a ! {a1} b ! 0 ?do b @ dup a @ {k} * + {m} mod b ! a ! loop a @ ;\n"
                f"2 term . {n} term .")
        fin = lambda o: (f"The first value is a_2 = {o.split()[0]}, as computed by hand, so the indexing is right and "
                         f"a_{n} = {o.split()[-1]}.", o.split()[-1])
    else:
        prog = f"variable a variable b\n{a0} a ! {a1} b !\n{n} 0 do b @ dup a @ {k} * + {m} mod b ! a ! loop\na @ ."
        fin = lambda o: (f"After {n} steps a holds a_{n} = {o.split()[-1]}.", o.split()[-1])
    return one(t, prog, fin)


def lcg_trace(R, s, a, c, m, n):
    t = [pick(R, [f"We need x_1 through x_{n} for x -> ({a}x + {c}) mod {m}, starting from x_0 = {s}.",
                  f"Starting from {s}, apply x -> ({a}x + {c}) mod {m} {n} times and list every value."]),
         pick(R, [f"Each step multiplies a number below {m} by {a} and reduces mod {m}; {n} of those by hand is slow and risky, so the machine prints them all.",
                  f"{n} multiplications and reductions mod {m}: the machine does them exactly in one loop."]),
         f"Keep x in a variable. Each pass: x @ {a} * {c} + {m} mod, then dup . prints the new value and x ! stores it back.",
         pick(R, ["The seed itself is not printed, so the output is exactly the list asked for.",
                  "Only the new values are printed, not the starting value."])]
    if R.random() < 0.5:
        prog = f"variable x\n{s} x !\n{n} 0 do x @ {a} * {c} + {m} mod dup . x ! loop"
    else:
        prog = f"variable x\n{s} x !\n: step ( -- x' ) x @ {a} * {c} + {m} mod dup x ! ;\n{n} 0 do step . loop"
    sep = pick(R, [" ", ", "])
    return one(t, prog, lambda o: (f"The machine printed the {n} values in order.", sep.join(o.split())))


def euclid_trace(R, a, b):
    t = [pick(R, [f"Euclid's algorithm on ({a}, {b}): replace (a, b) by (b, a mod b) until b is 0.",
                  f"Run Euclid on {a} and {b}: (a, b) -> (b, a mod b) until the remainder is 0."]),
         f"The trace starts with {b}, then every nonzero remainder, then the gcd (the last nonzero remainder, so it appears twice).",
         pick(R, ["Long divisions of 10-digit numbers by hand are where mistakes happen, so let the machine print the trace.",
                  "Each step is a long division of big numbers: the machine prints the whole trace exactly."]),
         "Word steps ( a b -- ): while b is nonzero, print b (dup .), then swap over mod turns a b into b (a mod b). "
         "When b is 0, drop it and print a, the gcd."]
    prog = f": steps ( a b -- ) begin dup while dup . swap over mod repeat drop . ;\n{a} {b} steps"
    return one(t, prog, lambda o: (f"The gcd is {o.split()[-1]}.", " ".join(o.split())))


def dm_prime(R, n, word):
    r = math.isqrt(n)
    t = [pick(R, [f"Is {n} {word}? Its square root is about {r}, so trial division by 2 and the odd numbers up to {r} decides it.",
                  f"{n} has {len(str(n))} digits and sqrt({n}) is about {r}: trial division up to {r} settles whether it is prime."]),
         pick(R, [f"That is about {r // 2} divisions: easy for the machine, far too many by hand.",
                  f"Around {r // 2} divisions is too many to do by hand, but one loop on the machine does them."]),
         "Word prime? ( n -- f ): an even n gives 0; then d = 3, 5, 7, ... while d*d <= n, and a d that divides n gives 0; "
         "if none does, -1 (prime)."]
    prog = (": prime? ( n -- f ) dup 2 mod 0= if drop 0 exit then 3 begin 2dup dup * >= while 2dup mod 0= if 2drop 0 exit "
            f"then 2 + repeat 2drop -1 ;\n{n} prime? .")
    return one(t, prog, lambda o: prime_answer(n, word, o.split()[-1] == "-1"))


def prime_answer(n, word, prime):
    yes = prime == (word == "prime")
    return (f"The flag is {'-1' if prime else '0'}: {n} is {'prime' if prime else 'not prime'}, so the answer is "
            f"{'yes' if yes else 'no'}.", "\\text{" + ("Yes" if yes else "No") + "}")


def mr_prime(R, n, word):
    """two blocks: a block runs at most 196608 kernel steps, and one base costs ~45k at 10 digits"""
    t = [pick(R, [f"Is {n} {word}? It has {len(str(n))} digits, so trial division up to sqrt(n) ~ {math.isqrt(n)} is more than one block can run.",
                  f"{n} is a {len(str(n))}-digit number: trial division would need about {math.isqrt(n) // 2} divisions, too many for one block."]),
         "Miller-Rabin with bases 2, 3, 5, 7, 11 is exact for n below 2*10^12, so use it.",
         "Write n - 1 = d * 2^s with d odd. Base a passes when a^d = 1 or a^(d*2^r) = n - 1 (mod n) for some r < s; n is prime "
         "exactly when all five bases pass.",
         pick(R, ["Every product of two numbers below n fits in 20 digits (n has at most 10), so mulmod is just * n @ mod.",
                  "With n at most 10 digits, a*b for a, b < n fits in 20 digits: mm = * n @ mod."]),
         "pow keeps the result, base and exponent in variables: multiply the result by the base when the exponent is odd, "
         "square the base, halve the exponent.",
         pick(R, ["Each base costs a few dozen big multiplications, so bases 2, 3, 5 in the first block and 7, 11 in a second one.",
                  "Five modular powers of a 10-digit number are a lot of steps for one block: 2, 3, 5 first, then 7 and 11."])]
    p1 = ("variable n variable d variable s variable r variable b variable e\n: mm ( a b -- c ) * n @ mod ;\n"
          ": pow ( b e -- x ) e ! b ! 1 r ! begin e @ while e @ 2 mod if r @ b @ mm r ! then b @ dup mm b ! e @ 2/ e ! repeat r @ ;\n"
          ": sprp ( a -- f ) d @ pow dup 1 = if drop -1 exit then "
          "s @ begin ?dup while over n @ 1- = if 2drop -1 exit then swap dup mm swap 1- repeat drop 0 ;\n"
          f"{n} n ! n @ 1- d ! 0 s ! begin d @ 2 mod 0= while d @ 2/ d ! s @ 1+ s ! repeat\n2 sprp . 3 sprp . 5 sprp .")
    p2 = "7 sprp . 11 sprp ."
    (_, o1), = run([p1])
    segs = [("", p1, o1)]
    if o1.split() == ["-1"] * 3:
        o2 = run([p1, p2])[1][1]
        segs.append((pick(R, ["Bases 2, 3 and 5 pass. Now 7 and 11:", "All three pass; the last two bases:"]), p2, o2))
    prime = all(x == "-1" for _, _, o in segs for x in o.split()) and len(segs) == 2
    yes = prime == (word == "prime")
    words = (f"All five bases pass, so {n} is prime" if prime else
             f"A base fails (a 0), so {n} is composite") + f": the answer is {'yes' if yes else 'no'}."
    return t, segs, words, "\\text{" + ("Yes" if yes else "No") + "}"


def factor_prime(R, n, word):
    t = [f"Is {n} {word}? It has {len(str(n))} digits: mod by an 11-digit n is past the machine's limit, so Miller-Rabin can't run, "
         "and trial division up to sqrt(n) is far too long.",
         pick(R, ["But a small factor would settle it: try 2 and the odd numbers below 1500 in one block.",
                  "A small divisor would settle it at once, so search 2 and the odd d below 1500 first."]),
         "Word factor ( n -- d ) leaves the first divisor found, or 0 if there is none below 1500."]
    prog = (": factor ( n -- d ) dup 2 mod 0= if drop 2 exit then 1500 3 do dup i mod 0= if drop i exit then 2 +loop drop 0 ;\n"
            f"{n} factor .")

    def fin(o):
        f = int(o.split()[-1])
        assert f > 1 and n % f == 0
        yes = word == "composite"
        return (f"{n} = {f} * {n // f}, so it is composite: the answer is {'yes' if yes else 'no'}.",
                "\\text{" + ("Yes" if yes else "No") + "}")
    return one(t, prog, fin)


def root(R, n, k):
    name, H = {2: ("square", 4000000000), 3: ("cube", 1000000)}[k]
    pw = "dup *" if k == 2 else "dup dup * swap *"  # m^2 must not be the top operand of *
    t = [pick(R, [f"We need the {name} root of {n} ({len(str(n))} digits) to the nearest integer.",
                  f"The {name} root of {n}, rounded to the nearest integer."]),
         pick(R, ["Getting every digit right by hand is slow; a binary search for the floor root on the machine is exact.",
                  "Estimating by hand gets close but the last digits are hard to pin down; binary search on the machine settles them."]),
         f"Search r with r^{k} <= n < (r+1)^{k}: lo = 0, hi = {H}; while lo < hi, mid = (lo + hi + 1) / 2, and mid^{k} > n gives "
         f"hi = mid - 1, else lo = mid. {H}^{k} still fits in 20 digits.",
         f"Then round: r + 1 if (r+1)^{k} - n < n - r^{k}, else r."]
    prog = (f"variable n variable lo variable hi\n: pw ( m -- m^{k} ) {pw} ;\n"
            f": root ( -- r ) 0 lo ! {H} hi ! begin lo @ hi @ < while lo @ hi @ + 1+ 2/ dup pw n @ > if 1- hi ! else lo ! then repeat lo @ ;\n"
            ": nearest ( -- r ) root dup 1+ pw n @ - n @ 2 pick pw - < if 1+ then ;\n"
            f"{n} n ! root . nearest .")
    return one(t, prog, lambda o: (f"The floor root is {o.split()[0]} and the nearest integer is {o.split()[-1]}.", o.split()[-1]))


# ---- round 6: the program shapes s2a_r5 reaches for, done right, and recovery from its real first-attempt bugs.
# A recovery demo returns (think, segs, words, boxed, masked): masked = indices of segs whose block ran without an error
# but is wrong (its tokens are loss-masked like an error block's); the fix re-sends only what changed, never re-declares.
def recurrence_stack(R, a0, a1, k, m, n):
    t = [pick(R, [f"We need a_{n} where a_0 = {a0}, a_1 = {a1} and a_(n+2) = (a_(n+1) + {k} * a_n) mod {m}.",
                  f"The sequence starts {a0}, {a1}; each next term is (previous + {k} times the one before it) mod {m}; we want term {n}."]),
         f"{n - 1} steps of modular arithmetic: a loop does them exactly.",
         f"Keep the pair on the stack: ( a b ) becomes ( b, (b + {k}*a) mod {m} ). swap gives ( b a ), {k} * gives ( b {k}a ), "
         f"over + gives ( b {k}a+b ), then {m} mod. The stack stays two deep.",
         f"After j steps the stack is a_j a_(j+1), so after {n} steps drop the top and print a_{n}."]
    prog = f"{a0} {a1}\n{n} 0 do swap {k} * over + {m} mod loop\ndrop ."
    return one(t, prog, lambda o: (f"The output is {o.split()[-1]}: that is a_{n}.", o.split()[-1]))


def lcg_stack(R, s, a, c, m, n):
    t = [pick(R, [f"We need x_1 through x_{n} for x -> ({a}x + {c}) mod {m}, starting from x_0 = {s}.",
                  f"Starting from {s}, apply x -> ({a}x + {c}) mod {m} {n} times and list every value."]),
         f"{n} multiply-and-reduce steps: the machine prints them all.",
         f"Keep x on the stack: {a} * {c} + {m} mod turns x into the next value; dup . prints a copy and leaves x for the next "
         "pass. The seed is not printed, and the last value is dropped at the end."]
    prog = f"{s}\n{n} 0 do {a} * {c} + {m} mod dup . loop drop"
    return one(t, prog, lambda o: (f"The output is {len(o.split())} values below {m}: x_1 through x_{n}.", pick(R, [" ", ", "]).join(o.split())))


def recurrence_fix(R, a0, a1, k, m, n):
    t = [pick(R, [f"We need a_{n} where a_0 = {a0}, a_1 = {a1} and a_(n+2) = (a_(n+1) + {k} * a_n) mod {m}.",
                  f"The sequence starts {a0}, {a1}; each next term is (previous + {k} times the one before it) mod {m}; we want term {n}."]),
         f"{n - 1} steps by hand is too long; use two variables a = a_j, b = a_(j+1): new = (b + {k}*a) mod {m}, then a gets "
         "the old b and b gets new."]
    bad = f"variable a variable b\n{a0} a ! {a1} b !\n{n} 0 do b @ a @ {k} * + {m} mod b ! a ! loop\na @ ."
    fix = f"{a0} a ! {a1} b !\n{n} 0 do b @ dup a @ {k} * + {m} mod b ! a ! loop\na @ ."
    (_, o1), (_, o2) = run([bad, fix])
    lead = (pick(R, ["Stack underflow: b ! used the only value, so a ! had nothing left; the old b was never kept.",
                     "Underflow: after b ! the stack is empty, so a ! has nothing to store. The old b has to be kept with dup."])
            + f" b @ dup keeps it. The failed run already overwrote b, so set both again first; a and b are still declared.")
    return t, [("", bad, o1), (lead, fix, o2)], f"The output is {o2.split()[-1]}: after {n} steps a holds a_{n}.", o2.split()[-1], ()


def lcg_fix(R, s, a, c, m, n):
    t = [pick(R, [f"We need x_1 through x_{n} for x -> ({a}x + {c}) mod {m}, starting from x_0 = {s}.",
                  f"Starting from {s}, apply x -> ({a}x + {c}) mod {m} {n} times and list every value."]),
         "Keep x in a variable and print each new value in a loop."]
    bad = f"variable x\n{s} x !\n{n} 0 do x @ {a} * {c} + {m} mod . loop"
    fix = f"{n} 0 do x @ {a} * {c} + {m} mod dup . x ! loop"
    (_, o1), (_, o2) = run([bad, fix])
    lead = (f"All {n} values are the same: x is never updated, so every pass starts from {s} again. Keep a copy and store "
            f"it back: dup . x !. x still holds {s}, so just run the loop again.")
    return (t, [("", bad, o1), (lead, fix, o2)], f"The output is now {n} different values below {m}: x_1 through x_{n}.",
            pick(R, [" ", ", "]).join(o2.split()), (0,))


ODD = (": prime? ( n -- f ) dup 2 mod 0= if drop 0 exit then 3 begin 2dup dup * >= while 2dup mod 0= if 2drop 0 exit "
       "then 2 + repeat 2drop -1 ;")


def prime_fix(R, n, word):
    r = math.isqrt(n)
    t = [f"Is {n} {word}? Its square root is about {r}, so trial division up to {r} decides it: a loop on the machine.",
         "Word prime? ( n -- f ): 0 if some d with d*d <= n divides n, else -1."]
    bad = (": prime? ( n -- f ) 2 begin 2dup dup * >= while 2dup mod 0= if 2drop 0 exit then 1+ repeat 2drop -1 ;\n"
           f"{n} prime? .")
    (_, o1), = run([bad])
    if o1.startswith("error: the block ran too long"):
        lead = (f"Too many steps: trying every d up to {r} is about {r} divisions. An even d can't divide an odd n, so test 2 "
                "once and then only odd d: half the work. Re-send prime? (the new definition replaces the old one) and the call.")
        segs, masked = [("", bad, o1), (lead, f"{ODD}\n{n} prime? .", None)], ()
    else:  # a composite: the loop stops early, so show the other common slip: the call without .
        bad = f"{ODD}\n{n} prime?"
        lead = "Nothing printed: the flag is still on the stack. prime? is defined, so just call it again with ."
        segs, masked = [("", bad, None), (lead, f"{n} prime? .", None)], (0,)
    outs = run([x[1] for x in segs])
    segs = [(l, p, o) for (l, p, _), (_, o) in zip(segs, outs)]
    flag = outs[-1][1].split()[-1] == "-1"
    yes = flag == (word == "prime")
    return (t, segs, f"The output is {outs[-1][1].split()[-1]}: {n} is {'prime' if flag else 'not prime'}, so the answer is "
            f"{'yes' if yes else 'no'}.", "\\text{" + ("Yes" if yes else "No") + "}", masked)


def root_fix(R, n, k):
    t = [f"We need the cube root of {n} ({len(str(n))} digits) to the nearest integer: binary search for the floor root r "
         "with r^3 <= n < (r+1)^3 on the machine, lo = 0, hi = 1000000, then round to r or r + 1, whichever cube is nearer."]
    bad = ("variable n variable lo variable hi\n: pw ( m -- m^3 ) dup dup * * ;\n"
           ": root ( -- r ) 0 lo ! 1000000 hi ! begin lo @ hi @ < while lo @ hi @ + 1+ 2/ dup pw n @ > if 1- hi ! else lo ! then repeat lo @ ;\n"
           ": nearest ( -- r ) root dup 1+ pw n @ - n @ 2 pick pw - < if 1+ then ;\n"
           f"{n} n ! root . nearest .")
    fix = f": pw ( m -- m^3 ) dup dup * swap * ;\nroot . nearest ."
    (_, o1), (_, o2) = run([bad, fix])
    lead = ("dup dup * * leaves m*m as the top operand of the second *, and m^2 can have 12 digits; the top operand may have "
            "at most 10. Swap so m is on top: dup dup * swap *. Re-send only pw: root and nearest use the new definition, "
            "and n is already set.")
    return (t, [("", bad, o1), (lead, fix, o2)], f"The output is {o2}: floor root {o2.split()[0]}, nearest integer "
            f"{o2.split()[-1]}.", o2.split()[-1], ())


STYLED = {("L6.recurrence", "stack"): recurrence_stack, ("L6.lcg_trace", "stack"): lcg_stack,
          ("L6.recurrence", "fix"): recurrence_fix, ("L6.lcg_trace", "fix"): lcg_fix,
          ("DMbig.is_prime", "fix"): prime_fix, ("DMbig.nearest_integer_root", "fix"): root_fix}
R6_N = {"L6.recurrence/hard+stack": 130, "L6.recurrence/small+stack": 30, "L6.lcg_trace/hard+stack": 90,
        "L6.lcg_trace/small+stack": 30, "DMbig.is_prime/hard": 40, "DMbig.nearest_integer_root/hard": 30,
        "L6.recurrence/hard+fix": 60, "L6.lcg_trace/hard+fix": 50, "DMbig.is_prime/hard+fix": 50,
        "DMbig.nearest_integer_root/hard+fix": 40}  # recovery 200 of 550 (36%): most rows still work the first time


INTRO = ["Let me run it.", "Here is the program.", "I'll compute it on the machine.", "Running it:", ""]


def assemble(R, think, segs, words, boxed):
    """-> completion text: plan, blocks with their outputs, answer; the blocks in the answer (2/3) or in the thinking (1/3)"""
    blocks = "".join(f"{lead + chr(10) * 2 if lead else ''}{F}forth\n{prog}\n{F}\n{F}output\n{out}\n{F}\n" for lead, prog, out in segs)
    plan = " ".join(think)
    if R.random() < 2 / 3:
        intro = pick(R, INTRO)
        return f"{plan}</think>{intro + chr(10) * 2 if intro else ''}{blocks}{words}\n\n\\boxed{{{boxed}}}"
    return f"{plan}\n\n{blocks}{words}</think>{words}\n\n\\boxed{{{boxed}}}"


def problems(R, cell, k, seen):
    """k fresh problems of a cell: dicts q, a (the generator's answer), family, kind, demo (think, prog, fin)"""
    fam, kind = cell.split("/")
    kind, _, style = kind.partition("+")
    out = []
    while len(out) < k:
        if fam.startswith("L6."):
            f = fam[3:]
            a = L6_ARGS[kind](f)(R)
            q = R.choice(L6[f][2]).format(*a)
            want = " ".join(loop_sim.kernel_output([L6[f][1](*a)], TOK).split())  # the generator's reference program
            demo = (STYLED[(fam, style)] if style else
                    {"recurrence": recurrence, "lcg_trace": lcg_trace, "euclid_trace": euclid_trace}[f])(R, *a)
            ps = [{"q": q, "a": want, "family": fam, "kind": kind, "demo": demo}]
        elif fam.startswith("DMbig."):
            ps = []
            for p in dmbig(R, 10, easy=kind == "variant"):
                if p["family"] != fam:
                    continue
                n = int(re.search(r"\d+", p["q"]).group())
                if style == "fix" and fam == "DMbig.nearest_integer_root" and "cube" not in p["q"]:
                    continue  # the root recovery demo is the cube-power bug
                demo = STYLED[(fam, style)](R, n, p["q"].split()[3] if fam == "DMbig.is_prime" else 3) if style else (dm_prime(R, n, p["q"].split()[3]) if fam == "DMbig.is_prime"
                        else root(R, n, 2 if "square" in p["q"] else 3))
                ps.append(p | {"kind": kind, "demo": demo})
        else:  # numbers.is_prime: DeepMind's phrasings / sizes (mass_pool: 7-12 digits, about half prime)
            d = R.choices([8, 9, 10, 11, 12], [10, 64, 138, 110, 30])[0]
            if R.random() < 0.5:
                n = nextprime(R.randrange(10 ** (d - 1), 10 ** d))
            else:  # smallest factor size as in mass_pool's composites
                fd = R.choices([1, 2, 3, 4, 5], [40, 45, 26, 36, 16])[0]
                p_ = R.choice([3, 7]) if fd == 1 else nextprime(R.randrange(10 ** (fd - 1), 10 ** fd))
                n = p_ * nextprime(R.randrange(10 ** (d - 1) // p_ + p_, 10 ** d // p_))
            if not 10 ** (d - 1) <= n < 10 ** d:
                continue
            word = R.choice(["prime", "composite"])
            q = R.choice(["Is {} {}?", "Is {} a {} number?"]).format(n, word)
            small = next((f for f in [2] + list(range(3, 1500, 2)) if n % f == 0), 0)
            if d > 10 and not small:
                continue  # no exact program for an 11-12 digit n without a small factor
            demo = (mr_prime if d <= 10 else factor_prime)(R, n, word)
            ps = [{"q": q, "a": str(isprime(n) == (word == "prime")), "family": fam, "kind": kind, "demo": demo}]
        for p in ps:
            if p["q"] not in seen and len(out) < k:
                seen.add(p["q"])
                out.append(p)
    return out


def check_mask(ids, mask, plen, segs):
    """a recovery row: the buggy block (an error or a masked one) and every output block are untrained; the diagnosis,
    the fix and the answer are trained"""
    runs, cur, k = {0: [], 1: []}, [], None
    for t, m in zip(ids[plen:], mask[plen:]):
        if m != k and cur:
            runs[k].append(TOK.decode(cur)); cur = []
        k = m; cur.append(t)
    runs[k].append(TOK.decode(cur))
    off, on = "".join(runs[0]), "".join(runs[1])
    for i, (lead, prog, out) in enumerate(segs):
        assert f"{F}output\n{out}" in off and f"{F}output" not in on, ("output trained", out[:40])
        if i < len(segs) - 1:
            assert prog in off and prog not in on, ("bug trained", prog[:60])
        else:
            assert prog in on and (not lead or lead in on), ("fix untrained", prog[:60])
    assert all(r.lstrip("`").startswith(("output", "forth", "\n")) or r.startswith(F) for r in runs[0]), runs[0]


def isprime(n):  # deterministic Miller-Rabin (n < 3.3e24), independent of the Forth program
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d, s = d // 2, s + 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def nextprime(n):
    n += 1
    while not isprime(n):
        n += 1
    return n


def main():
    global TOK
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--prompt", required=True, help="the row prompt (card v4)")
    a.add_argument("--n", action="append", default=[], help="FAMILY/KIND[+STYLE]=K (default: DEFAULT_N, or R6_N with --set r6)")
    a.add_argument("--set", default="r5", choices=["r5", "r6"])
    a.add_argument("--seed", default="r5-demos-2026-11-06")
    a.add_argument("--val-mod", type=int, default=25)
    a.add_argument("--show", type=int, default=0, help="print N rows decoded (what the model is trained on)")
    a.add_argument("--out", required=True)
    args = a.parse_args()
    from transformers import AutoTokenizer
    TOK = AutoTokenizer.from_pretrained(os.environ.get("LOOP_TOKENIZER", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"))
    system = open(args.prompt).read()
    card = D.grams(json.load(open(ROOT / "training/data/cards/rubric_v24.json"))["system"]) - D.grams(system)
    seen = set()  # every problem of every data/loop set (held-out and earlier training sets alike)
    for f in glob.glob(str(ROOT / "training/data/**/*.json"), recursive=True):
        d = json.load(open(f))
        if isinstance(d, list) and d and isinstance(d[0], dict) and "q" in d[0]:
            seen |= {p["q"] for p in d}
    n_held = len(seen)
    want = (R6_N if args.set == "r6" else DEFAULT_N) | dict(x.split("=") for x in args.n)
    R = random.Random(args.seed)
    rows, drop = {"train": [], "val": []}, collections.Counter()
    for cell, k in want.items():
        for p in problems(R, cell, int(k), seen):
            think, segs, words, boxed, *masked = p["demo"]
            masked = masked[0] if masked else ()
            # the last block works; an earlier one is the demo's bug: an error, or a masked block that ran
            last = len(segs) - 1
            bad = [o for i, (_, _, o) in enumerate(segs)
                   if (o.startswith(("error", "(")) if i == last else not (o.startswith("error") or i in masked))]
            if bad:
                drop[(cell, "program error")] += 1
                print("PROGRAM ERROR", cell, p["q"], bad[0][:120], flush=True)
                continue
            text = assemble(R, think, segs, words, boxed)
            final = text.rsplit("</think>", 1)[-1]
            got = loop_sim.last_boxed(final)
            r = {"q": p["q"], "a": p["a"], "text": text, "got": got, "ok": loop_sim.same(got, p["a"], p),
                 "blocks": [{"program": prog, "output": out} for _, prog, out in segs]}
            if not r["ok"]:
                drop[(cell, "wrong")] += 1
                print("WRONG", cell, p["q"], "want", p["a"], "got", got, "out", out[:80], flush=True)
                continue
            if not D.clean(r, card, system):
                drop[(cell, "unclean")] += 1
                continue
            spans = [(text.index(blk := f"{F}forth\n{segs[i][1]}\n{F}\n"), text.index(blk) + len(blk)) for i in masked]
            try:
                prompt = D.prompt_text({"system": system}, p["q"], TOK)
                ids, mask = D.row(prompt, text, TOK, spans)
                if len(segs) > 1:
                    check_mask(ids, mask, len(TOK(prompt, add_special_tokens=False).input_ids), segs)
            except ValueError:
                drop[(cell, "stream differs")] += 1
                continue
            except AssertionError as e:
                drop[(cell, "mask check")] += 1
                print("MASK CHECK", cell, p["q"], e, flush=True)
                continue
            split = "val" if int(hashlib.sha1(p["q"].encode()).hexdigest(), 16) % args.val_mod == 0 else "train"
            rows[split].append({"input_ids": ids, "train_mask": mask, "q": p["q"], "family": p["family"], "kind": p["kind"],
                                "answer": p["a"], "machine": D.accepted(r) or segs[-1][2].split()[-1] in ("-1", "0"),
                                "reruns": 0, "stage": "demo" if args.set == "r5" else "demo6",
                                "recovery": len(segs) > 1 and args.set == "r6"})
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for s, rs in rows.items():
        with open(f"{args.out}/{s}.jsonl", "w") as f:
            f.write("".join(json.dumps(r) + "\n" for r in rs))
        print(f"{s}: {len(rs)} rows, trained tokens {sum(map(sum, (r['train_mask'] for r in rs)))}, total "
              f"{sum(len(r['input_ids']) for r in rs)}, {dict(sorted(collections.Counter(r['family'] + '/' + r['kind'] for r in rs).items()))}")
    print(f"excluded {n_held} existing problems; dropped {dict(drop)}")
    for r in random.Random(1).sample(rows["train"], min(args.show, len(rows["train"]))):
        k = r["train_mask"].index(1)
        print(f"===== {r['family']}/{r['kind']} answer {r['answer']} | {len(r['input_ids'])} tokens\nQ: {r['q']}")
        out, cur, seg = [], 1, []  # untrained spans (kernel output, buggy blocks) in <<< >>>
        for t, m in zip(r["input_ids"][k:], r["train_mask"][k:]):
            if m != cur and seg:
                out.append(TOK.decode(seg) if cur else "<<<" + TOK.decode(seg) + ">>>"); seg = []
            cur = m; seg.append(t)
        out.append(TOK.decode(seg) if cur else "<<<" + TOK.decode(seg) + ">>>")
        print("".join(out))


assert [isprime(n) for n in (2, 9, 561, 2147483647, 3215031751)] == [True, False, False, True, False]

if __name__ == "__main__":
    main()
