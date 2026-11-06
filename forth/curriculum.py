"""Curriculum for the stored-program kernel: seeded generators of (question, reference program, exact answer).

L0 mechanics: one op on small / medium numbers.   L1 big arithmetic: one op at 10-20 digits (kernel limits).
L2 expressions: 2-5 ops, precedence, negatives, squares (DUP *), reuse (OVER), SWAP / NEGATE when the kernel has them.
L3 algorithms: definitions, IF, BEGIN WHILE REPEAT, VARIABLE (sums, factorial, gcd, primes, powers, fibonacci, ...).
L4 word problems: stories with heavy numbers (MATH level 2-3 shapes). L5 programs: recursion and printed sequences.
L6 needs-the-tool: long exact work (search, counting, simulation, big compound arithmetic, full traces), eval.
row(level, seed, words, tok): one verified row: ref.py (check_programs) AND the kernel's reference machine (forth/sp)
print exactly the answer. `words` = the words the kernel has; generators only write those (SWAP NEGATE / are new).
Hanoi / blackjack never appear (check_programs.BANNED, which also rejects the number 21).
"""

import random

from . import sp
from .check_programs import problems as ref_problems
from .ref import Forth

NEW = {"SWAP", "NEGATE", "/"}

ASK = ["What is {e}?", "Compute {e}.", "Calculate {e}.", "Evaluate {e}.", "Find the value of {e}.", "{e} = ?",
       "Quick question: what's {e}?", "Work out {e} exactly.", "Can you tell me what {e} equals?",
       "Give me the exact value of {e}.", "Please compute {e} for me.", "Solve: {e}", "I need {e}.", "How much is {e}?",
       "Determine {e}.", "what's {e}", "{e}?", "Tell me {e}, exactly."]
VERBAL = {
    "+": ["Add {a} and {b}.", "What is the sum of {a} and {b}?", "What do you get if you add {b} to {a}?",
          "Find the total of {a} and {b}.", "{a} plus {b} is what?"],
    "-": ["Subtract {b} from {a}.", "What is {a} minus {b}?", "What is the difference {a} - {b}?",
          "Take {b} away from {a}.", "If I start at {a} and subtract {b}, where do I end up?"],
    "*": ["Multiply {a} by {b}.", "What is the product of {a} and {b}?", "What is {a} times {b}?",
          "How much is {b} lots of {a}?", "{a} multiplied by {b} gives what?"],
    "MOD": ["What is the remainder when {a} is divided by {b}?", "Compute {a} mod {b}.", "Find {a} modulo {b}.",
            "What's left over after dividing {a} by {b}?", "{a} mod {b} = ?"],
    "/": ["What is {a} divided by {b}, dropping the remainder?", "Divide {a} by {b} (integer division).",
          "What is the integer quotient of {a} by {b}?", "How many whole times does {b} go into {a}?",
          "{a} / {b} in integer division is what?"],
}
SHOW = {"+": "+", "-": "-", "*": "*", "MOD": "mod", "/": "/"}
BIG = [" Every digit matters.", " Exactly, please.", "", " No rounding.", " I need all the digits."]


def num(rng, lo, hi, neg=0.0):
    d = rng.randint(lo, hi)
    v = rng.randint(10 ** (d - 1), 10**d - 1)
    return -v if rng.random() < neg else v


def one_op(rng, words, digits, big):
    """digits(op) -> [(lo, hi) of a, (lo, hi) of b]."""
    op = rng.choice([o for o in VERBAL if o in words or o not in NEW])
    digits = digits(op)
    if op in ("MOD", "/"):  # non-negative: no sign-convention questions here
        a, b = num(rng, *digits[0]), num(rng, *digits[1])
    else:
        a, b = num(rng, *digits[0], neg=0.15), num(rng, *digits[1], neg=0.15)
    shown = lambda v: f"({v})" if v < 0 else str(v)
    if rng.random() < 0.6:
        q = rng.choice(ASK).format(e=f"{shown(a)} {SHOW[op]} {shown(b)}") + " Use integer division." * (op == "/")
    else:
        q = rng.choice(VERBAL[op]).format(a=a, b=b)
    return q + (rng.choice(BIG) if big else ""), f"{a} {b} {op} ."


def level0(rng, words):
    return one_op(rng, words, lambda op: [(1, 6), (1, 4)], False)


def level1(rng, words):
    da = rng.randint(10, 18)  # * MOD /: the top operand has <= 10 digits, a product <= 20
    return one_op(rng, words, lambda op: [(da, da), (10, 19) if op in "+-" else (2, min(10, 20 - da))], True)


def expr(rng, k, words):
    """k ops -> (infix, forth, precedence); precedence 0 = always parenthesize as an operand."""
    if k == 0:
        v = num(rng, 1, 2, neg=0.2)
        return (f"({v})" if v < 0 else str(v)), str(v), 9
    kinds = ["bin"] * 6 + ["sq"] + ["neg"] * ("NEGATE" in words) + ["over"] * (k >= 2)
    kind = rng.choice(kinds)
    if kind == "sq":
        i, f, p = expr(rng, k - 1, words)
        return (f"{i}^2" if p == 9 else f"({i})^2"), f"{f} DUP *", 8
    if kind == "neg":
        i, f, p = expr(rng, k - 1, words)
        return f"-({i})", f"{f} NEGATE", 0
    ops = ["+", "-", "*"] * 3 + [o for o in ("MOD", "/") if o in words or o not in NEW]
    op = rng.choice(ops)
    pr = 1 if op in "+-" else 2
    if kind == "over":  # a op (b op2 a): a is reused, OVER copies it
        a, b = num(rng, 1, 2), num(rng, 1, 2)
        op2 = rng.choice("+-*")
        rp, inner = 1 if op2 in "+-" else 2, f"{b} {SHOW[op2]} {a}"
        inner = inner if rp > pr or (rp == pr and op == "+") else f"({inner})"
        return f"{a} {SHOW[op]} {inner}", f"{a} {b} OVER {op2} {op}", pr
    if op in ("MOD", "/"):  # divisor: a positive one-digit literal
        d = str(rng.randint(2, 9))
        (li, lf, lp), (ri, rf, rp) = expr(rng, k - 1, words), (d, d, 9)
    else:
        j = rng.randint(0, k - 1)
        (li, lf, lp), (ri, rf, rp) = expr(rng, j, words), expr(rng, k - 1 - j, words)
    li = li if lp >= pr else f"({li})"
    ri = ri if rp > pr or (rp == pr and op == "+") else f"({ri})"  # a * (b / c) keeps its parentheses
    swap = "SWAP" in words and op in ("-", "/", "MOD") and " " in rf and rng.random() < 0.4
    return f"{li} {SHOW[op]} {ri}", (f"{rf} {lf} SWAP {op}" if swap else f"{lf} {rf} {op}"), pr


L2ASK = ["Evaluate {e} using the usual order of operations.", "Using standard precedence, what is {e}?",
         "Compute {e} (multiplication before addition, as usual).", "What does {e} come out to?",
         "Simplify {e} to a single integer.", "Work out the value of the expression {e}."]
L2NOTE = " (/ is integer division truncating toward zero; mod takes the sign of the dividend.)"


def level2(rng, words):
    i, f, _ = expr(rng, rng.randint(2, 5), words)
    q = rng.choice(ASK + L2ASK).format(e=i)
    return q + (L2NOTE if " mod " in i or " / " in i else ""), f + " ."


SWAPGCD = ": common ( a b -- g ) BEGIN DUP WHILE SWAP OVER MOD REPEAT DROP ;"
VARGCD = "VARIABLE x VARIABLE y\n: common ( a b -- g ) y ! x ! BEGIN y @ WHILE x @ y @ MOD y @ x ! y ! REPEAT x @ ;"
PRIME = (": prime ( n -- f ) DUP n ! 2 < IF 0 EXIT THEN 2 BEGIN DUP DUP * n @ > 0= n @ 2 PICK MOD 0= 0= * WHILE 1+ "
         "REPEAT DUP * n @ > IF 1 ELSE 0 THEN ;")
# family: (words it needs, args from rng, program(args), questions)
L3 = {
    "sumto": (set(), lambda r: [r.randint(3, 60)],
              lambda n: f"VARIABLE t\n: sum ( n -- s ) 0 t ! BEGIN DUP WHILE DUP t @ + t ! 1- REPEAT DROP t @ ;\n{n} sum .",
              ["Add up the whole numbers from 1 to {0}.", "What is 1 + 2 + ... + {0}?",
               "Use a loop to sum the integers 1 through {0}.", "What's the sum of the first {0} positive integers?"]),
    "squares": (set(), lambda r: [r.randint(2, 40)],
                lambda n: f"VARIABLE t\n: squares ( n -- s ) 0 t ! BEGIN DUP WHILE DUP DUP * t @ + t ! 1- REPEAT DROP "
                          f"t @ ;\n{n} squares .",
                ["What is 1^2 + 2^2 + ... + {0}^2?", "Sum the squares of the integers from 1 to {0}.",
                 "Add up the first {0} perfect squares."]),
    "fact": (set(), lambda r: [r.randint(1, 20)],
             lambda n: f"VARIABLE r\n: fact ( n -- n! ) 1 r ! BEGIN DUP 1 > WHILE r @ OVER * r ! 1- REPEAT DROP r @ ;"
                       f"\n{n} fact .",
             ["What is {0} factorial?", "Compute {0}! exactly.", "Write a factorial word and use it for {0}.",
              "How many ways can {0} different books be lined up on a shelf?"]),
    "gcd": (set(), lambda r: [r.randint(2, 10**r.randint(2, 6)), r.randint(2, 10**r.randint(2, 6))], None,
            ["What is the greatest common divisor of {0} and {1}?", "gcd({0}, {1}) = ?",
             "Find the GCD of {0} and {1} with Euclid's algorithm.", "What's the largest number dividing both {0} "
             "and {1}?"]),
    "power": (set(), lambda r: [r.randint(2, 12), r.randint(2, 15)],
              lambda b, e: f"VARIABLE r\n: power ( b e -- b^e ) 1 r ! BEGIN DUP WHILE r @ 2 PICK * r ! 1- REPEAT DROP "
                           f"DROP r @ ;\n{b} {e} power .",
              ["What is {0} to the power {1}?", "Compute {0}^{1} exactly.", "Raise {0} to the {1}th power.",
               "What is {0} multiplied by itself {1} times?"]),
    "powmod": (set(), lambda r: [r.randint(2, 99), r.randint(2, 999), r.randint(2, 30)],
               lambda b, m, e: f"VARIABLE r\n: pow ( b m e -- r ) 1 r ! BEGIN DUP WHILE r @ 3 PICK * 2 PICK MOD r ! "
                               f"1- REPEAT DROP DROP DROP r @ ;\n{b} {m} {e} pow .",
               ["What is {0}^{2} mod {1}?", "Compute {0} to the power {2}, modulo {1}.",
                "Find the remainder of {0}^{2} divided by {1}.", "Modular exponentiation: {0}^{2} (mod {1})?"]),
    "fib": (set(), lambda r: [r.randint(2, 60)],
            lambda n: f"VARIABLE a VARIABLE b\n: fib ( n -- f ) 0 a ! 1 b ! BEGIN DUP WHILE a @ b @ DUP a ! + b ! 1- "
                      f"REPEAT DROP a @ ;\n{n} fib .",
            ["What is the {0}th Fibonacci number (fib(0) = 0, fib(1) = 1)?", "Compute fib({0}) with fib(0)=0, fib(1)=1.",
             "Starting the Fibonacci sequence 0, 1, 1, 2, ..., what is term number {0} (counting from 0)?"]),
    "prime": (set(), lambda r: [r.randint(2, 3000)], lambda n: f"VARIABLE n\n{PRIME}\n{n} prime .",
              ["Is {0} prime? Answer 1 for yes, 0 for no.", "Test whether {0} is a prime number (1 = prime, 0 = not).",
               "Print 1 if {0} is prime and 0 otherwise."]),
    "primes": (set(), lambda r: [r.randint(2, 40)],
               lambda n: f"VARIABLE n VARIABLE c\n{PRIME}\n: primes ( m -- k ) 0 c ! BEGIN DUP 1 > WHILE DUP prime c @ "
                         f"+ c ! 1- REPEAT DROP c @ ;\n{n} primes .",
               ["How many primes are there up to {0}?", "Count the prime numbers between 1 and {0}.",
                "How many primes are less than or equal to {0}?"]),
    "divisors": (set(), lambda r: [r.randint(1, 80)],
                 lambda n: f"VARIABLE n VARIABLE c\n: count ( n -- k ) DUP n ! 0 c ! BEGIN DUP WHILE c @ n @ 2 PICK MOD "
                           f"0= - c ! 1- REPEAT DROP c @ ;\n{n} count .",
                 ["How many divisors does {0} have?", "Count the positive divisors of {0}.",
                  "How many whole numbers divide {0} exactly?"]),
    "digits": ({"/"}, lambda r: [r.randint(1, 10**r.randint(1, 15))],
               lambda n: f"VARIABLE s\n: digits ( n -- s ) 0 s ! BEGIN DUP WHILE DUP 10 MOD s @ + s ! 10 / REPEAT DROP "
                         f"s @ ;\n{n} digits .",
               ["What is the digit sum of {0}?", "Add up the decimal digits of {0}.", "Sum the digits of {0}."]),
    "collatz": ({"/"}, lambda r: [r.randint(2, 60)],
                lambda n: f"VARIABLE c\n: next ( n -- m ) DUP 2 MOD IF 3 * 1+ ELSE 2 / THEN ;\n: steps ( n -- k ) 0 c ! "
                          f"BEGIN DUP 1 > WHILE next c @ 1+ c ! REPEAT DROP c @ ;\n{n} steps .",
                ["How many Collatz steps does it take to get from {0} down to 1?",
                 "Starting from {0}, apply n/2 (even) or 3n+1 (odd) until reaching 1. How many steps?",
                 "Count the 3n+1 steps starting at {0}."]),
}


def level3(rng, words):
    fams = [k for k, v in L3.items() if v[0] <= words]
    _, args, prog, qs = L3[rng.choice(fams)]
    a = args(rng)
    if prog is None:  # gcd: SWAP is the natural Euclid loop
        src = (SWAPGCD if "SWAP" in words else VARGCD) + f"\n{a[0]} {a[1]} common ."
    else:
        src = prog(*a)
    return rng.choice(qs).format(*a), src




def kernel_words():
    return {w for w in NEW if w in sp.V.WORDS} | {"+", "-", "*", "MOD"}


def row(level, seed, words, tok, max_tokens=8192):
    """One verified row; redraws (same seed stream) until ref.py and the kernel agree within the limits. L4 rows carry
    a plan (the story as a formula) for thinking rows; L5 answers are the whole printed line."""
    rng = random.Random(f"curriculum-L{level}-{seed}")
    if level == 6:  # shortcut: raises the module-wide stream cap; L6 streams run to ~70k tokens (eval only)
        sp.MAX_TOKENS, max_tokens = max(sp.MAX_TOKENS, 70000), 70000
    for _ in range(200):
        q, src, *plan = LEVELS[level](rng, words)
        r = {"level": level, "prompt": q, "forth": src} | ({"plan": plan[0]} if plan else {})
        try:
            out = Forth().run(src)
        except Exception:  # noqa: BLE001 - e.g. a division by zero: draw again
            continue
        r["output"] = out
        if ref_problems(r) or not out or (" " in out and level < 5) or (" " not in out and level == 5):
            continue
        p, ids, _ = sp.problems(src, out, tok)
        if p is None and len(ids) <= max_tokens:
            ans = out.replace(" ", "/") if level == 8 and "reduce" in src else (out if " " in out else int(out))
            return {**r, "answer": ans}
    raise RuntimeError(f"L{level} seed {seed}: no verified row in 200 draws")


# L4 word problems: a story with heavy numbers -> an expression or a short program. family: (args, program, questions,
# plan) where plan is the one-line translation a thinking row states before the program.
def _shop(r):
    return [r.randint(12, 9999), r.randint(100, 99999), r.randint(12, 9999), r.randint(100, 99999)]


def _change(r):
    a, p = r.randint(12, 999), r.randint(100, 99999)
    return [a, p, a * p + r.randint(1, 10 ** r.randint(2, 6))]


def _share(r):
    return [r.randint(10**4, 10**9), r.randint(3, 9999)]


def _mean(r):
    xs = [r.randint(10**3, 10**7) for _ in range(r.randint(3, 6))]
    xs[-1] += -sum(xs) % len(xs)  # an exact mean
    return xs


L4 = {
    "shop": (_shop, lambda a, p, b, q: f"{a} {p} * {b} {q} * + .",
             ["A store sells {0} laptops at ${1} each and {2} monitors at ${3} each. What is the total revenue in "
              "dollars?", "An order has {0} boxes costing {1} cents each and {2} crates costing {3} cents each. What "
              "is the total cost in cents?", "A farm ships {0} crates of apples at ${1} a crate and {2} crates of "
              "pears at ${3} a crate. How many dollars does it earn?"],
             "Total = {0} x {1} + {2} x {3}."),
    "change": (_change, lambda a, p, m: f"{m} {a} {p} * - .",
               ["A school buys {0} desks at ${1} each and pays ${2}. How much change does it get back?",
                "You buy {0} tickets that cost {1} cents each and hand over {2} cents. How many cents of change "
                "do you get?"],
               "Change = {2} - {0} x {1}."),
    "rate": (lambda r: [r.randint(2, 999), r.randint(100, 99999), r.randint(2, 9999)],
             lambda k, x, h: f"{k} {x} * {h} * .",
             ["A factory has {0} machines. Each makes {1} parts per hour. How many parts do they make in {2} hours?",
              "{0} servers each handle {1} requests per minute. How many requests in {2} minutes?",
              "A delivery company runs {0} trucks; each drives {1} meters per hour. How many meters do they cover in "
              "total over {2} hours?"],
             "Total = {0} x {1} x {2}."),
    "each": (_share, lambda n, k: f"{n} {k} / .",
             ["{0} marbles are shared equally among {1} children. How many whole marbles does each child get?",
              "A charity splits {0} dollars evenly between {1} schools, in whole dollars. How much does each school "
              "get?"],
             "Each gets {0} / {1}, dropping the remainder."),
    "left": (_share, lambda n, k: f"{n} {k} MOD .",
             ["{0} eggs are packed into cartons of {1}. How many eggs are left over?",
              "{0} people need buses that seat {1} each. After filling as many full buses as possible, how many "
              "people are left?"],
             "Left over = {0} mod {1}."),
    "percent": (lambda r: [100 * r.randint(10, 10**6), r.randint(1, 99)], lambda n, x: f"{n} {x} * 100 / .",
                ["A town has {0} residents and {1}% of them voted. How many people voted?",
                 "A company with revenue ${0} spends {1}% of it on research. How many dollars is that?",
                 "What is {1}% of {0}?"],
                "Amount = {0} x {1} / 100."),
    "interest": (lambda r: [100 * r.randint(10, 10**5), r.randint(1, 15), r.randint(2, 30)],
                 lambda p, x, t: f"{p} {x} * {t} * 100 / .",
                 ["${0} is invested at {1}% simple interest per year. How much interest does it earn in {2} years?",
                  "A loan of {0} dollars charges {1}% simple interest a year. What is the total interest after {2} "
                  "years?"],
                 "Interest = {0} x {1} x {2} / 100."),
    "grow": (lambda r: [r.randint(2, 999), r.choice([2, 3]), r.randint(3, 25)],
             lambda a, f, n: f": grow ( a n -- b ) BEGIN DUP WHILE SWAP {f} * SWAP 1- REPEAT DROP ;\n{a} {n} grow .",
             ["A colony starts with {0} bacteria and multiplies by {1} every hour. How many are there after {2} "
              "hours?", "A chain letter is sent to {0} people, and each round every recipient's count is multiplied "
              "by {1}. How many people receive it in round {2} (round 0 = {0})?"],
             "Start at {0} and multiply by {1}, {2} times: a loop."),
    "frame": (lambda r: [r.randint(10, 99999), r.randint(10, 99999), r.randint(1, 999)],
              lambda w, h, b: f"{w} {b} 2 * + {h} {b} 2 * + * {w} {h} * - .",
              ["A picture is {0} mm by {1} mm and has a frame {2} mm wide all around it. What is the area of the frame "
               "alone, in square mm?", "A pool measuring {0} cm by {1} cm is surrounded by a path {2} cm wide. What is "
               "the area of the path in square cm?"],
              "Frame area = ({0} + 2 x {2}) x ({1} + 2 x {2}) - {0} x {1}."),
    "seconds": (lambda r: [r.randint(1, 99999), r.randint(0, 23), r.randint(0, 59), r.randint(0, 59)],
                lambda d, h, m, s: f"{d} 24 * {h} + 60 * {m} + 60 * {s} + .",
                ["How many seconds are in {0} days, {1} hours, {2} minutes and {3} seconds?",
                 "A probe has flown for {0} days {1} hours {2} minutes {3} seconds. How many seconds is that?"],
                "Seconds = (({0} x 24 + {1}) x 60 + {2}) x 60 + {3}."),
    "saving": (lambda r: [r.randint(1, 9999), r.randint(1, 999), r.randint(5, 60)],
               lambda a, d, n: f"VARIABLE t VARIABLE w\n: save ( n -- s ) 0 t ! {a} w ! BEGIN DUP WHILE w @ t @ + t ! "
                               f"w @ {d} + w ! 1- REPEAT DROP t @ ;\n{n} save .",
               ["You save ${0} in week 1 and ${1} more each week than the week before. How much have you saved after "
                "{2} weeks?", "A runner does {0} meters on day 1 and adds {1} meters every day. What is the total "
                "distance over {2} days?"],
               "Add up {2} terms starting at {0}, each {1} more than the last: a loop."),
    "handshake": (lambda r: [r.randint(3, 99999)], lambda n: f"{n} DUP 1- * 2 / .",
                  ["At a conference, each of {0} people shakes hands once with every other person. How many "
                   "handshakes are there?", "A league of {0} teams plays every other team exactly once. How many "
                   "games is that?"],
                  "Pairs = {0} x ({0} - 1) / 2."),
    "choose": (lambda r: [r.randint(5, 40), r.randint(2, 8)],
               lambda n, k: f"VARIABLE r VARIABLE i\n: choose ( n k -- c ) 1 r ! 0 i ! BEGIN DUP i @ > WHILE i @ 1+ "
                            f"i ! r @ 2 PICK 2 PICK - i @ + * i @ / r ! REPEAT DROP DROP r @ ;\n{n} {k} choose .",
               ["In how many ways can a committee of {1} be chosen from {0} people?",
                "How many different hands of {1} cards can be drawn from {0} distinct cards?",
                "A pizza place has {0} toppings. How many pizzas with exactly {1} different toppings can it make?"],
               "C({0}, {1}): multiply and divide step by step in a loop."),
    "buses": (lambda r: [r.randint(2, 99999), r.randint(2, 99999)],
              lambda a, b: f": common ( a b -- g ) BEGIN DUP WHILE SWAP OVER MOD REPEAT DROP ;\n{a} {b} * {a} {b} "
                           f"common / .",
              ["Bus A leaves every {0} seconds and bus B every {1} seconds. Both leave now. After how many seconds do "
               "they next leave together?", "What is the least common multiple of {0} and {1}?",
               "Two lighthouses flash every {0} and {1} milliseconds, together right now. When do they next flash "
               "together?"],
              "Together again at lcm({0}, {1}) = {0} x {1} / gcd({0}, {1})."),
    "number": (lambda r: (lambda a, x, b: [a, b, a * x + b])(r.randint(2, 9999), r.randint(1, 10**6), r.randint(1, 10**6)),
              lambda a, b, c: f"{c} {b} - {a} / .",
              ["I think of a number, multiply it by {0}, then add {1}, and get {2}. What was my number?",
               "Solve for x: {0}x + {1} = {2}."],
              "x = ({2} - {1}) / {0}."),
    "ages": (lambda r: (lambda y, d: [2 * y + d, d])(r.randint(1, 10**7), r.randint(1, 10**5)),
             lambda s, d: f"{s} {d} + 2 / .",
             ["Two numbers add up to {0} and one is {1} more than the other. What is the larger one?",
              "A and B have {0} dollars together; A has {1} dollars more than B. How many dollars does A have?"],
             "Larger = ({0} + {1}) / 2."),
    "multiples": (lambda r: (lambda a: [a, a + r.randint(10, 10**8), r.randint(2, 999)])(r.randint(1, 10**8)),
                  lambda a, b, k: f"{b} {k} / {a} 1- {k} / - .",
                  ["How many integers from {0} to {1} inclusive are divisible by {2}?",
                   "Count the multiples of {2} between {0} and {1}, inclusive."],
                  "Count = {1} / {2} - ({0} - 1) / {2}."),
    "meet": (lambda r: (lambda a, b, t: [(a + b) * t, a, b])(r.randint(10, 999), r.randint(10, 999), r.randint(1, 9999)),
             lambda d, a, b: f"{d} {a} {b} + / .",
             ["Two trains are {0} km apart and head toward each other at {1} km/h and {2} km/h. How many hours until "
              "they meet?", "Two robots {0} meters apart roll toward each other at {1} and {2} meters per minute. "
              "After how many minutes do they meet?"],
             "Time = {0} / ({1} + {2})."),
    "profit": (lambda r: [r.randint(10, 99999), r.randint(50, 9999), r.randint(10, 9999), r.randint(10, 10**8)],
               lambda n, p, c, f: f"{n} {p} {c} - * {f} - .",
               ["A bakery sells {0} cakes at ${1} each. Each cake costs ${2} to make, and rent is ${3}. What is the "
                "profit in dollars (negative for a loss)?", "A studio sells {0} licenses at ${1}; each costs ${2} to "
                "support, and fixed costs are ${3}. What is the net profit (negative means a loss)?"],
               "Profit = {0} x ({1} - {2}) - {3}."),
    "mean": (_mean, lambda *xs: " ".join([str(xs[0])] + [f"{x} +" for x in xs[1:]]) + f" {len(xs)} / .",
             ["The scores were {all}. What is the mean score?", "Find the average of {all}.",
              "A shop's daily sales were {all} dollars. What were the average daily sales?"],
             "Mean = ({plus}) / {n}."),
}


def level4(rng, words):
    fam = rng.choice(sorted(L4))
    args, prog, qs, plan = L4[fam]
    a = args(rng)
    fmt = {"all": ", ".join(map(str, a[:-1])) + f" and {a[-1]}", "plus": " + ".join(map(str, a)), "n": len(a)}
    return rng.choice(qs).format(*a, **fmt), prog(*a), plan.format(*a, **fmt)


# L5 programs: recursion and printed sequences (the shapes larger programs are built from). The answer is the whole
# printed line. Recursive words test their argument first and EXIT, as one IF per definition allows.
L5 = {
    "ruler": (lambda r: [r.randint(1, 6)],
              lambda n: f": ruler ( n -- ) DUP 0= IF DROP EXIT THEN DUP 1- ruler DUP . 1- ruler ;\n{n} ruler",
              ["Print the ruler sequence of order {0}: order n is order n-1, then n, then order n-1 again; order 0 is "
               "empty.", "Write a recursive word that prints, for n = {0}: the pattern for n-1, then n, then the "
               "pattern for n-1 (nothing for 0)."]),
    "down": (lambda r: [r.randint(2, 30)], lambda n: f": down ( n -- ) BEGIN DUP WHILE DUP . 1- REPEAT DROP ;\n{n} down",
             ["Count down from {0} to 1, printing each number.", "Print {0}, {0}-1, ..., 1 with a loop."]),
    "rdown": (lambda r: [r.randint(2, 30)], lambda n: f": down ( n -- ) DUP 0= IF DROP EXIT THEN DUP . 1- down ;\n{n} down",
              ["Using recursion, print the numbers from {0} down to 1.", "Write a recursive countdown from {0}."]),
    "rup": (lambda r: [r.randint(2, 30)], lambda n: f": up ( n -- ) DUP 0= IF DROP EXIT THEN DUP 1- up . ;\n{n} up",
            ["Using recursion, print the numbers from 1 up to {0}.",
             "Write a recursive word that prints 1, 2, ..., {0}: it handles n-1 first, then prints n."]),
    "binary": (lambda r: [r.randint(2, 10**6)],
               lambda n: f": bin ( n -- ) DUP 2 < IF . EXIT THEN DUP 2 / bin 2 MOD . ;\n{n} bin",
               ["Print the binary digits of {0}, most significant first, separated by spaces.",
                "Using recursion, print {0} in base 2, one digit at a time."]),
    "multiples": (lambda r: [r.randint(2, 10**6), r.randint(2, 20)],
                  lambda m, k: f"VARIABLE i\n: row ( m k -- ) 1 i ! BEGIN DUP i @ < 0= WHILE OVER i @ * . i @ 1+ i ! "
                               f"REPEAT DROP DROP ;\n{m} {k} row",
                  ["Print the first {1} multiples of {0}.", "Print {0} x 1, {0} x 2, ..., {0} x {1}."]),
    "squares": (lambda r: [r.randint(2, 25)],
                lambda n: f"VARIABLE i\n: show ( n -- ) 1 i ! BEGIN DUP i @ < 0= WHILE i @ DUP * . i @ 1+ i ! REPEAT "
                          f"DROP ;\n{n} show",
                ["Print the squares of 1 through {0}.", "List 1^2, 2^2, ..., {0}^2."]),
    "pairs": (lambda r: [r.randint(1, 8), r.randint(0, 999), r.randint(0, 999)],
              lambda n, a, b: f": pairs ( n a b -- ) 2 PICK 0= IF DROP DROP DROP EXIT THEN OVER . DUP . 2 PICK 1- OVER "
                              f"3 PICK OVER + pairs DROP DROP DROP ;\n{n} {a} {b} pairs",
              ["Start with the pair {1}, {2}. Each next pair is (second, first + second). Using a recursive word "
               "with three arguments, print both numbers of each of the first {0} pairs.",
               "Recursively print {0} pairs, starting from ({1}, {2}), where (a, b) is followed by (b, a + b)."]),
    "paths": (lambda r: [r.randint(1, 4), r.randint(1, 4)],
              lambda a, b: f": paths ( r c -- n ) OVER 0= OVER 0= + IF DROP DROP 1 EXIT THEN OVER 1- OVER paths 2 PICK "
                           f"2 PICK 1- paths + SWAP DROP SWAP DROP ;\n{a} {b} paths .",
              ["Moving only right or down, how many paths cross a {0} by {1} grid of blocks from one corner to the "
               "opposite? Use a recursive word: paths(r, c) = paths(r-1, c) + paths(r, c-1), and 1 when r or c is 0.",
               "Recursively count lattice paths: p(r, c) = 1 if r = 0 or c = 0, else p(r-1, c) + p(r, c-1). What is "
               "p({0}, {1})?"]),
}


def level5(rng, words):
    _, args, prog, qs = (None, *L5[rng.choice(sorted(L5))])
    a = args(rng)
    return rng.choice(qs).format(*a), prog(*a)


LEVELS = [level0, level1, level2, level3, level4, level5]


# Sessions: multi-turn stateful programs. Turn 1 states the rules and the model writes the words (and runs the first
# step); later turns are short requests answered by calling a word. Kernel state (variables, definitions, the seed)
# lives in earlier turns, so these rows are thinking-off (the chat template drops earlier turns' reasoning).
# A family: setup(rng) -> (state, prompt, program, reply(out)); act(rng, state, turn) -> (user, forth, reply(out)) or
# None to end. SIGN: (a - b) -> 1 / 0 / -1 (true is -1).
SIGN = ": sign ( n -- s ) DUP 0 < SWAP 0 > - ;"


def _bank_setup(r):
    s = r.randint(100, 10**6)
    q = r.choice(["Let's track a bank account that starts with {s} dollars. Define words to deposit an amount, to "
                  "withdraw an amount (refuse a withdrawal bigger than the balance by printing -1), and to show the "
                  "balance. Show it now.",
                  "Be my account ledger. Opening balance: {s}. I'll ask you to deposit, withdraw (if I ask for more than "
                  "I have, print -1 and change nothing) or show the balance. Set it up and show the balance."])
    src = ("VARIABLE money\n: deposit ( n -- ) money @ + DUP money ! . ;\n: withdraw ( n -- ) DUP money @ > IF DROP -1 . "
           f"EXIT THEN money @ SWAP - DUP money ! . ;\n: balance ( -- ) money @ . ;\n{s} money ! balance")
    return {"bal": s}, q.format(s=s), src, lambda o: f"Your account is set up with a balance of {o}."


def _bank_act(r, st, turn):
    if turn > r.randint(3, 7):
        return None
    k = r.choice(["deposit", "withdraw", "withdraw", "balance"])
    if k == "deposit":
        x = r.randint(1, 10**5)
        st["bal"] += x
        return (r.choice(["Deposit {x}.", "Put in {x} dollars.", "I'm adding {x}.", "deposit {x}"]).format(x=x),
                f"{x} deposit", lambda o: f"Deposited {x}. The balance is now {o}.")
    if k == "withdraw":
        x = r.randint(1, st["bal"]) if r.random() < 0.7 else st["bal"] + r.randint(1, 10**5)
        st["bal"] -= x * (x <= st["bal"])
        return (r.choice(["Withdraw {x}.", "Take out {x} dollars.", "I need {x} in cash.", "withdraw {x}"]).format(x=x),
                f"{x} withdraw", lambda o: "Refused: that's more than the balance." if o == "-1" else
                f"Withdrew {x}. The balance is now {o}.")
    return (r.choice(["What's the balance?", "How much is in the account?", "balance", "Show me the balance."]),
            "balance", lambda o: f"The balance is {o}.")


def _dice_setup(r):
    seed, top = r.randint(1, 65000), r.choice([15, 18, 20, 24, 30])
    st = {"top": top, "stand": top - 4, "rolls": 0, "rounds": 0}
    q = (f"Let's play a dice game. Random numbers: keep a seed, starting at {seed}; each draw sets seed = (seed x 75 + "
         f"74) mod 65537, and the die is seed mod 6 + 1. A round starts with two dice for me and one for you, the house. "
         f"I can roll to add a die to my total; over {top} I bust. When I stay, you roll until you have at least "
         f"{st['stand']}, and the higher total that hasn't busted wins. Print: at the start, my total and your die; on "
         f"a roll, the die and my new total; when I stay, your final total and the result (1 I win, 0 tie, -1 you "
         f"win). Start a round.")
    src = ("VARIABLE seed VARIABLE me VARIABLE house\n: die ( -- d ) seed @ 75 * 74 + 65537 MOD DUP seed ! 6 MOD 1+ ;\n"
           ": start ( -- ) die die + DUP me ! . die DUP house ! . ;\n: roll ( -- ) die DUP . me @ + DUP me ! . ;\n"
           f"{SIGN}\n: result ( -- ) me @ {top} > IF -1 . EXIT THEN house @ {top} > IF 1 . EXIT THEN me @ house @ - "
           f"sign . ;\n: stay ( -- ) BEGIN house @ {st['stand']} < WHILE die house @ + house ! REPEAT house @ . "
           f"result ;\n{seed} seed ! start")
    return st, q, src, lambda o: "Your total is {}; my die shows {}. Roll or stay?".format(*o.split())


def _dice_act(r, st, turn):
    if st["rolls"] < 0:  # the round is over
        if st["rounds"] >= r.randint(1, 2):
            return None
        st["rounds"], st["rolls"] = st["rounds"] + 1, 0
        return (r.choice(["Start another round.", "Again!", "New round, please."]), "start",
                lambda o: "New round: your total is {}; my die shows {}. Roll or stay?".format(*o.split()))
    if st["rolls"] < r.randint(0, 3):
        st["rolls"] += 1
        top = st["top"]
        return (r.choice(["Roll.", "roll", "I'll roll again.", "One more die."]), "roll",
                lambda o: "You rolled {}; your total is {}.".format(*o.split()) + (
                    f" That's over {top}: you bust." if int(o.split()[1]) > top else " Roll or stay?"))
    st["rolls"] = -1
    return (r.choice(["Stay.", "I'll stay.", "stay", "I'm done, your turn."]), "stay",
            lambda o: "I finish with {}. ".format(o.split()[0]) + {"1": "You win!", "0": "It's a tie.",
                                                                   "-1": "I win."}[o.split()[1]])


def _guess_setup(r):
    seed = r.randint(1, 65000)
    s, st = seed, {"lo": 1, "hi": 100}
    s = (s * 25173 + 13849) % 65536
    st["secret"] = s % 100 + 1
    q = (f"Let's play higher or lower. Pick a secret number from 1 to 100 with seed {seed}: seed = (seed x 25173 + "
         f"13849) mod 65536, and the secret is seed mod 100 + 1. Don't print it. When I guess, print 1 if my guess is "
         f"too high, -1 if too low and 0 if it's right.")
    src = ("VARIABLE seed VARIABLE secret\n: hide ( -- ) seed @ 25173 * 13849 + 65536 MOD DUP seed ! 100 MOD 1+ secret "
           f"! ;\n{SIGN}\n: guess ( n -- ) secret @ - sign . ;\n{seed} seed ! hide")
    return st, q, src, lambda o: "I've picked a secret number from 1 to 100. Your guess?"


def _guess_act(r, st, turn):
    if st["lo"] > st["hi"] or turn > 8:
        return None
    g = r.randint(st["lo"], st["hi"]) if r.random() < 0.5 else (st["lo"] + st["hi"]) // 2
    if g == st["secret"]:
        st["lo"] = st["hi"] + 1
    elif g > st["secret"]:
        st["hi"] = g - 1
    else:
        st["lo"] = g + 1
    return (r.choice(["Is it {g}?", "{g}", "My guess is {g}.", "I'll say {g}."]).format(g=g), f"{g} guess",
            lambda o: {"1": f"{g} is too high.", "-1": f"{g} is too low.", "0": f"Yes, {g} is right!"}[o])


TEAMS = ["Hawks", "Bears", "Comets", "Tigers", "Rovers", "Owls", "Pirates", "Wolves"]


def _score_setup(r):
    a, b = r.sample(TEAMS, 2)
    q = r.choice(["Keep score for a game between the {a} and the {b}, starting at 0 to 0. Make words to add points to "
                  "each team (print that team's new total) and to show both totals. Show the score.",
                  "Be the scoreboard: {a} versus {b}, both at 0. Points come in as I tell you; print the team's new "
                  "total each time, and both totals when I ask for the score. Show it now."])
    src = ("VARIABLE home VARIABLE away\n: first ( n -- ) home @ + DUP home ! . ;\n: second ( n -- ) away @ + DUP away "
           "! . ;\n: score ( -- ) home @ . away @ . ;\n0 home ! 0 away ! score")
    return {"a": a, "b": b}, q.format(a=a, b=b), src, lambda o: f"{a} 0, {b} 0. Let's go!"


def _score_act(r, st, turn):
    if turn > r.randint(3, 7):
        return None
    if r.random() < 0.25:
        return (r.choice(["What's the score?", "score", "Score check."]), "score",
                lambda o: "{} {}, {} {}.".format(st["a"], o.split()[0], st["b"], o.split()[1]))
    x, w = r.randint(1, 99999), r.choice(["first", "second"])
    t = st["a"] if w == "first" else st["b"]
    return (r.choice(["{x} points for the {t}.", "The {t} score {x}.", "{t} +{x}"]).format(x=x, t=t), f"{x} {w}",
            lambda o: f"The {t} now have {o}.")


SESSIONS = {"bank": (_bank_setup, _bank_act), "dice": (_dice_setup, _dice_act), "guess": (_guess_setup, _guess_act),
            "score": (_score_setup, _score_act)}


def session(seed, tok, max_tokens=12000):
    """One verified multi-turn session: every turn's printed output agrees between ref.py (one machine across turns)
    and the kernel streamed over all turns' programs; turns[i] = {prompt, forth, output, reply}."""
    rng = random.Random(f"session-{seed}")
    fam = rng.choice(sorted(SESSIONS))
    setup, act = SESSIONS[fam]
    st, q, src, reply = setup(rng)
    turns, f = [{"prompt": q, "forth": src, "reply": reply}], Forth()
    while (a := act(rng, st, len(turns))) is not None:
        turns.append(dict(zip(("prompt", "forth", "reply"), a)))
    for t in turns:
        n = len(f.out)
        f.run(t["forth"])
        t["output"] = " ".join(f.out[n:])
        t["reply"] = t["reply"](t["output"])
    full = "\n".join(t["forth"] for t in turns)
    assert not ref_problems({"prompt": " ".join(t["prompt"] for t in turns), "forth": full, "output": " ".join(f.out)})
    p, ids, _ = sp.problems(full, " ".join(f.out), tok)
    assert p is None and len(ids) <= max_tokens, (fam, p)
    return {"family": fam, "turns": turns}


# L6 "needs the tool": long exact work (~100-1500 loop iterations, <= ~60k kernel tokens, big numbers, every intermediate exact)
# that reasoning alone gets wrong; the headline benchmark (eval). Inner loops live in their own words (no nesting).
# family: (args, program, questions); answers are one integer or, for trace families, the whole printed line.
L6 = {
    "count_digitsum": (lambda r: [r.randint(80, 180), r.randint(3, 13)],
                       lambda n, k: f"VARIABLE s VARIABLE c\n: ds ( n -- s ) 0 s ! BEGIN DUP WHILE DUP 10 MOD s @ + s ! "
                                    f"10 / REPEAT DROP s @ ;\n: count ( n -- c ) 0 c ! BEGIN DUP WHILE c @ OVER ds {k} "
                                    f"MOD 0= - c ! 1- REPEAT DROP c @ ;\n{n} count .",
                       ["How many integers from 1 to {0} have a digit sum divisible by {1}?",
                        "Count the numbers n with 1 <= n <= {0} whose decimal digits add up to a multiple of {1}."]),
    "count_quadres": (lambda r: [r.randint(100, 350), r.randint(7, 97)],
                      lambda n, m: f"VARIABLE c\n: count ( n -- c ) 0 c ! BEGIN DUP WHILE c @ OVER DUP * {m} MOD 1 - 0= "
                                   f"- c ! 1- REPEAT DROP c @ ;\n{n} count .",
                      ["How many integers n from 1 to {0} satisfy n^2 mod {1} = 1?",
                       "For how many n in 1..{0} does n squared leave remainder 1 when divided by {1}?"]),
    "crt_search": (lambda r: (lambda a, b, n0: [a, b, n0 % a, n0 % b])(r.randint(11, 40), r.randint(11, 40),
                                                                       r.randint(200, 1000)),
                   lambda a, b, x, y: f": ok ( n -- f ) DUP {a} MOD {x} - 0= SWAP {b} MOD {y} - 0= * ;\n: find ( -- n ) "
                                      f"0 BEGIN DUP ok 0= WHILE 1+ REPEAT ;\nfind .",
                   ["What is the smallest non-negative integer n with n mod {0} = {2} and n mod {1} = {3}?",
                    "Find the least n >= 0 that leaves remainder {2} on division by {0} and remainder {3} on division "
                    "by {1}."]),
    "lcg": (lambda r: [r.randint(1, 10**5), r.randint(2, 99999), r.randint(0, 99999), r.randint(10**4, 10**6),
                       r.randint(60, 160)],
            lambda s, a, c, m, n: f": run ( x n -- x ) BEGIN DUP WHILE SWAP {a} * {c} + {m} MOD SWAP 1- REPEAT DROP "
                                  f";\n{s} {n} run .",
            ["Let x_0 = {0} and x_(k+1) = ({1} * x_k + {2}) mod {3}. What is x_{4}?",
             "A random number generator starts at {0} and repeatedly replaces x with ({1}x + {2}) mod {3}. What is "
             "the value after {4} steps?"]),
    "recurrence": (lambda r: [r.randint(0, 999), r.randint(0, 999), r.randint(2, 9), r.randint(10**3, 10**6),
                              r.randint(60, 200)],
                   lambda a, b, k, m, n: f"VARIABLE x VARIABLE y\n: rec ( n -- a ) BEGIN DUP WHILE y @ DUP x @ {k} * + "
                                         f"{m} MOD y ! x ! 1- REPEAT DROP x @ ;\n{a} x ! {b} y ! {n} rec .",
                   ["a_0 = {0}, a_1 = {1}, and a_(n+2) = (a_(n+1) + {2} * a_n) mod {3}. What is a_{4}?",
                    "A sequence starts {0}, {1}; each next term is (previous + {2} x the one before it) mod {3}. What "
                    "is term number {4} (the first term is number 0)?"]),
    "compound": (lambda r: [r.randint(1000, 99999), r.randint(1, 9), r.randint(20, 60)],
                 lambda p, x, y: f": grow ( p y -- p ) BEGIN DUP WHILE SWAP DUP {x} * 100 / + SWAP 1- REPEAT DROP ;\n"
                                 f"{p} {y} grow .",
                 ["An account holds ${0}. Each year it earns {1}% interest, rounded down to a whole dollar, and the "
                  "interest is added. How many dollars after {2} years?",
                  "A population of {0} grows by {1}% per year (fractional individuals dropped each year). What is it "
                  "after {2} years?"]),
    "lcg_trace": (lambda r: [r.randint(1, 10**5), r.randint(2, 99999), r.randint(0, 99999), r.randint(10**4, 10**6),
                             r.randint(20, 40)],
                  lambda s, a, c, m, n: f": show ( x n -- ) BEGIN DUP WHILE SWAP {a} * {c} + {m} MOD DUP . SWAP 1- "
                                        f"REPEAT DROP DROP ;\n{s} {n} show",
                  ["Let x_0 = {0} and x_(k+1) = ({1} * x_k + {2}) mod {3}. Print x_1 through x_{4}, in order.",
                   "Starting from {0}, apply x -> ({1}x + {2}) mod {3} {4} times and print every value produced."]),
    "euclid_trace": (lambda r: [r.randint(10**8, 10**10), r.randint(10**8, 10**10)],
                     lambda a, b: f": steps ( a b -- ) BEGIN DUP WHILE DUP . SWAP OVER MOD REPEAT DROP . ;\n{a} {b} "
                                  f"steps",
                     ["Run Euclid's algorithm on ({0}, {1}): print {1}, then every nonzero remainder as it appears, "
                      "then the gcd.", "Euclid on {0} and {1}: print the second number, each nonzero remainder in "
                      "order, and finally the gcd."]),
}


def level6(rng, words):
    _, args, prog, qs = (None, *L6[rng.choice(sorted(L6))])
    a = args(rng)
    return rng.choice(qs).format(*a), prog(*a)


LEVELS.append(level6)


# L7 = small L6 (training): the same programs at 5-40 iterations, so rows fit training (<= ~8k kernel tokens); L6's
# large sizes stay unseen, and the eval measures whether the skill carries over to long runs.
L6_SMALL = {
    "count_digitsum": lambda r: [r.randint(10, 40), r.randint(3, 13)],
    "count_quadres": lambda r: [r.randint(10, 60), r.randint(7, 97)],
    "crt_search": lambda r: (lambda a, b, n0: [a, b, n0 % a, n0 % b])(r.randint(3, 12), r.randint(3, 12), r.randint(5, 60)),
    "lcg": lambda r: [r.randint(1, 10**5), r.randint(2, 99999), r.randint(0, 99999), r.randint(10**4, 10**6),
                      r.randint(3, 20)],
    "recurrence": lambda r: [r.randint(0, 999), r.randint(0, 999), r.randint(2, 9), r.randint(10**3, 10**6),
                             r.randint(3, 25)],
    "compound": lambda r: [r.randint(1000, 99999), r.randint(1, 9), r.randint(3, 15)],
    "lcg_trace": lambda r: [r.randint(1, 10**5), r.randint(2, 99999), r.randint(0, 99999), r.randint(10**4, 10**6),
                            r.randint(3, 12)],
    "euclid_trace": lambda r: [r.randint(10**4, 10**10), r.randint(10**4, 10**10)],
}


# The variant knob for the L6 families nobody solves at full size (stage 2a, training/stage2a_sets.py): the same programs
# and questions on small inputs (short traces, few steps), so some rollouts succeed and there is something to learn from.
L6_EASY = {
    "recurrence": lambda r: [r.randint(0, 99), r.randint(0, 99), r.randint(2, 9), r.randint(100, 10**4), r.randint(6, 30)],
    "euclid_trace": lambda r: [r.randint(10**3, 10**6), r.randint(10**3, 10**6)],
}


def level7(rng, words):
    fam = rng.choice(sorted(L6_SMALL))
    _, prog, qs = L6[fam]
    a = L6_SMALL[fam](rng)
    return rng.choice(qs).format(*a), prog(*a)


LEVELS.append(level7)


# L8 math methods: the model supplies the method (antiderivative, derivative, Cramer's rule, rational roots, closed
# forms, extended Euclid), the kernel the heavy exact arithmetic. Answers: an integer, "x y", or a reduced fraction
# p/q (printed "p q"; row() boxes it as p/q). Kernel words only (no return stack): helpers use variables.
def _poly_str(cs):  # [c0, c1, ...] -> "5x^3 - 2x^2 + 7x - 4"
    terms = [f"{c}x^{k}" if k > 1 else (f"{c}x" if k == 1 else f"{c}") for k, c in reversed(list(enumerate(cs))) if c]
    return " + ".join(terms).replace("+ -", "- ")


def _horner(cs):
    """Forth body ( x -- p(x) ) by Horner's rule; x kept in variable hx."""
    body = f"hx ! {cs[-1]}"
    for c in reversed(cs[:-1]):
        body += f" hx @ * {c} +"
    return body


GCDW = ": common ( a b -- g ) BEGIN DUP WHILE SWAP OVER MOD REPEAT DROP ;"
REDUCE = (f"VARIABLE g\n{GCDW}\n: reduce ( n d -- n' d' ) OVER OVER common DUP 0 < IF NEGATE THEN g ! g @ / SWAP g @ / "
          f"SWAP ;")


def _cubic_roots(r):
    rs = sorted(r.sample(range(-50, 51), 3))
    k = r.choice([1, 2, 3, 5])
    c3, c2, c1, c0 = k, -k * sum(rs), k * (rs[0] * rs[1] + rs[0] * rs[2] + rs[1] * rs[2]), -k * rs[0] * rs[1] * rs[2]
    return [c0, c1, c2, c3], rs


def _quad(r):
    a, p, q = r.choice([1, 2, 3, 4, 6, 7]), r.randint(-5000, 5000), r.randint(-5000, 5000)
    return a, -a * (p + q), a * p * q


def _lin2(r):
    x, y = r.randint(-10**4, 10**4), r.randint(-10**4, 10**4)
    while True:
        a1, b1, a2, b2 = (r.randint(-999, 999) for _ in range(4))
        if a1 * b2 - a2 * b1:
            return [a1, b1, a1 * x + b1 * y, a2, b2, a2 * x + b2 * y]


def _congr(r):
    from math import gcd
    m = r.randint(10**5, 10**9)
    while True:
        a = r.randint(2, m - 1)
        if gcd(a, m) == 1:
            return [a, r.randint(1, m - 1), m]


L8 = {
    "integral": (lambda r: [r.randint(-999, 999) for _ in range(4)] + [r.randint(-20, 10), r.randint(11, 40)],
                 lambda c0, c1, c2, c3, a, b: f"VARIABLE hx\n: F ( x -- 12F ) {_horner([0, 12 * c0, 6 * c1, 4 * c2, 3 * c3])} "
                                              f";\n{REDUCE}\n{b} F {a} F - 12 reduce SWAP . .",
                 ["Evaluate the definite integral of {p} with respect to x from x = {4} to x = {5}. Give the exact value as "
                  "a reduced fraction.", "Compute the exact value of the integral from {4} to {5} of ({p}) dx, in lowest "
                  "terms."], lambda a: _poly_str(a[:4])),
    "derivative": (lambda r: [r.randint(-999, 999) for _ in range(5)] + [r.randint(100, 9999)],
                   lambda c0, c1, c2, c3, c4, x: f"VARIABLE hx\n: d ( x -- f' ) {_horner([c1, 2 * c2, 3 * c3, 4 * c4])} ;\n"
                                                f"{x} d .",
                   ["Let f(x) = {p}. What is f'({5})?", "Find the slope of the tangent line to y = {p} at x = {5}."],
                   lambda a: _poly_str(a[:5])),
    "cubic": (lambda r: (lambda cs, rs: cs)(*_cubic_roots(r)),
              lambda c0, c1, c2, c3: f"VARIABLE hx\n: p ( x -- y ) {_horner([c0, c1, c2, c3])} ;\n: top ( -- r ) 60 BEGIN "
                                     f"DUP p WHILE 1- REPEAT ;\ntop .",
              ["The polynomial {p} has three integer roots. What is the largest one?",
               "Find the greatest integer x with {p} = 0 (all its roots are integers)."], lambda a: _poly_str(a)),
    "quadratic": (lambda r: list(_quad(r)),
                  lambda a, b, c: f"VARIABLE n\n: root ( d -- s ) DUP n ! BEGIN DUP DUP * n @ > WHILE DUP n @ OVER / + 2 / "
                                  f"REPEAT ;\n{b} DUP * {c} {4 * a} * - root {b} NEGATE + {2 * a} / .",
                  ["Solve {p} = 0 and give the larger root.", "What is the larger solution of the equation {p} = 0?"],
                  lambda a: _poly_str([a[2], a[1], a[0]])),
    "linear2": (_lin2,
                lambda a1, b1, c1, a2, b2, c2: f"{a1} {b2} * {a2} {b1} * - DUP {c1} {b2} * {c2} {b1} * - SWAP / . {a1} {c2} "
                                               f"* {a2} {c1} * - SWAP / .",
                ["Solve the system {0}x + {1}y = {2} and {3}x + {4}y = {5}. Give x, then y.",
                 "Find the integers x and y with {0}x + {1}y = {2} and {3}x + {4}y = {5} (x first, then y)."],
                lambda a: None),
    "congruence": (_congr,
                   lambda a, b, m: f"VARIABLE ra VARIABLE rb VARIABLE ta VARIABLE tb VARIABLE qq\n: inv ( -- t ) {m} ra ! {a} "
                                   f"rb ! 0 ta ! 1 tb ! BEGIN rb @ WHILE ra @ rb @ / qq ! rb @ ra @ qq @ rb @ * - rb ! ra ! "
                                   f"tb @ ta @ qq @ tb @ * - tb ! ta ! REPEAT ta @ ;\ninv {m} + {m} MOD {b} * {m} MOD .",
                   ["Find the smallest non-negative integer x with {0}x ≡ {1} (mod {2}).",
                    "Solve the congruence {0}x = {1} (mod {2}) for the least x >= 0."], lambda a: None),
    "geometric": (lambda r: [r.randint(1, 9999), r.randint(2, 9), r.randint(5, 12)],
                  lambda a, q, n: f": pw ( b e -- b^e ) 1 SWAP BEGIN DUP WHILE SWAP 2 PICK * SWAP 1- REPEAT DROP SWAP DROP ;\n"
                                  f"{q} {n} pw 1 - {a} * {q} 1 - / .",
                  ["What is the sum of the first {2} terms of the geometric series {0} + {0}*{1} + {0}*{1}^2 + ...?",
                   "A geometric sequence starts at {0} with common ratio {1}. Add up its first {2} terms."],
                  lambda a: None),
}


def _antider(c0, c1, c2, c3):
    return " + ".join(t for t in (f"({c3}/4)x^4", f"({c2}/3)x^3", f"({c1}/2)x^2", f"{c0}x")).replace("+ -", "- ")


L8_PLAN = {  # the method, stated before the kernel does the arithmetic (thinking rows)
    "integral": lambda a, p: f"An antiderivative of {p} is {_antider(*a[:4])}. Evaluate it at x = {a[5]} and x = {a[4]} and "
                             f"subtract; scaling by 12 keeps every term an integer, then reduce the fraction.",
    "derivative": lambda a, p: f"f'(x) = {_poly_str([a[1], 2 * a[2], 3 * a[3], 4 * a[4]])}; evaluate it at x = {a[5]}.",
    "cubic": lambda a, p: "The roots are integers, so they divide the constant term; test x downward from 60 until the "
                          "polynomial is 0.",
    "quadratic": lambda a, p: f"Quadratic formula with a = {a[0]}, b = {a[1]}, c = {a[2]}: the larger root is "
                              f"(-b + sqrt(b^2 - 4ac)) / (2a); the discriminant is a perfect square.",
    "linear2": lambda a, p: "Cramer's rule: D = a1*b2 - a2*b1, x = (c1*b2 - c2*b1) / D, y = (a1*c2 - a2*c1) / D.",
    "congruence": lambda a, p: f"{a[0]} is invertible mod {a[2]}: the extended Euclidean algorithm gives its inverse, then "
                               f"x = inverse * {a[1]} mod {a[2]}.",
    "geometric": lambda a, p: f"Sum of n terms = a(r^n - 1)/(r - 1) with a = {a[0]}, r = {a[1]}, n = {a[2]}.",
}


def level8(rng, words):
    fam = rng.choice(sorted(L8))
    args, prog, qs, poly = L8[fam]
    a = args(rng)
    p = poly(a)
    return rng.choice(qs).format(*a, p=p).replace("+ -", "- "), prog(*a), L8_PLAN[fam](a, p).replace("+ -", "- ")


LEVELS.append(level8)
