"""Demo pool: 25 L6 problems per TN_eval family + scaled-up DM families (exact answers in Python, inside the machine's
limits: values <= 20 digits, top operand of * / mod <= 10 digits, / dividend <= 10 digits).
  python training/demo_gen.py POOL_WITH_REFS.json   (writes training/data/demo_pool.json; demo_set.json = the families where
  no machine <= 50% and the one-line machine prompt >= +20 points, 2 samples each: 2026-11-04, 21.1% vs 47.2%)"""
import json, math, random, collections, sys
isprime = lambda n: n > 1 and all(n % k for k in range(2, math.isqrt(n) + 1))  # <= 9 digits: trial division is fine
def nextprime(n):
    n += 1
    while not isprime(n): n += 1
    return n
ROOT = __import__("os").path.dirname(__import__("os").path.abspath(__file__)) + "/../"
def dmbig(R, N=20, easy=False):
    """The scaled-up DM families, N each, from the random stream R (training/stage1_sets.py draws fresh ones). easy: the
    variant knob for the families nobody solves at full size (stage 2a): roots of 9-12 / 10-14 digit numbers, primality of
    5-6 digit numbers (trial division <= ~700 steps)."""
    pool = []
    d = lambda k: R.randrange(10 ** (k - 1), 10 ** k)
    for _ in range(N):  # a (12-13 digits) * b (6-7 digits), product <= 20 digits
        a, b = d(R.choice([12, 13])), d(R.choice([6, 7]))
        a, b = a * R.choice([1, -1]), b
        pool.append({"q": f"What is {a} times {b}?", "a": str(a * b), "family": "DMbig.mul",
                     "forth": f"{a} {b} * ."})
    for _ in range(N):  # 17-19 digit dividend, 7-9 digit divisor
        a, b = d(R.choice([17, 18, 19])), d(R.choice([7, 8, 9]))
        pool.append({"q": f"Calculate the remainder when {a} is divided by {b}.", "a": str(a % b), "family": "DMbig.div_remainder",
                     "forth": f"{a} {b} mod ."})
    for _ in range(N):  # 16-18 digit a, 9-10 digit b with a planted common factor
        g = d(R.choice([4, 5, 6]))
        b = g * d(10 - len(str(g)) - R.choice([0, 1])); a = g * d(18 - len(str(g)) - R.choice([0, 1, 2]))
        pool.append({"q": f"What is the highest common factor of {a} and {b}?", "a": str(math.gcd(a, b)), "family": "DMbig.gcd",
                     "forth": f"{a} {b} gcd ."})
    for _ in range(N):  # two 9-10 digit numbers with a common factor, lcm <= 20 digits
        while True:
            g = d(R.choice([3, 4, 5])); a, b = g * d(R.choice([5, 6])), g * d(R.choice([5, 6]))
            l = a * b // math.gcd(a, b)
            if len(str(a)) <= 10 and len(str(b)) <= 10 and len(str(l)) <= 20: break
        pool.append({"q": f"Calculate the lowest common multiple of {a} and {b}.", "a": str(l), "family": "DMbig.lcm",
                     "forth": f"{a} {b} lcm ."})
    def nearest_root(n, k):
        r = round(n ** (1 / k))
        return min((c for c in range(r - 2, r + 3) if c >= 0), key=lambda c: abs(c ** k - n))  # no ties for k = 2, 3
    for i in range(N):  # square roots of 16-19 digit numbers, cube roots of 15-18 digit numbers
        k = 2 if i % 2 == 0 else 3
        n = d(R.choice([9, 10, 11, 12] if easy else [16, 17, 18, 19]) if k == 2 else
              R.choice([10, 11, 12, 13, 14] if easy else [15, 16, 17, 18]))
        name = {2: "square", 3: "cube"}[k]
        pool.append({"q": f"What is the {name} root of {n} to the nearest integer?", "a": str(nearest_root(n, k)),
                     "family": "DMbig.nearest_integer_root", "forth": None})
    for i in range(N):  # 1e6..3e6 (past ~3e6 trial division runs over the machine's 200k kernel tokens per block): half primes,
        k = 7               # half products of two primes > 300 (no small factor)
        lo, hi, plo, phi, span = (10 ** 4, 5 * 10 ** 5, 50, 700, 5 * 10 ** 5) if easy else (10 ** 6, 3 * 10 ** 6, 300, 1700, 2 * 10 ** 6)
        if i % 2 == 0: n = nextprime(R.randrange(lo, hi))
        else:
            while True:
                p = nextprime(R.randrange(plo, phi)); q = nextprime(lo // p + R.randrange(1, span // p))
                n = p * q
                if lo <= n < hi: break
        word = R.choice(["prime", "composite"])
        ans = isprime(n) if word == "prime" else not isprime(n)
        pool.append({"q": f"Is {n} a {word} number?", "a": str(ans), "family": "DMbig.is_prime", "forth": None})
    return pool


if __name__ == "__main__":
    R = random.Random(7)
    tn = [json.loads(l) for l in open(ROOT + "data/rendered/sp_kernel/TN_eval.jsonl")]
    by = collections.defaultdict(list)
    for r in tn: by[r["family"]].append(r)
    pool = []
    for f, rs in sorted(by.items()):
        for r in R.sample(rs, 25):
            pool.append({"q": r["prompt"], "a": str(r["answer"]), "family": "L6." + f, "forth": r["forth"]})
    pool += dmbig(R)
    json.dump([{k: p[k] for k in ("q", "a", "family")} for p in pool], open(ROOT + "training/data/demo_pool.json", "w"), indent=1)
    json.dump(pool, open(sys.argv[1], "w"), indent=1)  # with reference programs where they exist
    assert all(len(str(abs(int(x)))) <= 20 for p in pool if p['family'].startswith('DMbig') for x in __import__('re').findall(r'-?\d+', p['q'] + ' ' + p['a']))
    print(len(pool), collections.Counter(p["family"] for p in pool))
