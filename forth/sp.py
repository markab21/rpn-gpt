"""Standard Forth (ref.py dialect) -> stored-program kernel tokens (kernel/sp_vm.py), with the kernel's limits.

sp_ready(src): source rewrites that keep it standard Forth (ref.py runs it) but inside the kernel's word set: names
lowercased and spelled as text tokens (digits -> words, '-' -> '_': digits and '-' end a name), renamed when their
first token's key is taken (the kernel keys a name by its first token), nested IF / BEGIN factored innermost-first
into helper definitions.
tokens(src, tok): 1:1 to kernel tokens. stream(ids): the true interleaving (the kernel speaks right after any user
token it has something to say to). problems(src, output, tok): every limit below, then printed output == output.

Limits (sp_vm.py / sp_kernel.py): numbers <= 20 digits; * MOD: top operand <= 10 digits, product <= 20 digits; /:
both operands <= 10 digits; * / MOD: two free entries above the top, SWAP one; 1024 data / return entries; <= 192 code
cells; <= 8 variables; names are text tokens with distinct first-token keys; IF / BEGIN only in definitions, never
nested; PICK's index is a one-digit literal right before it; <= 63 user literals in a row (the depth counter lo < 256
is re-based only by kernel-emitted carries). A top-level <call> / <@> / <!> is written with the name's first token
only: the kernel takes over right after it.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kernel"))
import sp_vm as V  # noqa: E402

TOKENIZER = ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", "a9904d24bcc1d289a1950fa9d2b978c47cf903b9")
BUILTIN = set(V.WORDS) | {":", ";", "VARIABLE", "@", "!"}
OPEN, CLOSE = {"IF": "THEN", "BEGIN": "REPEAT"}, {"THEN", "REPEAT"}
DIGIT_WORDS = "zero one two three four five six seven eight nine".split()
MAX_DIGITS, MAX_MUL_TOP, MAX_DIV, MAX_LITS, MAX_TOKENS = V.ND, V.ND // 2, V.ND // 2, 63, 20480  # (SP_ND: wide)
PREFIXES = ("my_", "do_", "the_", "get_", "run_", "new_", "aux_", "calc_")  # renames for a taken first-token key


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])


def is_num(w):
    return re.fullmatch(r"-?\d+", w) is not None


# Top-level names with no one-token spelling (the kernel resolves a top-level reference on its first name token).
RENAME = {"D20": "twenty", "ROLLS-TO-SIX": "rolls", "FLIPS-TO-TAILS": "tails", "FARTHEST": "far", "FIVE-STEPS": "steps",
          "ADD-ROLL": "add", "MAX2": "max", "MIN3": "smallest", "MAX3": "largest", "ODD?": "odd", "LEAP?": "leap",
          "BIGGER?": "bigger", "BIG?": "big", "CLIP0": "clip", "COUNTDOWN": "down", "SUMEVENS": "even",
          "DIVISORS": "divisor", "PRIME?": "prime", "NTHPRIME": "primes", "FIBS": "fib", "SUMTO": "sum", "TUCK": "under",
          "REVSUB": "sub", "MAXDUP": "larger", "GCD": "common", "ISQRT": "root", "MODPOWLOOP": "power", "SUMLOOP": "sum",
          "FACTLOOP": "factorial", "S1": "left", "S2": "middle", "S3": "right", "SHOW-POINTS": "show", "FAR": "track",
          "DSUM": "sum", "DIGSUM": "sum", "NDIGITS": "digits", "AVG3": "average", "AVG4": "average", "STOPCOUNT": "count",
          "HALVINGS": "halve", "MAXDIG": "largest", "REVDIG": "reverse", "WALK-FAR": "wander", "BASEV": "base",
          "MODULUS": "modulus"}


def spell(name, tok):
    """Lowercase name as text tokens."""
    return RENAME.get(name.upper()) or re.sub(r"\d", lambda d: DIGIT_WORDS[int(d.group())], name.lower().replace("-", "_"))


def first_key(s, tok):
    return V.key(tok(" " + s, add_special_tokens=False).input_ids)


def unique(s, taken, tok):
    """s, or the first PREFIXES rename whose first-token key is free; records the key in taken."""
    for c in (s,) + tuple(p + s for p in PREFIXES):
        if taken.setdefault(first_key(c, tok), c) == c:
            return c
    raise ValueError(f"no free first-token key for {s!r}")


def factor(name, body, defs, fresh=lambda h: h):
    """Pull nested control structures out of a definition body into helper words (innermost first)."""
    while True:
        stack, cut = [], None
        for i, w in enumerate(body):
            if w in OPEN:
                stack.append(i)
            elif w in CLOSE:
                j = stack.pop()
                if stack:  # nested in an outer structure: factor this one out
                    cut = (j, i)
                    break
        if cut is None:
            return body
        j, i = cut
        if "EXIT" in body[j:i + 1]:
            raise ValueError("nested control with EXIT inside")
        helper = fresh(f"{name}_{'abcdefghijklmnopqrstuvwxyz'[len(defs)]}")
        defs.append((helper, body[j:i + 1]))
        body = body[:j] + [helper] + body[i + 1:]


def sp_ready(src, tok, init=False):
    """init: store 0 into every new variable (ref.py zero-fills; the kernel's cells start undefined)."""
    words = re.sub(r"\( [^)]*\)", " ", src).split()
    upper = [w.upper() for w in words]
    names, taken, final = {upper[i + 1] for i, w in enumerate(upper[:-1]) if w in (":", "VARIABLE")}, {}, {}
    new = lambda w: final.setdefault(w, unique(spell(w, tok), taken, tok)) if w not in final else final[w]
    word = lambda w: new(w) if w in names else w
    lines, line, i = [], [], 0
    while i < len(words):
        if upper[i] == ":":
            end = upper.index(";", i)
            name, helpers = new(upper[i + 1]), []
            body = factor(name, [word(x) for x in upper[i + 2:end]], helpers, lambda h: unique(h, taken, tok))
            lines += [" ".join(line)] * bool(line) + [f": {h} {' '.join(b)} ;" for h, b in helpers]
            lines.append(f": {name} {' '.join(body)} ;")
            line, i = [], end + 1
            continue
        if upper[i] == "VARIABLE":
            v = new(upper[i + 1])
            line += ["VARIABLE", v] + [f"0 {v} !"] * init
            i += 2
            continue
        line.append(word(upper[i]))
        i += 1
    return "\n".join(lines + [" ".join(line)] * bool(line))


def tokens(src, tok):
    """sp_ready source -> kernel source tokens; names are tokenized as ' name' (case kept). -> (ids, name info)."""
    words = re.sub(r"\( [^)]*\)", " ", src).split()
    name = lambda w: tok(" " + w, add_special_tokens=False).input_ids
    out, info, depth, i = [], [], 0, 0  # info: (name, ids, kind, top level?)
    while i < len(words):
        w, nx = words[i], words[i + 1] if i + 1 < len(words) else None
        u = w.upper()
        if u in (":", "VARIABLE"):
            out += [V.DEF if u == ":" else V.VAR] + name(nx)
            info.append((nx, name(nx), "def" if u == ":" else "var", True))
            depth += u == ":"
            i += 2
            continue
        if u == ";":
            out.append(V.SEMI)
            depth -= 1
        elif is_num(w):
            out += [V.LIT] + [V.MINUS] * w.startswith("-") + [V.DIGITS[int(c)] for c in w.lstrip("-")]
        elif u in V.WORDS:
            out.append(V.WORDS[u])
        elif u in BUILTIN:
            raise ValueError(f"stray {w}")
        elif nx in ("@", "!"):
            out += [V.FETCHM if nx == "@" else V.STOREM] + name(w)[:1 if depth == 0 else None]
            info.append((w, name(w), "ref", depth == 0))
            i += 2
            continue
        else:
            out += [V.CALL] + name(w)[:1 if depth == 0 else None]  # top level: the kernel takes over after one
            info.append((w, name(w), "ref", depth == 0))
        i += 1
    return out, info


def source_problems(src, info):
    words = re.sub(r"\( [^)]*\)", " ", src).split()
    upper = [w.upper() for w in words]
    unknown = {w for w in upper if not is_num(w)} - BUILTIN - {n.upper() for n, _, kind, _ in info if kind != "ref"}
    if unknown:
        return f"words not in the kernel {sorted(unknown)}"
    keys, defined = {}, [n.upper() for n, _, kind, _ in info if kind != "ref"]
    if len(set(defined)) < len(defined):
        return "a name is defined twice"
    for n, ids, kind, top in info:
        if not ids or not all(V.is_text(t) for t in ids):
            return f"name {n!r} is not text tokens"
        k = V.key(ids)
        if keys.setdefault(k, n.upper()) != n.upper():
            return f"name key collision {keys[k]!r} / {n!r}"
    if sum(kind == "var" for _, _, kind, _ in info) > V.NCELL:
        return f"more than {V.NCELL} variables"
    open_, depth = None, 0
    for i, w in enumerate(upper):
        depth += (w == ":") - (w == ";")
        if w in OPEN or w in CLOSE or w in ("ELSE", "WHILE", "EXIT"):
            if not depth:
                return f"{w} outside a definition"
            if w in OPEN:
                if open_:
                    return "nested control"
                open_ = w
            elif w in CLOSE:
                open_ = None
        if w == "PICK" and not (i and re.fullmatch(r"\d", upper[i - 1])):
            return "PICK index is not a one-digit literal"
        if is_num(w) and len(w.lstrip("-")) > MAX_DIGITS:
            return f"literal over {MAX_DIGITS} digits"
    return None


class Checked(V.Machine):
    """sp_vm.Machine plus the value limits the kernel's rows impose (sp_vm itself uses Python ints)."""

    def push(self, v):
        assert len(str(abs(v))) <= MAX_DIGITS, f"value over {MAX_DIGITS} digits"
        super().push(v)

    def execute(self, t, step):
        if t in (V.MUL, V.MOD, V.DIV):
            x, y = self.stack[-2], self.stack[-1]
            assert len(str(abs(y))) <= MAX_MUL_TOP, f"* / MOD top operand over {MAX_MUL_TOP} digits"
            assert t == V.MUL or y, "/ or MOD by zero"
            assert t != V.MUL or len(str(abs(x * y))) <= MAX_DIGITS, f"product over {MAX_DIGITS} digits"
            assert t != V.DIV or len(str(abs(x))) <= MAX_DIV, f"/ dividend over {MAX_DIV} digits"
            assert len(self.stack) <= V.NPOS - 2, "* / MOD without two free entries"
        if t == V.SWAP:
            assert len(self.stack) <= V.NPOS - 1, "SWAP without a free entry"
        super().execute(t, step)


def stream(ids):
    """User tokens one at a time; after each, the kernel runs until it hands back. -> (all ids, kernel-owned offsets)."""
    m, out, owned, lits = Checked(), [], [], 0
    for t in ids:
        lits += t == V.LIT and not m.compile
        assert lits <= MAX_LITS, f"over {MAX_LITS} user literals between kernel turns"
        m.feed(t)
        out.append(t)
        while (k := m.nxt()) is not None:
            lits = 0
            assert k != V.OVF, "stack overflow"
            owned.append(len(out))
            out.append(k)
            m.feed(k, kernel=True)
            assert len(out) <= MAX_TOKENS, f"over {MAX_TOKENS} tokens"
    assert len(m.code) <= V.NADDR, f"over {V.NADDR} code cells"
    assert not m.compile, "unterminated definition"
    return out, owned


def printed(ids, owned):
    s = "".join("-" if ids[o] == V.MINUS else " " if ids[o] == V.SPACE else str(V.DIGITS.index(ids[o]))
                for o in owned if ids[o] in V.DIGITS + [V.MINUS, V.SPACE])
    return s.strip()


def problems(src, output, tok):
    """None, or why src (sp_ready) does not run on the kernel and print exactly `output`. -> (problem, ids, owned)."""
    try:
        user, info = tokens(src, tok)
    except ValueError as e:
        return str(e), None, None
    if p := source_problems(src, info):
        return p, None, None
    try:
        ids, owned = stream(user)
    except (AssertionError, KeyError, ValueError, IndexError) as e:
        return f"kernel: {type(e).__name__} {e}", None, None
    got = printed(ids, owned)
    return (None if got == output else f"printed {got!r} != {output!r}"), ids, owned


def verified(r, src, tok):
    """-> (sp_ready source that ref.py and the kernel both run to r['output'], None) or (None, why not)."""
    from .check_programs import problems as ref_problems
    p = None
    for init in (False, True):
        try:
            s = sp_ready(src, tok, init)
        except ValueError as e:
            return None, str(e)
        if not (p := ref_problems({**r, "forth": s}) or problems(s, r["output"], tok)[0]):
            return s, None
    return None, p


def main():
    """python -m forth.sp [--out DIR] data/programs/{math,logic,games}.jsonl: pass counts as written / after the
    automatic rewrites / after data/programs/sp_rewrites.jsonl; --out writes the verified records (sp-ready Forth)."""
    import argparse
    import collections
    import json
    a = argparse.ArgumentParser(description=main.__doc__)
    a.add_argument("files", nargs="+")
    a.add_argument("--out")
    a.add_argument("-v", action="store_true")
    args = a.parse_args()
    tok = tokenizer()
    rewrites = {r["prompt"]: r["forth"] for r in map(json.loads, open(Path(args.files[0]).parent / "sp_rewrites.jsonl"))}
    for path in args.files:
        rows = [json.loads(line) for line in open(path) if line.strip()]
        kept = []
        for label in ("as written", "auto", "rewritten"):
            why = collections.Counter()
            for r in rows:
                if label == "as written":
                    p = problems(r["forth"], r["output"], tok)[0]
                else:
                    s, p = verified(r, rewrites.get(r["prompt"], r["forth"]) if label == "rewritten" else r["forth"], tok)
                    if s and label == "rewritten":
                        kept.append({**r, "forth": s})
                if p and args.v:
                    print("  ", label, "|", p, "|", r["prompt"][:60])
                why[re.sub(r"'.*'|\[.*\]|\d+", "_", p) if p else "ok"] += 1
            print(f"{path} {label}: {why.pop('ok', 0)}/{len(rows)} pass; {dict(why.most_common())}")
        if args.out:
            Path(args.out).mkdir(parents=True, exist_ok=True)
            (Path(args.out) / Path(path).name).write_text("".join(json.dumps(r) + "\n" for r in kept))


if __name__ == "__main__":
    main()
