"""Plain-text Forth front end (reference for the compiled kernel): the model writes ordinary Forth text inside a
```forth fence; this reads its TEXT TOKENS causally, one at a time, and produces the kernel's existing token protocol
(sp_vm.py), so everything downstream (execution, printing, limits) is unchanged.

Forth's own outer-interpreter rule, at the token level: a word is the characters between whitespace. Each token's
text is appended to the current word; a token that starts with whitespace closes the previous word first, a token
that ends with whitespace closes it after (e.g. '+\\n'). A closed word is classified, case-insensitively:
  number (optional leading '-', digits)           -> <lit> ['-'] digits
  known word (dup, DUP, 1+, 0=, =, mod, ...)       -> its kernel token (= -> '- 0=', the only expansion)
  '(' ... a word ending in ')'                     -> comment, skipped
  ':' NAME / 'variable' NAME                       -> <:> / <var> + the name's tokens
  NAME '@' / NAME '!' (a variable)                 -> <@> / <!> + name           (postfix, as text Forth writes it)
  NAME (a defined word)                            -> <call> + name
The fence: active after the tokens '```' 'forth'; the next '```' token ends it (and closes the last word).
Causality: a word is decided on the token that closes it; postfix '@' / '!' resolve the name on the following word,
which the compiled kernel must hold as pending state (one word of delay, as here).
  translate(ids, tok) -> (kernel source tokens, words seen)   |   python kernel/text_front.py   (self-test)
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sp_vm as V  # noqa: E402

WORDS = {w.lower(): t for w, t in V.WORDS.items()} | {":": V.DEF, ";": V.SEMI, "variable": V.VAR}
FENCE = "```"


def _name_ids(name, tok, top):
    ids = tok(" " + name.lower(), add_special_tokens=False).input_ids  # names are case-insensitive, lowercase inside
    return ids[:1] if top else ids  # a top-level reference resolves on its first token (sp_vm convention)


def translate(ids, tok):
    """Text token ids of a whole turn (prose + fences) -> kernel source tokens for the fenced Forth only."""
    out, words = [], []
    state = {"fence": False, "after_fence": False, "word": "", "comment": False, "pending": None, "depth": 0,
             "expect": None, "vars": set(), "defs": set()}

    def emit_name_ref(name):
        top = state["depth"] == 0
        out.extend([V.CALL] + _name_ids(name, tok, top))

    def close():
        w, state["word"] = state["word"], ""
        if not w:
            return
        words.append(w)
        lw = w.lower()
        if state["comment"]:
            state["comment"] = not w.endswith(")")
            return
        if w == "(":
            state["comment"] = True
            return
        pend = state["pending"]
        if pend is not None and lw not in ("@", "!"):  # a variable name not followed by @ / !: treat as a call
            state["pending"] = None
            emit_name_ref(pend)
        if state["expect"]:  # the name after ':' or 'variable'
            kind, state["expect"] = state["expect"], None
            out.extend([V.DEF if kind == "def" else V.VAR] + tok(" " + lw, add_special_tokens=False).input_ids)
            state["defs" if kind == "def" else "vars"].add(lw)
            state["depth"] += kind == "def"
            return
        if re.fullmatch(r"-?\d+", w):
            out.extend([V.LIT] + [V.MINUS] * w.startswith("-") + [V.DIGITS[int(c)] for c in w.lstrip("-")])
        elif lw in ("@", "!") and pend is not None:
            state["pending"] = None
            out.extend([V.FETCHM if lw == "@" else V.STOREM] + _name_ids(pend, tok, state["depth"] == 0))
        elif lw == "=":
            out.extend([V.WORDS["-"], V.WORDS["0="]])
        elif lw == ":":
            state["expect"] = "def"
        elif lw == "variable":
            state["expect"] = "var"
        elif lw == ";":
            out.append(V.SEMI)
            state["depth"] = 0
        elif lw in WORDS:
            out.append(WORDS[lw])
        elif lw in state["vars"]:
            state["pending"] = w  # decided by the next word: @ or !
        else:
            emit_name_ref(w)  # a defined word (or a forward / unknown name: the kernel reports it)

    for t in ids:
        s = tok.decode([t])
        if not state["fence"]:
            if s.strip() == FENCE:
                state["after_fence"] = True
            elif state["after_fence"] and s.strip().lower() == "forth":
                state["fence"], state["after_fence"] = True, False
            else:
                state["after_fence"] = False
            continue
        if s.strip() == FENCE:  # end of the block
            close()
            if state["pending"] is not None:
                emit_name_ref(state["pending"])
                state["pending"] = None
            state["fence"] = False
            continue
        if s[:1].isspace():
            close()
        state["word"] += s.strip()
        if s[-1:].isspace() and s.strip():
            close()
    return out, words


def run_text(program, tok):
    """Fence a standard Forth program, tokenize it as the model would write it, translate, and run it on the kernel's
    reference machine (forth/sp.py's stream: every kernel limit). -> printed line (or raises)."""
    from forth import sp
    ids = tok("```forth\n" + program.strip() + "\n```", add_special_tokens=False).input_ids
    user, _ = translate(ids, tok)
    stream, owned = sp.stream(user)
    return sp.printed(stream, owned)


CASES = [("3 4 + .", "7"), ("3 4 + .\n", "7"), ("10 1+ .", "11"), ("5 0= . 0 0= .", "0 -1"), ("-17 3 * .", "-51"),
         ("7 3 - .", "4"), ("( a comment ) 2 3 * .", "6"), (": sq ( n -- n*n ) DUP * ;\n12 sq .", "144"),
         (": sq dup * ;\n12 sq .", "144"), ("VARIABLE t 5 t ! t @ 2 * .", "10"), ("variable t\n5 t !\nt @ .", "5"),
         ("4 4 = . 4 5 = .", "-1 0"), ("17 NEGATE .", "-17"), ("20 3 MOD .", "2"),
         (": step dup * ;\n7 STEP .", "49"), ("VARIABLE Acc 3 acc ! ACC @ .", "3")]  # (program, printed)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from forth import sp
    tok = sp.tokenizer()
    bad = 0
    for prog, want in CASES:
        try:
            got = run_text(prog, tok)
        except Exception as e:  # noqa: BLE001
            got = f"{type(e).__name__}: {e}"
        bad += got != want
        print(("ok  " if got == want else "FAIL"), repr(prog), "->", repr(got), "" if got == want else f"(want {want})")
    print("self-test:", len(CASES) - bad, "/", len(CASES))
