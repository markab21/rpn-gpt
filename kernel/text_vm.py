"""Text-mode reference machine: the compiled kernel reading plain Forth text with no lookahead and no inserted model
tokens. The model writes only text tokens; the kernel watches them causally and, at the position where the NEXT model
token closes a word, emits that word's protocol translation (sp_vm.py) as its own hidden tokens. Those run exactly as
protocol source does (sp_vm's replay / spill / clear / print tokens follow as now).

A word closes at a model token that starts with whitespace, at the token after one that ended with whitespace, or at
the closing ``` (at most one word per position). The fence opens at the token after 'forth' (any case) right after a
``` token, if that token starts with a newline (as training/loop_sim.py runs only "```forth\n" blocks; prose such as "a
```forth block" opens nothing), and closes at the next ``` token. Only model tokens feed the word accumulator, never the kernel's own. Emitted at the
closing position (names are one <nm> token, keyed by name_key(name, version)):
  number (-?[0-9]+)  -> <lit> [<m->] machine digits, normalized ("-0", "007" are written 0, 7)
  built-in word      -> its op token(s): 1+ 1- 0= are single ops; = -> <-> <0=>; and / or (on flags: -1 / 0) ->
                        0= 0= swap 0= 0= * negate / 0= swap 0= * 0=; cr -> <cr> (a newline in the output block)
  : NAME ... ;       -> <:> <nm> ... <;>    variable NAME -> <var> <nm>    N value|constant NAME -> <var> <nm> <!> <nm>
  to NAME (a value)  -> <!> <nm>            value / constant NAME -> <@> <nm>      defined NAME -> <call> <nm>
  variable NAME      -> decided by the next word: @ -> <@> <nm>, ! -> <!> <nm>, anything else -> <@> <nm> (a fetch)
  2dup nip tuck not 2* 2/ abs rot -rot min max /mod */: built from the ops above (rot / -rot / min / max through hidden
                        scratch variables); space: a blank in the output block; emit: an error
  gcd lcm: hidden words like ?dup; depth: one op (752). Not in the dialect (2026-11-05, kernel parity: the kernel
                        never had them; used in <= 2.5% of machine rollouts, mostly wrong): .s spaces >r r> r@ are
                        unknown words, as in the kernel
  <= >= <> 0<> 0> 0< 2drop invert true false: short op sequences; N x +! adds to a variable; until / again close a
                        begin (begin .. f until, begin .. again); ?do = do; ?dup: a hidden word, defined once
                        at first use and called; ! @ +! without a variable, ." and other strings: errors that say why
  ( ... ), \\ ... end of line -> nothing
  closing ```        -> the last word's tokens, then the output block (visible): ['\\n'] '```' 'output' '\\n', the
                        printed line, '\\n' '```\\n'; then the model's turn again
Names have one binding each (no versions, as the kernel's dictionary has one entry per key): redefining a name rebinds
it everywhere, including in code compiled earlier (real programs never relied on the old binding). Inside ': foo ... ;' the name foo always means the word
being defined (recursion; recurse is not in the dialect: an unknown word, as in the kernel).
Control (if else then, begin while repeat until again, do loop +loop with i, leave, exit): compiled in place with a
compile-time control stack (the kernel's: per-level registers), at most 4 structures open at once (else 'structures
nested too deep'). Every loop starts with an entry <while> (begin: -1 <begin> <while>; do: its i < limit flag <dob>
<while>); leave and a begin loop's while (when false) push 0 and jump back to it (<lv0>, its target the innermost loop's
<begin>), so no forward jump needs a list of patches: until = 0= <repeat>, again / repeat = -1 <repeat>, while =
0= if 0 <lv0> then. (Nesting in place replaced hoisting nested structures into hidden helper words; replay vs d8d8d42:
17 of 11410 blocks change, all already erroring, 0 answers.)
do: 'limit start do' keeps i / limit in two hidden cells per lexical loop depth (i = start .. limit - 1 for loop, which
runs once if start > limit; +loop n stops when i crosses the limit, either direction; start = limit runs nothing); the
continue flag stays on the stack for the loop's <while>. Hidden cells (loop cells, the scratch cells of rot -rot min
max /mod */) have their own tokens <@h> <!h> and no names. At top level, the first if / begin / do opens a hidden
definition (its name: <nmtop>) that collects the following top-level words; it is compiled and called at the closing
``` or before the next : / variable / value / constant, so top-level code still runs in order. gcd lcm ?dup: hidden
words (names <nmgcd> <nmlcm> <nm?dup>), defined at first use (top level: there; in a definition: right after it).
Piece 2c spec change (hidden streams; loops were all hoisted helpers with a third 'go' cell, named hidden variables):
visible outputs equal the previous spec's on every frontier / demo turn but 24 of 7961 frontier blocks: 22 where the
reference-only MAXRUN guard fires at another iteration (inline loops emit fewer tokens; same first error line), 2 that
ran out of code memory and now run (an inline do loop takes 6 fewer cells); replay: 462 changed (was 454), 1880 answers.
With the one-line error rule below: replay 3372 changed (every error line's text), 1880 answers.
A print ('.') runs as today but its text is hidden (<m0>..<m9>, <m->, <psp> for the space, <pnl> for cr); the block
shows it. An error stops the machine for the rest of the block (a hidden <err..> right after the failing token); the
block shows its prints, then two fixed lines: 'error: <why>' (ERRS: one fixed text per error class, no names) and what to
do (RETRY). At the next fence the kernel resets: a replay that was running ends (<done>), a half-written definition gets
its ; (it stays defined with what it had), the stack is dropped (<drop>s); code, names, variables and return frames are
kept. (Piece 2d-minimal, the user's decision: one fixed report everywhere; the old detail lines were dropped.)
No re-send rollback (dropped 2026-11-05 for kernel parity, the user's rule): a block that re-declares an earlier
block's names rebinds them in place (below) and its code takes new cells, as in the kernel. R fired in 42-44% of
machine rollouts but changed visible output in 3.4% (s2a demo4) / 8.7% (stage-A a00), always 'code memory full', mostly
in already-failing rollouts (4 / 11 and 11 / 93 correct); R_DESIGN.md keeps the design if code memory becomes a limit.
Rebind (hidden only): a name the kernel's dictionary already holds, declared again, is preceded by <rbh_hi> <rbl_kind_lo>
(its old value and kind: the kernel cancels the add-only entry); a constant binds at a hidden <cst> before its <!>.
Item 2 (kernel limits as errors; replay vs 48b3e3e over frontier_v22b + demo_oneline + demo4_card, 11410 blocks: 120
outputs change, 0 answers): the token cap is 196608 machine tokens per block (12 * 128 * 128: the kernel counts them
with hidden <tick> / <tick2>, emitted where a carry could be; the error comes at the next replayed cell) with its own
text (87 blocks); a data / return stack overflow is 'stack overflow (over 1024 values)' / 'too many nested calls (over
1024)' and both stacks start each block empty (30 blocks; the old text said 'kernel limit' or, by a pattern bug, 'a
structure word without its opener'); * errors when digits(x) + digits(y) > 20 (known at the op; 1 block); cr and space
count as prints (each takes a print-memory row; 2 blocks); + - 1+ 1- over 20 digits and the capacity checks fail before
the op changes anything; code memory full fires one cell early (the ; always fits).
Declarations at the limits (2026-11-05, kernel parity; replay vs 4b1f650 over demo4 sets + stage-A a00, 7857 blocks: 88
outputs change, 0 answers; 12k eval: 2 of 1792): a ':' name with no code cell free (len(code) >= NADDR - 1) and a
variable / value / constant name with all 32 cells taken are errors at the name, before any token ('code memory
full' via <err code2>, which undoes nothing; 'over 32 variables'), so the old binding stays and no name is bound past
the code end (a re-declared one crashed rebind's RBH range: a bogus 'stack underflow').
Dead top-level code is freed (a1, 2026-11-05): a hidden top-level definition runs once and is never called again. When
its ; was the last code compiled (no definition, shared word or other top-level code after it) and no error came
after it, the next block's first token frees code memory from cut = 16 * ceil(its first cell / 16) (the kernel wipes
whole 16-cell address heads), when cut < len(code): a hidden <free_h> (h = cut / 16), code truncated there (freeat).
Replay vs 62b5eff over the card A/B rollouts (devcf v3 / v3resend, 703 turns) + 4 frozen-test sets (1642 turns,
measured only): 310 of 11790 block outputs change, all in turns that filled code memory (307 were 'code memory
full'; 87 of 269 such turns never fill now), 0 answers lost, 114 gained; identical to code_measure.py a1 on every set.
Names (2026-11-05, kernel parity): recurse is not in the dialect (an unknown word: a word calls itself by its own name);
a control word (if else then begin while repeat until again do ?do loop +loop i exit leave) declared with : variable
value constant is 'a name collides' at the name (the kernel reads them as control words everywhere); a hidden shared
word (gcd lcm ?dup) first used in a definition that fails before its ; is not defined (the next use defines it, as
the kernel's DECL). Replay vs 17eb681 (card A/B + frozen x4 + dev_round x2 + demo4, 3218 turns, 15231 blocks): 217
outputs change, 0 answers lost, 4 gained.
Output block edges (as training/loop_sim.py's rollouts read): the block ends with '```\n' as one token (2640), and starts
with '\n' only after a bare closing '```' (a closing '```\n' already ended the line): the visible text is always
'```\n```output\n<line>\n```\n' (replay: no block output changes; only these edge newlines).
  stream(ids, tok) -> (all ids, kernel-owned offsets, <nm> keys)   |   python kernel/text_vm.py [progs.json]  (self-check)
"""

import copy
import itertools
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forth import sp  # noqa: E402

V = sp.V
FENCE = "```"
THINK_END = "</think>"  # one special token: the end of the thinking is a hard boundary for blocks
NL, TICKS, OUTPUT, TICK1, TICKNL = 1010, 1975, 10114, 1096, 2640  # Lightning's '\n' '```' 'output' '`' '```\n' (main)
MAXPRINT, MAXCELL = 128, V.NCELL + V.NXCELL  # prints per block (the kernel's print memory, cr / space too), variables
PRINTS = (V.PR, V.CR, V.SPACEOP)  # (incl. hidden)
CAPT = 12  # machine tokens per block: an error at the first replayed cell once 12 <tick2>s (12 * 128 * 128 = 196608)
WORDS = {w.lower(): [t] for w, t in V.WORDS.items()} | {
    "=": [V.SUB, V.ZEQ], "and": [V.ZEQ, V.ZEQ, V.SWAP, V.ZEQ, V.ZEQ, V.MUL, V.NEG],
    "or": [V.ZEQ, V.SWAP, V.ZEQ, V.MUL, V.ZEQ], "cr": [V.CR], "space": [V.SPACEOP], "2dup": [V.OVER, V.OVER],
    "nip": [V.SWAP, V.DROP], "tuck": [V.SWAP, V.OVER], "not": [V.ZEQ], "2*": [V.DUP, V.ADD],
    "2/": [V.LIT, V.MD[2], V.DIV], "abs": [V.DUP, V.DUP, V.LIT, V.MD[0], V.LT, V.MUL, V.DUP, V.ADD, V.ADD],
    "<=": [V.GT, V.ZEQ], ">=": [V.LT, V.ZEQ], "<>": [V.SUB, V.ZEQ, V.ZEQ], "0<>": [V.ZEQ, V.ZEQ],
    "0>": [V.LIT, V.MD[0], V.GT], "0<": [V.LIT, V.MD[0], V.LT], "2drop": [V.DROP, V.DROP], "invert": [V.NEG, V.DECR],
    "true": [V.LIT, V.MMINUS, V.MD[1]], "false": [V.LIT, V.MD[0]],
    "until": [V.ZEQ, V.WHILE, V.REPEAT], "again": [V.LIT, V.MMINUS, V.MD[1], V.WHILE, V.REPEAT],  # close a begin
    "depth": [V.DEPTH]}
GCD = [V.BEGIN, V.DUP, V.WHILE, V.SWAP, V.OVER, V.MOD, V.REPEAT, V.DROP]  # begin dup while tuck mod repeat drop
SHARED = {"?dup": [V.DUP, V.IF, V.DUP, V.THEN], "gcd": GCD + WORDS["abs"],
          "lcm": [V.OVER, V.OVER] + GCD + [V.DIV, V.MUL] + WORDS["abs"],  # 2dup gcd / * (divide first: smaller values)
          }  # hidden words, defined once at first use and called
SCRATCH = {"rot": "bca", "-rot": "cab", "/mod": None, "*/": None}  # via hidden scratch cells: pop c b a into t1 t2 t3, push in this order
MAXNEST = 4  # structures open at once (the kernel's control stack)
LOOPC = {1: (32, 33), 2: (37, 38), 3: (39, 40)}  # do loop at loop depth d: hidden cells i, limit (3 deep: the kernel's
# code memory holds an op code per hidden-cell token; 0 real blocks nest do loops 4 deep)
SCR = (34, 35, 36)  # scratch t1 t2 t3 (rot -rot min max /mod */)
F, S = V.fetch_tok, V.store_tok  # a hidden cell's fetch / store token


OPENERS, CLOSERS = ("if", "begin", "do"), {"else": ("if",), "then": ("if", "else"), "while": ("begin",),
                                           "repeat": ("while",), "until": ("begin",), "again": ("begin",),
                                           "loop": ("do",), "+loop": ("do",)}
ALIAS = {"?do": "do"}  # do already runs nothing when start = limit
TRUEF = [V.LIT, V.MMINUS, V.MD[1]]
LOOPS = {"begin": TRUEF + [V.BEGIN, V.WHILE], "until": [V.ZEQ, V.REPEAT], "again": TRUEF + [V.REPEAT],  # every loop:
         "repeat": TRUEF + [V.REPEAT], "while": [V.ZEQ, V.IF, V.LIT, V.MD[0], V.LEAVES[0], V.THEN],  # an entry <while>
         "leave": [V.LIT, V.MD[0], V.LEAVES[0]]}  # (leave / while's false: a 0 flag, back to that <while>)
EMPTY = "(nothing printed: run a word and print its result with . in a ```forth block)"  # the only empty-output note:
# the kernel has no per-word "this block only defined NAME" note (dropped 2026-11-05 for parity; 3.7-10.4% of rollouts saw it)
ERRS = {  # every error is one fixed line 'error: <why>' (the kernel's ROM text); class -> why
    "underflow": "stack underflow", "zero": "/ or MOD by zero", "unknown": "unknown word",
    "strings": "strings and emit are not supported: the machine prints numbers only (. prints one; cr, space for layout)",
    "semi": "; outside a definition", "semiopen": "; with a structure still open",
    "noopener": "a structure word without its opener", "unterm": "unterminated definition",
    "unstruct": "unterminated structure", "iout": "i outside a do loop",
    "defin": "a definition or variable inside a definition",
    "varword": "@ ! +! need a variable name right before them (variable x  5 x !  x @ .  2 x +!)",
    "toval": "to needs the name of a value or variable", "recurse": "recurse outside a definition",
    "leave": "leave outside a loop", "deep": "structures nested too deep",
    "twowords": "a token holds two words", "collide": "a name collides with another name: rename it",
    "digits": "a number over 20 digits", "mulop": "* / MOD top operand over 10 digits",
    "divop": "/ dividend over 10 digits", "vars": "over 32 variables", "code": "code memory full",
    "prints": "over 128 prints in one block", "tokens": "the block ran too long (over 196608 kernel steps)",
    "lits": "over 63 numbers in a row", "pick": "PICK needs its index as the number just before it",
    "runtime": "undefined name at run time", "limit": "kernel limit", "full": "stack overflow (over 1024 values)",
    "rfull": "too many nested calls (over 1024)"}
ETOK = {"digits2": "digits", "full2": "full", "prints1": "prints", "code2": "code"}  # error tokens that undo another
# depth change (code2: none, at a ':' name): one line
AGAIN = "fix the program and send the corrected block again"
RETRY = {  # the line after each error's line (what to do); the kernel's ROM text with ERRS (edit both here, rebuild)
    "underflow": "a word needed more values than the stack held: check what each word takes and leaves; " + AGAIN,
    "zero": "the divisor was 0: check the values that reach / or mod; " + AGAIN,
    "unknown": "a word is not defined: check its spelling, or define it with : or declare it with variable; " + AGAIN,
    "full": "a loop probably leaves values on the stack each pass: check what each pass pushes and pops; " + AGAIN,
    "rfull": "a word probably calls itself without stopping; " + AGAIN,
    "tokens": "an endless loop or too many iterations: check each loop's exit condition; " + AGAIN}
CLASSES = [  # (message pattern, class) for the reference's internal messages, first match
    (r"stack underflow", "underflow"), (r"/ or MOD by zero", "zero"), (r"unknown word", "unknown"),
    (r".* is not supported", "strings"), (r"; outside a definition", "semi"), (r"; with .* still open", "semiopen"),
    (r"recurse outside", "recurse"), (r"i outside a do loop", "iout"), (r"leave outside", "leave"),
    (r".* (without|outside a definition)", "noopener"), (r"unterminated definition", "unterm"),
    (r"unterminated", "unstruct"), (r".* inside a definition", "defin"), (r".* needs a variable name", "varword"),
    (r"to .*: not a value", "toval"), (r"data stack over", "full"), (r"return stack over 1024", "rfull"),
    (r".*nested over", "deep"), (r"token .* holds two words", "twowords"), (r"name key collision", "collide"),
    (r"(value|product|literal) over", "digits"), (r"\* / MOD top operand", "mulop"), (r"/ dividend", "divop"),
    (r"over \d+ variables", "vars"), (r"code memory full \(no free cell", "code2"), (r"code memory full", "code"),
    (r"over \d+ prints", "prints"),
    (r"the block ran too long", "tokens"), (r"over \d+ literals", "lits"), (r"PICK needs", "pick")]


def err_text(c):
    """An error class's two fixed lines (after 'error: ')."""
    c = ETOK.get(c, c)
    return ERRS[c] + "\n" + RETRY.get(c, AGAIN)


def err_class(e):
    """An exception the reference raised -> its error class (ERRS)."""
    if isinstance(e, IndexError):
        return "underflow"
    if isinstance(e, KeyError):
        return "runtime"
    return next((c for pat, c in CLASSES if re.match(pat, str(e))), "limit")


P1, P2 = 127, 113  # name key space: (sum of pair code A mod P1, sum of pair code B mod P2)


def pair_code(p, q):
    h = ((ord(p) << 16 | ord(q)) * 2654435761 + 0x9E3779B9) & 0xFFFFFFFF
    h = ((h ^ (h >> 15)) * 2246822519) & 0xFFFFFFFF
    return (h >> 4) % P1, (h >> 18) % P2


def name_key(name, version=0):
    """A name's key, from its lowercase characters only: the sum of a fixed code pair per character pair of '^' + name
    (each token adds its own pairs plus the one across its left edge, so any split into tokens gives the same key),
    shifted per version (redefinitions)."""
    s, a, b = "^" + name.lower(), 37 * version, 53 * version
    for p, q in itertools.pairwise(s):
        x, y = pair_code(p, q)
        a, b = a + x, b + y
    return a % P1, b % P2


def standin(k):
    """The inner sp_vm machine keys a name by a stand-in text id (>= 2^24, never a real token; sp_vm.KEYOF maps it to
    the key), so execution is sp_vm's unchanged. In the stream the token is NM; its key is kept in TextMachine.keys."""
    t = (1 << 24) + k[0] * P2 + k[1]
    V.KEYOF[t] = k
    return t


PREV = set()  # stand-ins written as <nmp>: a pending variable flushed at the next word (the kernel keys it by the word
# before the one it just closed, so it needs no lookup of whether that word was a variable)


def prev(s):
    t = s + (1 << 25)
    V.KEYOF[t] = V.KEYOF[s]
    PREV.add(t)
    return t


RESERVED = set(WORDS) | set(SHARED) | set(SCRATCH) | set(OPENERS) | set(CLOSERS) | set(ALIAS) | {
    ":", ";", "variable", "value", "constant", "to", "@", "!", "+!", "(", "\\", "i", "exit", "leave",
    "min", "max", "emit", '."', ".(", "type", 's"'}
CTLNAMES = set(OPENERS) | set(CLOSERS) | set(ALIAS) | {"i", "exit", "leave"}  # the kernel's control words: never a name


def sig(name):
    """What the kernel matches a word against the built-in words by: its key, length and last character."""
    return name_key(name), len(name), name[-1:]


RESERVED_SIG = {sig(w): w for w in RESERVED}
TOPS = standin(name_key("\x01top"))  # the hidden top-level definition: its own name token <nmtop>
SHORDER = {standin(name_key("\x01" + w)): k for k, w in enumerate(V.NMSH)}
HNAME = {TOPS: V.NMTOP} | {standin(name_key("\x01" + w)): t for w, t in V.NMSH.items()}  # hidden names: own tokens
assert len(RESERVED_SIG) == len(RESERVED)


def scratch_seq(lw):
    """rot -rot min max /mod */ through the hidden scratch cells t1 t2 t3."""
    t = SCR
    if lw == "/mod":  # a b -- a mod b, a / b
        return [S(t[0]), S(t[1]), F(t[1]), F(t[0]), V.MOD, F(t[1]), F(t[0]), V.DIV]
    if lw == "*/":  # a b c -- a * b / c
        return [S(t[0]), V.MUL, F(t[0]), V.DIV]
    if lw in SCRATCH:  # pop c b a into t1 t2 t3, push in SCRATCH's order
        return [S(t[0]), S(t[1]), S(t[2])] + [F(t["cba".index(ch)]) for ch in SCRATCH[lw]]
    return [S(t[1]), S(t[2]), F(t[2]), F(t[1]), V.LT if lw == "min" else V.GT, S(t[0]), F(t[1]), F(t[2]),  # b - (a - b) * flag
            F(t[1]), V.SUB, F(t[0]), V.MUL, V.SUB]


NEED = {V.ADD: 2, V.SUB: 2, V.LT: 2, V.GT: 2}  # pops that would happen before an underflow is noticed
GROW = {V.LIT, V.DUP, V.OVER, V.SWAP, V.DEPTH}  # ops that need a free stack row (+ cell fetches)


DUE = {V.CARRY, V.BORROW, V.RCARRY, V.RBORROW, V.TICK, V.TICK2}


class Exec(sp.Checked):
    """sp.Checked (sp_vm + the kernel's value limits); a definition also stores literals written in machine digits.
    failop: the op being run when an error was raised."""

    failop, ticking = None, True

    def execute(self, t, step):
        self.failop = t
        s = self.stack
        if len(s) < (self.lastdig + 2 if t == V.PICK else NEED.get(t, 0)):  # (nothing changed yet)
            raise IndexError("stack underflow")
        if len(s) + (2 if t in V.MULS else t in GROW or bool(V.cell_of(t) and V.cell_of(t)[0] == "fetch")) > V.NPOS:
            raise ForthError(f"data stack over {V.NPOS} values")  # (* / MOD: a scratch row above the top too)
        if t in (V.ADD, V.SUB, V.INC, V.DECR) and len(s) >= (2 if t in (V.ADD, V.SUB) else 1):
            r = s[-2] + s[-1] if t == V.ADD else s[-2] - s[-1] if t == V.SUB else s[-1] + (1 if t == V.INC else -1)
            if len(str(abs(r))) > sp.MAX_DIGITS:
                raise ForthError(f"value over {sp.MAX_DIGITS} digits")
        if t == V.MUL and len(s) >= 2 and len(str(abs(s[-1]))) <= sp.MAX_MUL_TOP and \
                len(str(abs(s[-2]))) + len(str(abs(s[-1]))) > sp.MAX_DIGITS:  # (by digit count: known at the op)
            raise ForthError(f"product over {sp.MAX_DIGITS} digits")
        super().execute(t, step)

    def rpush(self, v):
        if len(self.rs) >= V.NPOS:
            self.failop = V.CALL
            raise ForthError(f"return stack over {V.NPOS}")
        super().rpush(v)

    def compile_tok(self, t):
        """sp_vm's, with a compile-time control stack (structures nest in place): <if> / <while> push their cell,
        <else> / <then> / <repeat> patch the top; <begin> pushes its address (<repeat>'s and <lv0>'s target)."""
        t = V.DIGITS[V.MD.index(t)] if t in V.MD else V.MINUS if t == V.MMINUS else t
        cell = t in V.CELLOPS or ((t in V.DIGITS or t == V.MINUS) and self.inlit)
        if cell and len(self.code) >= V.NADDR - 1:
            raise ForthError(f"code memory full ({V.NADDR} cells: fewer or shorter definitions)")  # (one cell kept for ;)
        if t not in (V.IF, V.WHILE, V.ELSE, V.REPEAT, V.THEN, V.BEGIN, V.DOB, V.SEMI, V.LEAVES[0]):
            return super().compile_tok(t)
        self.inlit = False
        a = len(self.code)
        if t == V.THEN:
            self.code[self.opens.pop()]["tgt"] = a
        elif t in (V.BEGIN, V.DOB):
            self.begins.append(a)
        elif t == V.SEMI:
            self.code.append({"op": V.EXIT})
            self.compile, self.opens, self.begins = False, [], []
        else:
            self.code.append({"op": t})
            if t in (V.IF, V.WHILE):
                self.opens.append(a)
            elif t == V.ELSE:
                self.code[self.opens[-1]]["tgt"], self.opens[-1] = a + 1, a
            elif t == V.REPEAT:
                self.code[self.opens.pop()]["tgt"], self.code[a]["tgt"] = a + 1, self.begins.pop()
            else:  # <lv0>: back to the innermost loop's entry <while> (with a 0 flag before it: the loop ends)
                self.code[a]["tgt"] = self.begins[-1]

    def __init__(self):
        super().__init__()
        self.opens, self.begins = [], []

    def pop(self):
        if not self.stack:
            raise IndexError("stack underflow")
        return super().pop()


class ForthError(Exception):
    """A program the kernel cannot run: reported in the output block, then the machine stops for the block."""


HIDE = {V.MINUS: V.MMINUS, V.SPACE: V.PSP, V.NL_TEXT: V.PNL} | dict(zip(V.DIGITS, V.MD))  # a print's text, hidden


POPS = {V.DROP, V.PR, V.IF, V.WHILE, V.MUL, V.MOD, V.DIV} | set(NEED) | set(range(V.STORE0, V.STORE0 + V.NCELL)) | \
    set(range(V.XSTORE0, V.XSTORE0 + V.NXCELL)) | set(range(V.HSTORE0, V.HSTORE0 + V.NHCELL))
PUSHES = {V.DUP, V.OVER, V.SWAP}


class TextMachine:
    """An error stops the block at <err..> (right after the failing token) with one fixed line 'error: <why>' (ERRS);
    at the next fence the kernel ends any replay (<done>) or half-written definition (its ; : it stays defined), drops
    the stack (<drop>s) and keeps everything else (code, names, variables, return frames)."""

    def __init__(self, tok):
        self.tok, self.m, self.keys, self.out, self.printed = tok, Exec(), [], [], []
        self.fence = self.after = self.forth = self.ws = self.wsnl = self.comment = self.linecomment = False
        self.word, self.error, self.lits, self.nprints = "", None, 0, 0
        self.cur, self.defkeys = "", set()  # (the word being translated)
        self.freeat = None  # a1: the last top-level hidden definition's first cell, while nothing compiled after it
        self.front = {"ver": {}, "cur": {}, "owner": {}, "cells": 0, "declared": set()}  # survives errors
        self.clear_transient()
        self.commit()

    def clear_transient(self):
        self.ctx, self.pending, self.expect, self.defining, self.flush, self.newvars = [], None, None, None, [], []

    def state(self):
        dic = dict(self.m.dict)
        if self.m.mode == "var" and self.m.name:  # sp_vm binds a variable at its next token: keep a just-declared one
            dic[V.key(self.m.name)] = self.m.nvar - 1
        return copy.deepcopy(self.front), len(self.m.code), dic, self.m.nvar

    def commit(self):
        """State that survives an error in a later block: completed definitions / variables (cell values stay live)."""
        self.snap = self.state()

    def abandon(self):
        """A block cut by </think>: back to the block's start (code memory, names, variables; the stack is dropped, as
        after an error), its prints discarded, the parser outside any block."""
        self.snap = self.bmark
        self.reset()
        self.fence, self.word, self.printed, self.comment, self.linecomment = False, "", [], False, False
        return self.out

    def reset(self):
        front, ncode, dic, nvar = self.snap
        m = Exec()
        m.code, m.dict, m.nvar, m.cells = self.m.code[:ncode], dict(dic), nvar, dict(self.m.cells)
        self.m, self.front, self.error = m, copy.deepcopy(front), None
        self.clear_transient()

    def kreset(self):
        """The kernel's reset after its error: end the replay / the half-written definition, drop the stack."""
        m, self.error = self.m, None
        m.err = False  # shortcut: a value overflow is cleared here; the kernel's overflow state (OVFO) is sticky (later)
        if self.kdef and m.compile:  # its ; : it stays defined (the dictionary bound it at its name)
            self.front["cur"][self.kdef[0]] = self.kdef[1]
        self.clear_transient()
        if m.queue or len(m.stack) > V.NPOS or len(m.rs) > V.NPOS:  # an overflow / the token cap stopped a run mid-way:
            m.stack, m.rs, m.queue, m.lo, m.hi, m.rlo, m.rhi = [], [], [], 0, 0, 0, 0  # both stacks emptied
            m.segmode = m.inlit = False
        self.run([V.DONE] * m.replay + [V.SEMI] * m.compile)
        self.run([V.DROP] * len(m.stack))
        self.commit()

    # ---- text tokens ----
    def feed(self, t):
        """One model text token -> the kernel tokens it emits right after it."""
        s, self.out = self.tok.decode([t]), []
        if not self.fence:
            if not (self.forth and s[:1] == "\n"):  # (a fence opens at the token after ```forth that starts a line)
                self.forth, self.after, self.ws = self.after and s.strip().lower() == "forth", s.strip() == FENCE, False
                return self.out
            self.fence, self.forth = True, False  # this token is the block's first (its newline: whitespace)
            self.nprints, self.comment, self.linecomment, self.wsnl = 0, False, False, False
            self.m.tc1 = self.m.tc2 = self.m.tc3 = 0  # (the reset's machine tokens count too)
            if self.error:
                self.kreset()
            elif self.freeat is not None and -(-self.freeat // 16) * 16 < len(self.m.code):  # a1: free its whole heads
                self.run([V.FREE[-(-self.freeat // 16)]])
                self.freeat = None
                self.commit()
            self.m.rs, self.m.rlo, self.m.rhi = [], 0, 0  # the return stack is emptied (frames left by an error)
            self.bmark = self.state()
        if s.strip() == THINK_END:  # the thinking ended inside the block: a draft, dropped whole (no output block)
            return self.abandon()
        if s.strip() == FENCE:  # last word, then the output block; hand back to the model
            self.close()
            self.guard(self.end_block)
            self.fence = False
            line = self.printed[:-1] if self.printed[-1:] == [V.SPACE] else self.printed  # the last print's space
            if self.error:
                line += [NL] * bool(line) + self.text("error: " + self.error)
            line = line or self.text(EMPTY) * (not self.printed)  # (a lone space printed: an empty line)
            lead = [NL] * (s[-1:] != "\n")  # (a closing '```\n' already ended the line)
            self.out += lead + [TICKS, OUTPUT, NL] + line + [NL, TICKNL]  # (ends '```\n', one token: as loop_sim's rollouts)
            self.printed = []
            self.clear_transient()
            return self.out
        if self.linecomment:
            if "\n" not in s:
                return self.out
            self.linecomment, s = False, "\n" + s[s.rindex("\n") + 1:]
        if re.search(r"\s", s.strip()):
            self.guard(lambda: self.fail(f"token {s.strip()!r} holds two words"))
        if s[:1].isspace() or self.ws:
            self.close()
            if self.linecomment:  # '\' just closed: this token is in the comment, unless the line already ended
                if self.wsnl or "\n" in s:
                    self.linecomment = False
                    if not self.wsnl:
                        s = "\n" + s[s.rindex("\n") + 1:]
                else:
                    return self.out
        tail = s[len(s.rstrip()):]
        self.word, self.ws, self.wsnl = self.word + s.strip(), bool(tail) or not s, "\n" in tail
        return self.out

    def text(self, s):
        """Kernel-written text: a token that is ``` is written as `` and ` (the same text), so the kernel's own output
        never opens or closes a fence."""
        out = []
        for t in self.tok(s, add_special_tokens=False).input_ids:
            d = self.tok.decode([t])
            out += self.tok(d[:-1], add_special_tokens=False).input_ids + [TICK1] if d.strip() == FENCE else [t]
        return out

    def close(self):
        w, self.word = self.word, ""
        if w:
            self.guard(lambda: self.translate(w))

    def guard(self, f):
        if self.error:
            return
        try:
            f()
        except Exception as e:  # any malformed program ends in an error line, never an exception
            if isinstance(e, ForthError) and len(self.ctx) == 1 and self.ctx[0]["kind"] == "top" and not self.ctx[0]["ctl"] \
                    and err_class(e) != "code":  # (code memory full errors at the cell, as in the kernel: nothing runs first)
                self.guard(self.end_top)  # finished top-level code before the bad word runs first, as an interpreter would
                if self.error:
                    return
            c, t = err_class(e), self.m.failop
            if c in ("digits", "full") and t in (V.ADD, V.SUB) + V.MULS:  # (their pop undone)
                c += "2"
            if c == "prints" and t != V.PR:
                c = "prints1"
            self.out.append(V.ERRT[c if c != "underflow" else "up" if t == V.PR else "u+" if t in POPS else
                                   "u-" if t in PUSHES else "u0"])  # (the kernel: <err..> right after the failing token)
            self.error, self.kdef = err_text(c), self.defining  # (a half-written definition: its name, for the reset)
            self.freeat = None

    def fail(self, why):
        raise ForthError(why)

    # ---- names ----
    def define(self, name, kind):
        """A new version of name -> its entry {'kind', 's' (stand-in)}; a key taken by another name is an error."""
        f = self.front
        v = 0  # no versions: a redefinition rebinds the name everywhere (the kernel dictionary has one entry per key)
        k = name_key(name, v)
        if name[:1] != "\x01" and name not in RESERVED and sig(name) in RESERVED_SIG:  # the kernel would read it as
            self.fail(f"name key collision {RESERVED_SIG[sig(name)]} / {name}")  # that built-in word
        if k in f["owner"] and f["owner"][k][0] != name:
            other = f["owner"][k][0].lstrip("\x01")
            self.fail(f"name key collision {other} / {name.lstrip(chr(1))}")
        f["owner"][k], f["ver"][name] = (name, v, kind), v + 1
        if kind == "def":
            self.defkeys.add(k)
        return {"kind": kind, "s": standin(k)}

    def cell(self, s):
        self.front["cells"] += 1
        if self.front["cells"] > MAXCELL:
            self.fail(f"over {MAXCELL} variables")
        return [V.VAR, s]

    def lookup(self, name):
        if self.defining and self.defining[0] == name:
            return self.defining[1]  # a word's own name inside its body: itself
        return self.front["cur"].get(name)

    def hidden_var(self, name):
        """A hidden variable's stand-in, declared now (top level) or after the enclosing definition."""
        if name not in self.front["declared"]:
            self.front["declared"].add(name)
            e = self.define(name, "var")
            self.front["cur"][name] = e
            if self.ctx:
                self.newvars.append(e["s"])
            else:
                self.run(self.cell(e["s"]))
                self.commit()
        return self.front["cur"][name]["s"]

    def shared(self, w):
        """A SHARED word: its hidden definition at first use, then a call."""
        name = "\x01" + w
        if name not in self.front["declared"] and not any(ids[1] == standin(name_key(name)) for ids in self.flush):
            e = self.define(name, "def")
            self.front["cur"][name] = e
            ids = [V.DEF, e["s"]] + SHARED[w] + [V.SEMI]
            if self.ctx:  # declared when its definition is compiled (finish): an error before that drops it, as in
                self.flush.append(ids)  # the kernel (DECL is set at the <nm..> after <:>)
            else:
                self.front["declared"].add(name)
                self.run(ids)
                self.commit()
        self.emit([V.CALL, self.front["cur"][name]["s"]])

    def opener(self, lw):
        """A structure starts: at top level it opens the hidden top-level definition; inside another open structure it
        compiles in place, one level deeper (the kernel: its control stack)."""
        if not self.ctx:  # top level: a hidden definition collects the code, run at the end of the statement run
            e = self.define("\x01top", "def")
            self.run([V.DEF, e["s"]])
            self.ctx.append({"kind": "top", "ctl": None, "owner": "\x01top", "s": e["s"], "start": len(self.m.code)})
        if self.ctx[-1]["ctl"]:  # nested: in place, one level deeper
            self.ctx.append({"kind": "nest", "ctl": None, "owner": self.ctx[0]["owner"], "s": None})
        if self.depth() > MAXNEST:
            self.fail(f"structures nested over {MAXNEST} deep")

    def depth(self):
        return sum(c["ctl"] is not None for c in self.ctx) + 1  # (the structure about to open / the while's if)

    def close_helper(self):
        if self.ctx[-1]["kind"] == "nest" and not self.ctx[-1]["ctl"]:  # the nested structure is complete
            self.ctx.pop()

    # ---- emission ----
    def emit(self, ids):
        self.run(ids)

    def finish(self, call=None):
        """After the outermost definition's <;>: hidden shared words, hidden variables, then (hidden top-level code) the
        call."""
        order = [k for k, ids in enumerate(self.flush) if ids[1] in SHORDER]
        for k, ids in zip(order, sorted((self.flush[k] for k in order), key=lambda ids: SHORDER[ids[1]])):
            self.flush[k] = ids  # hidden shared words in a fixed order (gcd lcm ?dup), as the kernel emits them
        for ids in self.flush:
            if ids[1] in SHORDER:
                self.front["declared"].add("\x01" + list(V.NMSH)[SHORDER[ids[1]]])
            self.run(ids)
        for s in self.newvars:
            self.run(self.cell(s))
        self.flush, self.newvars = [], []
        if call:
            self.run([V.CALL, call])
        self.commit()

    def end_top(self):
        """Close the hidden top-level definition and run it."""
        c = self.ctx[-1]
        if len(self.ctx) > 1 or c["ctl"]:
            self.fail(f"unterminated {(self.ctx[-1]['ctl'] or 'structure').replace('while', 'begin')}")
        self.run([V.SEMI])
        self.freeat = c["start"]  # (a hidden shared word flushed after it is a DEF: run() clears this again)
        self.ctx.pop()
        self.finish(call=c["s"])

    def end_block(self):
        if self.pending:
            self.emit([V.FETCHM, self.pending["s"]])
            self.pending = None
        if self.ctx:
            if self.ctx[0]["kind"] == "def":
                self.fail("unterminated definition")
            self.end_top()

    # ---- words ----
    def translate(self, w):
        """A closed word -> its protocol tokens, emitted."""
        self.m.failop = None
        lw = ALIAS.get(w.lower(), w.lower())
        self.cur = w
        if not self.comment and w in ("(", "\\"):
            if self.pending:  # a comment ends a pending variable name: fetched before the comment
                p, self.pending = self.pending["s"], None
                self.emit([V.FETCHM, prev(p)])
        if self.comment or w == "(":
            self.comment = w == "(" or (self.comment and not w.endswith(")"))
            return
        if w == "\\":
            self.linecomment = True
            return
        if self.pending:
            p, self.pending = prev(self.pending["s"]), None
            if lw in ("@", "!"):
                return self.emit([V.FETCHM if lw == "@" else V.STOREM, p])
            if lw == "+!":
                return self.emit([V.FETCHM, p, V.ADD, V.STOREM, p])
            self.emit([V.FETCHM, p])  # a bare variable name is a fetch
        if self.expect:
            return self.name_after(self.expect, lw)
        if re.fullmatch(r"-?[0-9]+", w):
            if len(w.lstrip("-")) > sp.MAX_DIGITS:  # (as written: the kernel's digit registers count characters)
                self.fail(f"literal over {sp.MAX_DIGITS} digits")
            v = int(w)
            return self.emit([V.LIT] + [V.MMINUS] * (v < 0) + [V.MD[int(c)] for c in str(abs(v))])
        if lw in (":", "variable", "value", "constant", "to"):
            if self.ctx and self.ctx[0]["kind"] == "top" and lw != "to":
                self.end_top()
            if self.ctx and lw != "to":
                self.fail(f"{lw} inside a definition")
            self.expect = lw  # <:> / <var> and the name at the next word (once it is known whether this block replaces one)
            return
        if lw == ";":
            c = self.ctx[-1] if self.ctx else None
            if not c or c["kind"] == "top":
                self.fail("; outside a definition")
            if c["ctl"] or c["kind"] == "nest":
                self.fail(f"; with {c['ctl']} still open")
            self.run([V.SEMI])
            self.front["cur"][self.defining[0]], self.defining = self.defining[1], None
            self.ctx.pop()
            return self.finish()
        if lw in OPENERS or lw in CLOSERS or lw in ("i", "exit"):
            return self.control(lw)
        if self.lookup(lw):  # a user definition shadows a built-in word (control words stay reserved)
            return self.word_ref(self.lookup(lw))
        if lw in WORDS:
            return self.emit(WORDS[lw])
        if lw in SCRATCH or lw in ("min", "max"):
            return self.emit(scratch_seq(lw))
        if lw in SHARED:
            return self.shared(lw)
        if lw in ("@", "!", "+!"):
            self.fail(f"{lw} needs a variable name right before it (variable x  5 x !  x @ .  2 x +!)")
        if lw in ('."', ".(", "type", "s\""):
            self.fail(f"{lw} is not supported: the machine prints numbers only (. prints one; cr, space for layout)")
        if lw == "leave":
            return self.leave()
        if lw == "emit":
            self.fail("emit is not supported (print numbers with . ; cr, space for layout)")
        e = self.lookup(lw)
        if not e:
            self.fail(f"unknown word {lw}")
        self.word_ref(e)

    def word_ref(self, e):
        if e["kind"] == "var":
            self.pending = e  # decided by the next word: @, !, or a fetch
        else:
            self.emit([V.CALL if e["kind"] == "def" else V.FETCHM, e["s"]])

    def rebind(self, lw):
        """A name the kernel's dictionary already holds, declared again: <rbh_hi> <rbl_kind_lo> carry the old entry's
        value and kind, so the kernel cancels that entry before the new binding (the dictionary is add-only state)."""
        k = name_key(lw)
        dic = self.state()[2]  # (a variable just declared is bound at its next token)
        if self.front["owner"].get(k, ("",))[0] == lw and k in dic:
            v, kind = dic[k], self.front["owner"][k][2]
            self.run([V.RBH[v // 16], V.RBL[{"def": 0, "var": 1, "constant": 2}[kind]][v % 16]])

    def name_after(self, kind, lw):
        self.expect = None
        if kind == "to":
            e = self.lookup(lw)
            if not e or e["kind"] not in ("value", "var"):  # models write `to` for variables too: it is a store
                self.fail(f"to {lw}: not a value")
            return self.emit([V.STOREM, e["s"]])
        if kind == ":" and len(self.m.code) >= V.NADDR - 1:  # at the name, before any token (the kernel: NOFREE, <err code2>)
            self.fail("code memory full (no free cell for a definition)")
        if kind != ":" and self.front["cells"] >= MAXCELL:  # likewise (the kernel: VARSFULL, KCLS 'vars')
            self.fail(f"over {MAXCELL} variables")
        if lw in CTLNAMES:  # (the kernel reads them as control words wherever they appear)
            self.fail(f"name key collision {lw} / {lw}")
        self.rebind(lw)
        self.run([V.DEF] if kind == ":" else self.cell(None)[:1])
        if kind == ":":
            e = self.define(lw, "def")
            self.defining = (lw, e)
            self.run([e["s"]])
            self.ctx.append({"kind": "def", "ctl": None, "owner": lw, "s": e["s"]})
            return
        e = self.define(lw, {"variable": "var", "value": "var"}.get(kind, kind))  # a value is a variable with its
        # number stored at once: bare name, @, ! and to all work on either (models mix them); constants stay read-only
        self.run([e["s"]] + [V.CST] * (kind == "constant") + [V.STOREM, e["s"]] * (kind != "variable"))  # value /
        # constant: the number before (<cst>: a constant's binding token)
        self.front["cur"][lw] = e
        self.commit()

    def control(self, lw):
        if lw == "i":  # the innermost do loop's index (i is never a name: CTLNAMES)
            c = next((c for c in reversed(self.ctx) if c["ctl"] == "do"), None)
            if c:
                return self.emit([F(c["loop"][0])])
            self.fail("i outside a do loop")
        if lw == "exit":
            return self.emit([V.EXIT])
        if lw in OPENERS:
            self.opener(lw)
            c = self.ctx[-1]
            c["ctl"] = lw
            if lw == "do":  # i, limit: the hidden cells of this lexical loop depth; the flag stays on the stack
                d = sum(x.get("loop") is not None for x in self.ctx[:-1]) + 1
                if d not in LOOPC:
                    self.fail(f"do loops nested over {max(LOOPC)} deep")
                c["loop"] = i, lim = LOOPC[d]
                return self.emit([S(i), S(lim), F(i), F(lim), V.SUB, V.ZEQ, V.ZEQ, V.DOB, V.WHILE])
            return self.emit(LOOPS[lw] if lw == "begin" else WORDS[lw])
        c = self.ctx[-1] if self.ctx else None
        if not c:
            self.fail(f"{lw} outside a definition")
        if lw == "repeat" and c["ctl"] == "begin":  # begin ... repeat with no while: the endless loop (= again) that
            lw = "again"                            # models write with a leave / exit inside
        if c["ctl"] not in CLOSERS[lw]:
            self.fail(f"{lw} without {' or '.join(CLOSERS[lw])}")
        if lw == "while" and self.depth() > MAXNEST:  # (its if: one level deeper)
            self.fail(f"structures nested over {MAXNEST} deep")
        c["ctl"] = {"else": "else", "while": "while"}.get(lw)
        rep = V.REPEAT
        if lw in ("loop", "+loop"):
            i, lim = c["loop"]
            c["loop"] = None
            if lw == "loop":  # i += 1; flag i < limit
                self.emit([F(i), V.INC, S(i), F(i), F(lim), V.LT, rep])
            else:  # n: flag (i < limit) == (i + n < limit); i += n
                self.emit([F(i), F(lim), V.LT, V.SWAP, F(i), V.ADD, V.DUP, S(i), F(lim), V.LT, V.SUB, V.ZEQ, rep])
        else:
            self.emit(LOOPS.get(lw, WORDS.get(lw)))
        self.close_helper()

    def leave(self):
        """leave: a 0 flag, then back to the innermost loop's entry <while> (every loop starts with one), which ends it."""
        if not any(c["ctl"] in ("do", "begin", "while") for c in self.ctx):
            self.fail("leave outside a loop")
        self.emit(LOOPS["leave"])

    def run(self, ids):
        """Emit protocol tokens one at a time; after each the machine runs until it hands back (as sp.stream). A print's
        text is hidden here (machine digits, <psp>, <pnl>) and kept for the output block."""
        for t in ids:
            self.lits += t == V.LIT and not self.m.compile
            assert self.lits <= sp.MAX_LITS, f"over {sp.MAX_LITS} literals between kernel turns"
            if t in V.KEYOF:
                self.keys.append(V.KEYOF[t])
            if t == V.DEF:  # code compiled after the top-level definition: it is not freed
                self.freeat = None
            self.out.append(V.NMP if t in PREV else HNAME.get(t) or (V.NM if t in V.KEYOF else t))
            self.count(t in PRINTS and not self.m.compile, t)
            self.m.feed(t)
            while (k := self.m.nxt()) is not None:
                assert k != V.OVF, "overflow"
                m = self.m
                if m.tc3 >= CAPT and m.replay and not m.queue and m.jump is None and k not in DUE:  # (a replayed cell)
                    raise ForthError(f"the block ran too long (over {CAPT << 14} kernel steps)")
                self.lits = 0
                if k in HIDE:
                    self.printed.append(k)
                self.out.append(HIDE.get(k, k if k >= 0 else 501))  # (a big store's address: shown as <A_0>)
                self.count(k in PRINTS, k)
                self.m.feed(k, kernel=True)

    def count(self, pr, t):
        """A print (. cr space: each takes a row of the kernel's print memory), right after its token."""
        self.m.failop = t
        if t == V.PR and not self.m.stack:  # (a print that underflows: that error, not counted)
            return
        self.nprints += pr
        if self.nprints > MAXPRINT:
            raise ForthError(f"over {MAXPRINT} prints in one block")


def stream(ids, tok):
    """Model text token ids (prose + fences) -> (all ids, kernel-owned offsets, <nm> keys in order)."""
    tm, out, owned = TextMachine(tok), [], []
    for t in ids:
        out.append(t)
        for k in tm.feed(t):
            owned.append(len(out))
            out.append(k)
    return out, owned, tm.keys


def text_ids(program, tok):
    return tok("```forth\n" + program.strip() + "\n```", add_special_tokens=False).input_ids


def visible(ids, owned, tok):
    """What the kernel showed: its non-special tokens, which must all come after the model's last token (the block)."""
    last = max(set(range(len(ids))) - set(owned))
    assert all(o > last for o in owned if ids[o] >= 1000), "visible text mid-program"
    return tok.decode([ids[o] for o in owned], skip_special_tokens=True)


def blocks(texts, tok):
    """Several ```forth blocks in one turn (state carries) -> each block's output text."""
    ids = tok("".join(f"```forth\n{b}\n```\nok\n" for b in texts), add_special_tokens=False).input_ids
    out, owned, _ = stream(ids, tok)
    return re.findall(r"```output\n(.*?)\n```", tok.decode([out[o] for o in owned], skip_special_tokens=True), re.S)


SHORT = "x y z i j k n m a b c s t run step sum next main loop count total".split()
CASES = [  # (program, its output block: prints, then any error's two fixed lines)
    ('0 value x 5 to x x 2 * .', '10'),
    ('7 VALUE n : twice n 2 * ; twice . 3 TO n twice .', '14 6'),
    ('0 value acc : add acc + to acc ; 3 add 4 add acc .', '7'),
    (': f 3 ; 4 f + .', '7'),
    (': f 1 repeat ; 1 .', 'error: a structure word without its opener\nfix the program and send the corrected block again'),
    (': f 0 begin 1+ dup 3 = if leave then repeat . ; f', '3'),
    (': f begin dup while 1- ; 3 f .', 'error: ; with a structure still open\nfix the program and send the corrected block again'),
    (': f until ;', 'error: a structure word without its opener\nfix the program and send the corrected block again'),
    ('1 2 . 5 to y 3 .', '2\nerror: to needs the name of a value or variable\nfix the program and send the corrected block again'),
    (': f 1', 'error: unterminated definition\nfix the program and send the corrected block again'),
    ('1 0 / .', 'error: / or MOD by zero\nthe divisor was 0: check the values that reach / or mod; fix the program and send the corrected block again'),
    (': foo 5 ; : bar foo ; : foo 6 ; bar . foo .', '6 6'),
    (': f 5 ; : f dup 0 > if 1- f then ; 3 f .', '0'),
    (': g dup 0 > if dup . 1- g then ; 3 g drop', '3 2 1'),
    ('variable i 5 i ! i @ .', 'error: a name collides with another name: rename it\nfix the program and send the corrected block again'),
    (': fact dup 1 > if dup 1- fact * then ; 5 fact .', '120'),
    (': f 1 ; variable f 3 f ! f @ .', '3'),
    (': f 1 ; : g f ; variable f 2 f ! g . f .', '1 2'),
    (': f 5 0 do i . loop ; f', '0 1 2 3 4'),
    ('10 0 do i . 3 +loop', '0 3 6 9'),
    ('0 3 do i . -1 +loop', '3 2 1 0'),
    ('3 3 do i . loop 7 .', '7'),
    (': sq 0 swap 1+ 1 do i dup * + loop ; 4 sq .', '30'),
    ('0 4 0 do 3 0 do 1+ loop loop .', '12'),
    ('i', 'error: i outside a do loop\nfix the program and send the corrected block again'),
    ('1 if 2 . then 3 .', '2 3'),
    ('1 . 0 if 2 . else 4 . then 5 .', '1 4 5'),
    ('variable q 3 q ! begin q @ while q @ . q @ 1- q ! repeat q @ .', '3 2 1 0'),
    ('1 if 2 . then : f 9 ; f .', '2 9'),
    ('1 if 2 . then variable v 4 v ! v .', '2 4'),
    ('begin 1 .', 'error: unterminated structure\nfix the program and send the corrected block again'),
    ('1 then', 'error: a structure word without its opener\nfix the program and send the corrected block again'),
    (': f 1 if 2 if 3 . then then ;', ''),
    (': f 1 if 2 if 3 . then then ; f', '3'),
    (': f dup 0 > if dup 2 mod if 1 . else 2 . then then drop ; 5 f 4 f -1 f', '1 2'),
    (': f 3 0 do i 2 mod if i . then loop ; f', '1'),
    (': f 1 if begin 0 while repeat 7 . then ; f', '7'),
    (': f 1 if 1 if exit then then 5 . ; f 6 .', '6'),
    (': f 3 0 do i 1 = if exit then i . loop 7 . ; f 9 .', '0 9'),
    (': g 1 if 1 if 1 if exit then then then 7 . ; g 8 . : h 0 if exit then 4 . ; h', '8 4'),
    ('17 5 /mod . . -17 5 /mod . . 6 7 3 */ .', '3 2 -3 -2 14'),
    ('variable n 6 n ! n 7 * .', '42'),
    ('variable v v .', '0'),
    ('300 constant max max 1+ .', '301'),
    (': dup 7 ; 1 dup . .', '7 1'),
    ('5 constant k 6 to k', 'error: to needs the name of a value or variable\nfix the program and send the corrected block again'),
    ('0 value l -23 l ! l @ . 7 to l l .', '-23 7'),
    ('7 value v v @ 1+ .', '8'),
    (': f 10 0 do i . i 3 = if leave then loop 99 . ; f', '0 1 2 3 99'),
    ('10 0 do i . i 2 = if leave then loop 7 .', '0 1 2 7'),
    (': f 10 0 do i . i 1 > if i 3 = if leave then then loop ; f', '0 1 2 3'),
    (': g 5 0 do i 2 = if leave else i . then loop ; g', '0 1'),
    (': f 10 0 do i 3 = if leave then i . loop 9 . ; f', '0 1 2 9'),
    (': f 10 0 do i 4 = if leave then i . 2 +loop ; f', '0 2'),
    (': f 0 begin dup . 1+ dup 3 = if leave then again . ; f', '0 1 2 3'),
    ('0 begin 1+ dup 5 > if leave then dup 7 < while repeat .', '6'),
    (': f 1 if leave then ;', 'error: leave outside a loop\nfix the program and send the corrected block again'),
    ('3 0 do i . loop but wait', '0 1 2\nerror: unknown word\na word is not defined: check its spelling, or define it with : or declare it with variable; fix the program and send the corrected block again'),
    (': f 10 0 do i 3 = if i exit then loop -1 ; f .', '3'),
    ('1 \\ 2 . nothing here\n3 + . \\ end', '4'),
    ('( a ) 1 ( b c ) 2 + .', '3'),
    ('3 0 and . 3 4 and . 0 0 or . 0 -2 or .', '0 -1 0 -1'),
    ('1 . cr 2 . cr', '1 \n2 \n'),
    ('0 1 = . 2 2 = . 5 1+ 1- 0= .', '0 -1 0'),
    ('variable x variable y variable z variable q variable j variable k variable n variable m variable a variable b variable c variable s variable t variable run variable step variable sum variable next variable main variable lp variable count variable total 7 total ! 8 q ! total . q .', '7 8'),
    ('variable v0 variable v1 variable v2 variable v3 variable v4 variable v5 variable v6 variable v7 variable v8 variable v9 variable v10 variable v11 variable v12 variable v13 variable v14 variable v15 variable v16 variable v17 variable v18 variable v19 variable v20 variable v21 variable v22 variable v23 variable v24 variable v25 variable v26 variable v27 variable v28 variable v29 variable v30 variable v31 variable v32', 'error: over 32 variables\nfix the program and send the corrected block again'),
    (': x 0 ; : y 1 ; : z 2 ; : q 3 ; : j 4 ; : k 5 ; : n 6 ; : m 7 ; : a 8 ; : b 9 ; : c 10 ; : s 11 ; : t 12 ; : run 13 ; : step 14 ; : sum 15 ; : next 16 ; : main 17 ; : lp 18 ; : count 19 ; : total 20 ; total . run . x .', '20 13 0'),
    ('1 if 2 then : abc 1 ; : ABC 2 ; abc .', '2'),
    ('variable a  284 a ! variable b  17 b ! : step  a @ b @ 9 * + 438026 mod  swap  b !  a ! ; : run  73 0 do  step  loop ; run b @ .', 'error: stack underflow\na word needed more values than the stack held: check what each word takes and leaves; fix the program and send the corrected block again'),
    ('drop', 'error: stack underflow\na word needed more values than the stack held: check what each word takes and leaves; fix the program and send the corrected block again'),
    ('1 2 2dup . . . .', '2 1 2 1'),
    ('1 2 3 rot . . .', '1 3 2'),
    ('1 2 3 -rot . . .', '2 1 3'),
    ('1 2 nip . 3 4 tuck . . .', '2 4 3 4'),
    ('0 not . 5 not .', '-1 0'),
    ('7 2* . 7 2/ . -7 abs . 7 abs .', '14 3 7 7'),
    ('3 5 min . 3 5 max . -2 -9 min . -2 -9 max .', '3 5 -9 -2'),
    ('7 >r 8 r> . .', 'error: unknown word\na word is not defined: check its spelling, or define it with : or declare it with variable; fix the program and send the corrected block again'),  # .s spaces >r r> r@: not in the dialect (the kernel's unknown word)
    ('1 . space 2 . 3 spaces 4 .', '1  2\nerror: unknown word\na word is not defined: check its spelling, or define it with : or declare it with variable; fix the program and send the corrected block again'),
    ('65 emit', 'error: strings and emit are not supported: the machine prints numbers only (. prints one; cr, space for layout)\nfix the program and send the corrected block again'),
    ('3 3 <= . 4 3 <= . 3 3 >= . 2 3 >= . 2 3 <> . 3 3 <> .', '-1 0 -1 0 -1 0'),
    ('5 0<> . 0 0<> . 5 0> . -5 0> . -5 0< . 0 0< .', '-1 0 -1 0 -1 0'),
    ('true . false . 0 invert . 5 invert .', '-1 0 -1 -6'),
    ('1 2 3 2drop .', '1'),
    ('variable n 5 n ! 3 n +! n @ . : f 10 n +! ; f n .', '8 18'),
    (': f 0 begin 1+ dup 5 = until ; f .', '5'),
    ('0 begin 1+ dup 3 > until .', '4'),
    (': f 0 begin 1+ dup 4 = if exit then again ; f .', '4'),
    (': f 3 0 ?do i . loop 2 2 ?do i . loop ; f', '0 1 2'),
    ('5 ?dup . . 0 ?dup .', '5 5 0'),
    (': f ?dup if . then ; 7 f 0 f 9 .', '7 9'),
    (': f again ;', 'error: a structure word without its opener\nfix the program and send the corrected block again'),
    ('48 18 gcd . 4 6 lcm . -4 6 gcd . 8360 33782 lcm . 0 5 gcd .', '6 12 2 7432040 5'),
    (': gcd 1 ; 8 12 gcd . .', '1 12'),
    ('depth . 5 6 depth . . .', '0 2 6 5'),
    ('1 2 3 .s 4 .', 'error: unknown word\na word is not defined: check its spelling, or define it with : or declare it with variable; fix the program and send the corrected block again'),
    ('72395 !', 'error: @ ! +! need a variable name right before them (variable x  5 x !  x @ .  2 x +!)\nfix the program and send the corrected block again'),
    ('5 . ." hi"', '5\nerror: strings and emit are not supported: the machine prints numbers only (. prints one; cr, space for layout)\nfix the program and send the corrected block again'),
    (': f begin 1 while repeat ; f', 'error: ' + err_text('tokens')),
    ('1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 . 1 .', '1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\nerror: over 128 prints in one block\nfix the program and send the corrected block again'),
]
# (blocks of one turn, each block's output): an error stops its block; the next block resets (stack dropped, a
# half-written definition ended and kept); state carries
MCASES = [
    (['variable x 5 x ! : f x @ ;', '1 0 / .', 'f .'], ['', 'error: / or MOD by zero\nthe divisor was 0: check the values that reach / or mod; fix the program and send the corrected block again', '5']),
    ([': g 7 ; : h g blah ;', 'g . h', ': h g 1+ ; h .'], ['error: unknown word\na word is not defined: check its spelling, or define it with : or declare it with variable; fix the program and send the corrected block again', '7', '8']),
    (['1 2 3 drop drop', 'frob', '4 .'], ['', 'error: unknown word\na word is not defined: check its spelling, or define it with : or declare it with variable; fix the program and send the corrected block again', '4']),
    (['variable v drop', '7 v ! v @ .'], ['error: stack underflow\na word needed more values than the stack held: check what each word takes and leaves; fix the program and send the corrected block again', '7']),
    (['variable v  3 v ! : f v @ ;', 'variable w drop', '5 w ! f w @ + .'], ['', 'error: stack underflow\na word needed more values than the stack held: check what each word takes and leaves; fix the program and send the corrected block again', '8']),
    (['5 0 do i . loop', '2 0 do i . loop'], ['0 1 2 3 4', '0 1']),
    ([': f 1 if 2 if 3 . then then ;', ': f 4 . ;', 'f'], ['', '', '4']),
    (['variable x 5 x ! : f x @ ;', 'f .', 'variable x : f 7 ;', 'f . x @ .'], ['', '5', '', '7 0']),  # no re-send rollback: x rebinds to a new cell
    ([': f 7 ;', ': g 9 ; 3 . </think> 4 .', 'f . 1 2 + .'], ['', '7 3']),
    (['3 4', ': f + ;', ': f * ;', 'f .'], ['', '', '', '12']),
    (['2 ( x', '2 3 + .'], ['', '5']),
    (['7 \\ y', '1 .'], ['', '1']),
    ([': f dup 1- f ; 5 f', '3 .'], [f'error: {err_text("full")}', '3']),  # a data / return stack overflow: an error,
    ([': f f ; f', '1 .'], [f'error: {err_text("rfull")}', '1']),  # then the next block runs (both stacks emptied)
    (['1 begin dup again', 'depth .'], [f'error: {err_text("full")}', '0']),
    (['99999999999999999999 1+ .', '7 . 99999999999999999999 1 + .', '2 .'], [f'error: {err_text("digits")}',
     f'7\nerror: {err_text("digits")}', '2']),
    (['12345678901 1234567890 * .', '1234567890 1234567890 * . 3 .'], [f'error: {err_text("digits")}',
     '1524157875019052100 3']),
]
# whole output blocks of a turn's last block (error lines, empty-output notes)
DCASES = [
    ([': g drop ; : f over * g / ; 8360 33782 f .'], 'error: stack underflow\na word needed more values than the stack held: check what each word takes and leaves; fix the program and send the corrected block again'),
    ([': step swap 2 * ; : run 3 0 do step loop ; 7 run'], 'error: stack underflow\na word needed more values than the stack held: check what each word takes and leaves; fix the program and send the corrected block again'),
    (['drop'], 'error: stack underflow\na word needed more values than the stack held: check what each word takes and leaves; fix the program and send the corrected block again'),
    (['variable n 3 n ! 1=  n @'], 'error: unknown word\na word is not defined: check its spelling, or define it with : or declare it with variable; fix the program and send the corrected block again'),
    ([': f 1 ;', ': g 2 ;', ': f 3 ;', 'g .'], '2'),  # re-defining f keeps g (no rollback)
    (['variable a a 711 !'], 'error: @ ! +! need a variable name right before them (variable x  5 x !  x @ .  2 x +!)\nfix the program and send the corrected block again'),
    ([': f dup 2 * ; : g 12 0 do i f 99 = if 1 then loop ; g 0 0 /'], 'error: / or MOD by zero\nthe divisor was 0: check the values that reach / or mod; fix the program and send the corrected block again'),
    (['( a = (b + 1) mod 7 ) 1 .'], 'error: stack underflow\na word needed more values than the stack held: check what each word takes and leaves; fix the program and send the corrected block again'),
    (['6 7 * .\n\nRun it.'], '42\nerror: unknown word\na word is not defined: check its spelling, or define it with : or declare it with variable; fix the program and send the corrected block again'),
    ([': f 7 ;', ': g 9 ; </think>', 'g'], 'error: unknown word\na word is not defined: check its spelling, or define it with : or declare it with variable; fix the program and send the corrected block again'),
    ([': f 1 ; variable e', ': f 2 ; 5 to e'], '(nothing printed: run a word and print its result with . in a ```forth block)'),  # e survives (no rollback)
    (['1 2 + .cr'], 'error: unknown word\na word is not defined: check its spelling, or define it with : or declare it with variable; fix the program and send the corrected block again'),
    ([': f (s is a comment ) 1 ;'], 'error: unknown word\na word is not defined: check its spelling, or define it with : or declare it with variable; fix the program and send the corrected block again'),
    ([': sq dup * ; : cube dup sq * ;'], '(nothing printed: run a word and print its result with . in a ```forth block)'),
    (['5 6 +'], '(nothing printed: run a word and print its result with . in a ```forth block)'),
    (['variable a b'], 'error: unknown word\na word is not defined: check its spelling, or define it with : or declare it with variable; fix the program and send the corrected block again'),
    ([': test ( -- n ) 23 negate dup * ;'], '(nothing printed: run a word and print its result with . in a ```forth block)'),
    ([': go ( -- ) 5 . ;'], '(nothing printed: run a word and print its result with . in a ```forth block)'),
    ([': two ( -- n ) 5 ( a b -- c ) ;'], '(nothing printed: run a word and print its result with . in a ```forth block)'),
    ([': sq ( n -- n*n ) dup * ;'], '(nothing printed: run a word and print its result with . in a ```forth block)'),
    ([': sq ( n -- m ) n n * ;'], 'error: unknown word\na word is not defined: check its spelling, or define it with : or declare it with variable; fix the program and send the corrected block again'),
    ([': f 2drp ;'], 'error: unknown word\na word is not defined: check its spelling, or define it with : or declare it with variable; fix the program and send the corrected block again'),
    (['5 0 / .'], 'error: / or MOD by zero\nthe divisor was 0: check the values that reach / or mod; fix the program and send the corrected block again'),
    (['8360 33782 lcm . cr\n\nBut I need to check it'], '7432040 \n\nerror: unknown word\na word is not defined: check its spelling, or define it with : or declare it with variable; fix the program and send the corrected block again'),
    (['variable n 7 to n n @ .'], '7'),
    ([': f 1 repeat ;'], 'error: a structure word without its opener\nfix the program and send the corrected block again'),
]


def brief(text):
    """Only the first line of an error report (the 'error: why' line)."""
    return re.sub(r"(error: [^\n]*)(\n  [^\n]*)+", r"\1", text)


def main():
    """Every program shows what ref.py prints (or the expected error) in its output block, and nothing visible before
    it; multi-block turns; stream length vs the protocol path (text_front.translate + sp.stream) where that path runs."""
    import json
    import statistics

    import text_front as F
    from forth.ref import Forth
    tok = sp.tokenizer()
    assert [tok.decode([t]) for t in (NL, TICKS, OUTPUT, TICK1, TICKNL)] == ["\n", FENCE, "output", "`", FENCE + "\n"]
    for b in ("5 .", "drop", "1 2 +", ": f 1 ;", "1 if 2"):  # every output block (prints, an error, the notes) ends
        for close in (TICKS, TICKNL):                         # '```\n'; after either closing fence: '```\n```output'
            full, owned, _ = stream(tok(f"```forth\n{b}\n", add_special_tokens=False).input_ids + [close], tok)
            text = tok.decode(full, skip_special_tokens=True)
            assert full[-1] == TICKNL and text.endswith("\n```\n") and "```\n```output\n" in text, (b, text)
    keys = {name_key(n) for n in SHORT}
    assert len(keys) == len(SHORT), "short names collide"
    sp.MAX_TOKENS = 10**6  # the protocol path's per-turn token cap is not what is compared here
    sets = {"text_vm cases": CASES, "text_front cases": [(p, Forth().run(p)) for p, _ in F.CASES]}
    if len(sys.argv) > 1:
        sets[sys.argv[1]] = [(p, Forth().run(p)) for p in json.load(open(sys.argv[1]))]
    for label, progs in sets.items():
        ok, extra = 0, []
        for p, want in progs:
            ids, owned, _ = stream(text_ids(p, tok), tok)
            got = brief(visible(ids, owned, tok))
            got = re.sub(r"(?<=```output\n)\(nothing printed[^\n]*\)(?=\n```)", "", got)  # the empty-output note
            ok += got == f"\n```output\n{want}\n```\n"
            if got != f"\n```output\n{want}\n```\n":
                print("not run:", repr(p[:60]), "| want", repr(want[:40]), "| block", repr(got[:90]))
            elif label != "text_vm cases":
                try:
                    proto, powned = sp.stream(F.translate(text_ids(p, tok), tok)[0])
                except AssertionError:
                    continue  # (the protocol path has no hoisting / top-level control)
                if sp.printed(proto, powned) == want:
                    extra.append(len(ids) - len(proto))
        print(f"{label}: {ok}/{len(progs)} as expected" + (extra and f"; extra tokens vs protocol path: median "
              f"{statistics.median(extra)}, max {max(extra)}, min {min(extra)}" or ""))
    ok = 0
    for bs, want in MCASES:
        got = ["" if o.startswith("(nothing printed") else brief(o) for o in blocks(bs, tok)]
        ok += got == want
        if got != want:
            print("multi-block:", bs, "| want", want, "| got", got)
    print(f"multi-block cases: {ok}/{len(MCASES)} as expected")
    ids = tok("Write it in a ```forth block, then ```forth run.\n```forth\n3 .\n```\n", add_special_tokens=False).input_ids
    got = re.findall(r"```output\n(.*?)\n```", tok.decode([t for t in stream(ids, tok)[0]], skip_special_tokens=True), re.S)
    assert got == ["3"], got  # prose '```forth' (no newline after it) opens no block
    ok = 0
    for bs, want in DCASES:
        got = blocks(bs, tok)[-1]
        ok += got == want
        if got != want:
            print("report:", bs, "\n  want", repr(want), "\n  got ", repr(got))
    print(f"whole-block cases: {ok}/{len(DCASES)} as expected")


if __name__ == "__main__":
    main()
