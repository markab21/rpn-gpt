"""Reference token machine for the stored-program kernel (sp_kernel.py): the exact token stream, user + kernel.

The model writes source as tokens: words are reserved tokens, literals `<lit> ['-'] digits`, names are plain text tokens
keyed by their FIRST token (any further tokens of a name are ignored: a top-level reference resolves on its first token).
  `<:>` NAME body `<;>`   definition (body stored, not run)      `<var>` NAME   variable (a memory cell)
  `<call>` NAME           call a word                           `<@>` NAME / `<!>` NAME   fetch / store a variable
The kernel then owns every token until it hands back: replayed code (op tokens, machine digits), call/jump/return
address tokens <A_x>, spill/clear segments, printed text (plain digits, '-', ' '), and DONE when a top-level
call returns (its own reserved id, never Lightning's end of turn <|im_end|>). Printed text is exactly
decode(skip_special_tokens=True) of the kernel's tokens.

Data stack: row-addressed (position p = depth - 1), rows above the top are always clean: every popped value is
cleared by re-emitting its digits in clear mode (<clr0>/<clr1>/<clr2>/<clrc> + machine digits). Results are written by
spill segments (<spl> top, <spx> scratch, <spc> variable cell). The depth is kept as lo + 128*hi with lo in [0, 256)
(bf16 residual values are exact only up to 256): the kernel emits <carry>/<borrow> before anything else it emits
once lo >= 192 / lo < 64 (hi > 0).
Overflow: a value over ND digits (a literal's digit ND + 1, a +, -, 1+, 1- result, a * step's partial product; NEGATE,
/ and MOD cannot overflow) puts the machine in a defined error state, as the capacity overflow (> NPOS entries) does:
from the token that made it, the kernel emits OVF at every position and ignores its input.
"""

import os
import re

# ---------------- token map ----------------
LIT, PR, ADD, DUP, DROP, OVER, SUB, MUL = 100, 101, 102, 103, 104, 106, 107, 116  # v2/v3 ids
(MOD, INC, DECR, ZEQ, LT, GT, PICK, DEF, SEMI, IF, ELSE, THEN, EXIT, BEGIN, WHILE, REPEAT, VAR, CALL, FETCHM,
 STOREM, SPL, SPX, SPC, CLR0, CLR1, CLR2, CLRC, END, MSTEP, CARRY, BORROW, RCARRY, RBORROW, OVF) = range(440, 474)
MD = list(range(474, 484))  # machine digits 0..9 (special: hidden by skip_special_tokens)
MMINUS = 484
NCELL, NADDR, NPOS = 8, int(os.environ.get("SP_NADDR", 192)), 1024  # variable cells, code cells, data / return stack entries
ND = int(os.environ.get("SP_ND", 20))  # digits per number (a data row's digit channels); SP_ND: reference only (the wide kernel)
QP = 10 ** (ND // 2 + 1)  # / keeps the quotient beside the remainder in the scratch row: S = Q QP + r
FETCH0, STORE0, A0 = 485, 493, 501  # <@c>, <!c> (c < NCELL), <A_x> (x < NADDR)
if NADDR != 192:  # shortcut: a bigger code store is reference-only (the kernel has 192); its <A_x> are off-vocabulary ids
    A0 = -(1 << 26)  # (negative: never text, never a real token)
TOPRET = max(200, NADDR)  # return-frame value of a top-level call: returning to it emits DONE
DONE = 693  # = 501 + 192 '<SPECIAL_693>': call finished; only Lightning ends a turn (<|im_end|> = 11)
SWAP, NEG, DIV = 105, 108, 400  # v3's ids for these words
SW2, SW3 = DONE + 1, DONE + 2  # 694, 695: SWAP's internal steps (kernel-emitted)
NM = DONE + 3  # 696 <nm>: a whole name as one kernel-emitted token (text mode, text_vm.py)
PSP = DONE + 4  # 697 <psp>: a hidden print's trailing space (text mode: prints show in the output block)
CR, PNL = DONE + 5, DONE + 6  # 698 <cr>: print a newline (text '\n', NL_TEXT); 699 <pnl>: it hidden (text mode)
NL_TEXT = 1010  # Lightning's '\n'
NXCELL = 24  # text mode: cells NCELL .. NCELL + NXCELL - 1 use <@x> / <!x> (XFETCH0 / XSTORE0 + c - NCELL)
XFETCH0, XSTORE0 = DONE + 7, DONE + 7 + NXCELL  # 700..723, 724..747
SPACEOP = XSTORE0 + NXCELL  # 748 <space>: print a space (text mode: hidden as <psp>)
EXITN = {n: SPACEOP + n - 1 for n in (2, 3, 4)}  # 749..751 <exitN>: return through N frames (exit from a hoisted helper)
DEPTH = SPACEOP + 4  # 752 <depth>: push the data stack's depth (text mode: depth, .s)
NMP = DEPTH + 1  # 753 <nmp>: a pending variable's name, fetched / stored at the next word (text mode: the word before)
HCELL0, NHCELL = 32, 11  # text mode: hidden cells 32..42 (loop i / limit per loop depth 1..4, scratch t1..t3), own tokens
HFETCH0, HSTORE0 = NMP + 1, NMP + 1 + NHCELL  # 754..764 <@h>, 765..775 <!h>: one token per hidden cell (no name)
NMTOP = HSTORE0 + NHCELL  # 776 <nmtop>: the hidden top-level definition's name (its key in the kernel's embedding)
NMSH = {w: NMTOP + 1 + k for k, w in enumerate(("gcd", "lcm", "?dup"))}  # 777..779: hidden shared words' names
LEAVES = {n: NMTOP + 4 + n for n in range(4)}  # 780..783 <lvN>: drop N frames, jump to the cell's target (text mode:
# only <lv0>, back to the innermost loop's entry <while>: leave and a begin loop's while)
ERRC = ("u+", "u-", "u0", "zero", "unknown", "up", "strings", "semi", "semiopen", "noopener", "unterm", "unstruct",
        "iout", "defin", "varword", "toval", "recurse", "rstack", "leave", "deep", "twowords", "collide", "digits", "mulop",
        "divop", "vars", "code", "prints", "tokens", "lits", "pick", "runtime", "limit", "full", "rfull", "digits2", "full2",
        "prints1")  # (digits2 full2: a popping op's depth change undone; prints1: a cr / space print)
ERRT = {k: LEAVES[3] + 1 + i for i, k in enumerate(ERRC)}  # 784..816 <err..>: the kernel stopped the block (text mode):
# a stack underflow at an op that popped / pushed / kept the depth / printed (u+ u- u0 up: the token undoes the depth
# change; up: the print is not counted), or another error class (text_vm.ERRS: its fixed line)
CST = max(ERRT.values()) + 1  # 822 <cst>: a constant's binding token (before its <!>; the dictionary's constant flag)
RBH = [CST + 1 + h for h in range(12)]  # 823..834 <rbh_h>: a bound name redeclared: its old value // 16, then
RBL = [[RBH[-1] + 1 + 16 * k + lo for lo in range(16)] for k in range(3)]  # 835..882 <rbl_k_lo>: its kind (def, var,
# const) and value % 16; the kernel cancels the old dictionary entry there (text mode, text_vm.rebind)
DOB = RBL[-1][-1] + 3  # 885 <dob>: a do loop's <begin> (text mode: the kernel counts loop depth by it)
ERRT["code2"] = DOB + 1  # 886 <err code2>: code memory full at a ':' name (no cell taken: nothing to undo)
FREE = [DOB + 2 + h for h in range(12)]  # 887..898 <free_h>: code memory from cell 16 h on is freed (text mode, a1:
# the last block's dead top-level code, emitted at the next block's first token; the kernel wipes those code heads)
TICK, TICK2 = RBL[-1][-1] + 1, RBL[-1][-1] + 2  # 883 <tick>, 884 <tick2>: text mode counts machine tokens per block as
# tc1 + 128 tc2 + 16384 tc3 (values the kernel keeps exact): <tick> when tc1 >= 128 (tc1 -128, tc2 +1), <tick2> when
# tc2 >= 128 (tc2 -128, tc3 +1), emitted where a carry could be (they are transparent the same way)
DIGITS = list(range(1048, 1058))
MINUS, SPACE = 1045, 1032
WORDS = {"+": ADD, "-": SUB, "*": MUL, "MOD": MOD, "1+": INC, "1-": DECR, "0=": ZEQ, "<": LT, ">": GT, "DUP": DUP,
         "DROP": DROP, "OVER": OVER, "PICK": PICK, ".": PR, "IF": IF, "ELSE": ELSE, "THEN": THEN, "EXIT": EXIT,
         "BEGIN": BEGIN, "WHILE": WHILE, "REPEAT": REPEAT, "SWAP": SWAP, "NEGATE": NEG, "/": DIV}
CELLOPS = {LIT, PR, ADD, DUP, DROP, OVER, SUB, MUL, MOD, INC, DECR, ZEQ, LT, GT, PICK, IF, ELSE, WHILE, REPEAT, EXIT,
           CALL, FETCHM, STOREM, SWAP, NEG, DIV, CR, SPACEOP, DEPTH} | set(EXITN.values()) | set(LEAVES.values()) | \
    set(range(HFETCH0, HSTORE0 + NHCELL))  # tokens stored as a code cell inside a definition (+ literal digits / '-')
BIN = {ADD, SUB, LT, GT}
UN = {INC, DECR, ZEQ, NEG}
MULS = (MUL, MOD, DIV)


def kappa(t):
    """Per-text-token hash codes (kappa_A < 32, kappa_B < 28); a name's key is its first token's."""
    h = (t * 2654435761 + 0x9E3779B9) & 0xFFFFFFFF
    return (h >> 8) % 32, (h >> 20) % 28


def is_text(t):
    return t >= 1000 and t not in DIGITS and t not in (MINUS, SPACE)


KEYOF = {}  # text mode (text_vm.py): stand-in name ids (>= 2^24, never real tokens) -> their key


def key(name_toks):
    """Names are keyed by their FIRST text token; the rest of a multi-token name is ignored everywhere."""
    return KEYOF.get(name_toks[0]) or kappa(name_toks[0])


def fetch_tok(c):
    return FETCH0 + c if c < NCELL else XFETCH0 + c - NCELL if c < HCELL0 else HFETCH0 + c - HCELL0


def store_tok(c):
    return STORE0 + c if c < NCELL else XSTORE0 + c - NCELL if c < HCELL0 else HSTORE0 + c - HCELL0


def cell_of(t):
    """<@c> / <!c> token -> ('fetch' | 'store', c), else None."""
    for base, n, off, kind in ((FETCH0, NCELL, 0, "fetch"), (STORE0, NCELL, 0, "store"),
                               (XFETCH0, NXCELL, NCELL, "fetch"), (XSTORE0, NXCELL, NCELL, "store"),
                               (HFETCH0, NHCELL, HCELL0, "fetch"), (HSTORE0, NHCELL, HCELL0, "store")):
        if base <= t < base + n:
            return kind, t - base + off
    return None


def mdig(v):
    return [MMINUS] * (v < 0) + [MD[int(c)] for c in str(abs(v))]


class Machine:
    """Feed tokens one at a time (user or kernel); nxt() is the kernel's next token (None = user's turn)."""

    def __init__(self):
        self.stack, self.cells, self.lo, self.hi, self.rlo, self.rhi = [], {}, 0, 0, 0, 0
        self.code, self.dict, self.nvar = [], {}, 0
        self.compile = False
        self.mode = None  # name mode after a marker: 'def' 'var' 'call' 'fetch' 'store'
        self.name = []
        self.open = self.begina = None
        self.queue = []  # pending kernel tokens of the current data run
        self.replay, self.npc, self.rs = False, 0, []
        self.jump = None  # pending control transfer: address, or 'done'
        self.lastdig = 0
        self.peak = {"data": 0, "return": 0}
        self.leaves, self.leavetgt = [], None  # text mode: <lv0> cells waiting for their loop's end, a <lvN> its <A_x>

    # ---- helpers ----
    def sp(self):
        return len(self.stack)

    def due(self):
        return (CARRY if self.lo >= 192 else BORROW if self.lo < 64 and self.hi > 0 else
                RCARRY if self.rlo >= 192 else RBORROW if self.rlo < 64 and self.rhi > 0 else
                TICK2 if self.ticking and self.tc2 >= 128 else TICK if self.ticking and self.tc1 >= 128 else None)

    ticking, tc1, tc2, tc3 = False, 0, 0, 0  # (text mode: machine tokens per block)

    def rpush(self, v):
        self.rs.append(v)
        self.rlo += 1
        assert self.rlo < 256
        self.peak["return"] = max(self.peak["return"], len(self.rs))

    def move(self, d):
        self.lo += d
        assert 0 <= self.lo < 256, "depth counter out of range (too many pushes without a kernel turn)"

    def push(self, v):
        self.stack.append(v)
        self.move(1)
        self.peak["data"] = max(self.peak["data"], self.sp())

    def pop(self):
        self.move(-1)
        return self.stack.pop()

    def cell_emit(self, c):
        op = c["op"]
        if op == "digit":
            return MD[c["v"]]
        if op == "minus":
            return MMINUS
        if op == FETCHM:
            return fetch_tok(self.dict[c["key"]])
        if op == STOREM:
            return store_tok(self.dict[c["key"]])
        return op

    # ---- kernel side ----
    def nxt(self):
        if self.err:
            return OVF  # defined error: a value over ND digits (sticky)
        t = self._nxt()
        if t is None:
            return None
        if len(self.stack) > NPOS or len(self.rs) > NPOS:
            return OVF  # defined error: capacity exceeded
        jump = t == DONE or A0 <= t < A0 + NADDR or (not self.queue and self.mode in ("fetch", "store") and self.name and
                                                      not self.compile)  # (a named cell: right after its name)
        return self.due() if self.due() and not self.segmode and not jump else t

    segmode = False  # a literal / segment / print is being written: no carry/borrow until it ends
    err = False  # a value over ND digits was made

    def _nxt(self):
        if self.queue:
            return self.queue[0]
        if self.mode in ("call", "fetch", "store") and self.name and not self.compile:
            k = key(self.name)
            return {"call": A0 + self.dict[k], "fetch": fetch_tok(self.dict[k]), "store": store_tok(self.dict[k])}[self.mode]
        if self.replay:
            if self.jump is not None:
                return DONE if self.jump == "done" else A0 + self.jump
            return self.cell_emit(self.code[self.npc])
        return None

    def feed(self, t, kernel=False):
        if self.err:
            return
        queued = kernel and bool(self.queue) and t == self.queue[0]
        if queued:
            self.queue.pop(0)
        if t in (LIT, PR, SPL, SPX, SPC, CLR0, CLR1, CLR2, CLRC):
            self.segmode = True
        elif t not in DIGITS + MD + [MINUS, MMINUS, CARRY, BORROW, RCARRY, RBORROW, TICK, TICK2]:
            self.segmode = False
        if t in FREE:
            del self.code[16 * FREE.index(t):]
            return
        if t == TICK:
            self.tc1, self.tc2 = self.tc1 - 128, self.tc2 + 1
            return
        if t == TICK2:
            self.tc2, self.tc3 = self.tc2 - 128, self.tc3 + 1
            return
        if kernel and self.ticking:
            self.tc1 += 1
        if t == CARRY:
            self.lo, self.hi = self.lo - 128, self.hi + 1
            return
        if t == BORROW:
            self.lo, self.hi = self.lo + 128, self.hi - 1
            return
        if t == RCARRY:
            self.rlo, self.rhi = self.rlo - 128, self.rhi + 1
            return
        if t == RBORROW:
            self.rlo, self.rhi = self.rlo + 128, self.rhi - 1
            return
        text = is_text(t)
        if text and self.mode:
            self.name.append(t)
            if self.compile and self.mode in ("call", "fetch", "store") and len(self.name) == 1:
                self.code[-1]["key"] = key(self.name)
            return
        if self.mode in ("def", "var") and self.name:  # first token after a definition / variable name: bind it
            self.dict[key(self.name)] = len(self.code) if self.mode == "def" else self.nvar - 1
        if not text:
            self.mode, self.name = None, []
        if t in DIGITS + MD or t in (MINUS, MMINUS):
            if t in DIGITS + MD:
                self.lastdig = (DIGITS + MD).index(t) % 10
        if self.compile:
            return self.compile_tok(t)
        if t == DEF:
            self.compile, self.mode = True, "def"
            return
        if t == VAR:
            self.nvar += 1
            self.mode = "var"
            return
        if t in (CALL, FETCHM, STOREM) and not (kernel and self.replay):
            self.mode = {CALL: "call", FETCHM: "fetch", STOREM: "store"}[t]
            if t == CALL:  # user call: return frame TOPRET
                self.rpush(TOPRET)
            return
        if A0 <= t < A0 + NADDR:
            self.replay, self.npc, self.jump = True, t - A0, None
            return
        if t == DONE:
            self.replay, self.jump = False, None
            return
        self.execute(t, kernel and self.replay and not queued)

    def compile_tok(self, t):
        if t in CELLOPS or ((t in DIGITS or t == MINUS) and self.inlit):
            if t == LIT:
                self.inlit = True
            elif t not in DIGITS and t != MINUS:
                self.inlit = False
            cell = {"op": "digit", "v": DIGITS.index(t)} if t in DIGITS else {"op": "minus"} if t == MINUS else {"op": t}
            self.code.append(cell)
            a = len(self.code) - 1
            if t in (IF, WHILE):
                self.open = a
            elif t == ELSE:
                self.code[self.open]["tgt"] = a + 1
                self.open = a
            elif t == REPEAT:
                self.code[self.open]["tgt"] = a + 1
                cell["tgt"] = self.begina
                for b in self.leaves:  # <lv0>: past this loop's end
                    self.code[b]["tgt"] = a + 1
                self.leaves = []
            elif t == LEAVES[0]:
                self.leaves.append(a)
            elif t in LEAVES.values():
                self.leavetgt = a  # its target: the <A_x> right after it
            if t in (CALL, FETCHM, STOREM):
                self.mode = {CALL: "call", FETCHM: "fetch", STOREM: "store"}[t]
            assert len(self.code) <= NADDR, f"code memory full ({NADDR} cells: fewer or shorter definitions)"
        elif self.leavetgt is not None and A0 <= t < A0 + NADDR:
            self.code[self.leavetgt]["tgt"], self.leavetgt = t - A0, None
        elif t == THEN:
            self.inlit = False
            self.code[self.open]["tgt"] = len(self.code)
        elif t == BEGIN:
            self.inlit = False
            self.begina = len(self.code)
        elif t == SEMI:
            self.code.append({"op": EXIT})
            self.compile, self.inlit = False, False
        else:
            raise ValueError(f"token {t} not allowed in a definition")

    inlit = False

    def execute(self, t, step):
        """A token outside definitions: data effect + the kernel's run (queue) + control transfer. step: a replayed
        code cell (advances the program counter)."""
        if step:
            pc = self.npc
            self.npc += 1
        s, q = self.stack, []
        if t == LIT:
            self.inlit = True
            self.push(0)
            self.litdigits = ""
            return
        if t in DIGITS + MD or t in (MINUS, MMINUS):
            if self.inlit:
                self.litdigits += "-" if t in (MINUS, MMINUS) else str(self.lastdig)
                s[-1] = int(self.litdigits) if self.litdigits != "-" else 0
                self.err = len(self.litdigits.lstrip("-")) > ND
            return
        self.inlit = False
        if t in (DUP, OVER):
            self.push(s[-1] if t == DUP else s[-2])
            q = [SPL] + mdig(s[-1]) + [END]
        elif t == PICK:
            u = self.pop()
            assert u == self.lastdig, "PICK needs its index as the literal just before it"
            self.push(s[-1 - u])
            q = [CLR0] + mdig(u) + [SPL] + mdig(s[-1]) + [END]
        elif t == DROP:
            q = [CLR1] + mdig(self.pop()) + [END]
        elif t == PR:
            v = self.pop()
            q = [MINUS] * (v < 0) + [DIGITS[int(c)] for c in str(abs(v))] + [SPACE]
        elif t in BIN:
            y, x = self.pop(), self.pop()
            r = {ADD: x + y, SUB: x - y, LT: -(x < y), GT: -(x > y)}[t]
            self.err = abs(r) >= 10**ND
            self.push(r)
            q = [CLR1] + mdig(y) + [CLR0] + mdig(x) + [SPL] + mdig(r) + [END]
        elif t in UN:
            x = self.pop()
            r = {INC: x + 1, DECR: x - 1, ZEQ: -(x == 0), NEG: -x}[t]
            self.err = abs(r) >= 10**ND
            self.push(r)
            q = [CLR0] + mdig(x) + [SPL] + mdig(r) + [END]
        elif t == SWAP:  # = OVER, then <sw2> <sw3>: a b -> a b a -> b _ a -> b a
            self.push(s[-2])
            q = [SPL] + mdig(s[-1]) + [SW2]
        elif t == SW2:  # pop 2: row of a := b, row of b cleared; the copy of a stays above the new top
            a, b, a2 = s[-3:]
            self.hold = a2
            s[-3:] = [b]
            self.move(-2)
            q = [CLR0] + mdig(a) + [SPL] + mdig(b) + [CLR1] + mdig(b) + [SW3]
        elif t == SW3:  # push the held copy: clear its row, write it one row lower
            self.push(self.hold)
            q = [CLR1] + mdig(self.hold) + [SPL] + mdig(self.hold) + [END]
        elif t in MULS:
            y, x = s[-1], s[-2]
            self.move(-1)  # depth counter drops at the op token; values stay until the final step
            self.mop = (t, x, y, 0, 0)
            q = self.mstep()
        elif t == MSTEP:
            q = self.mstep()
        elif cell_of(t) and cell_of(t)[0] == "fetch":
            self.push(self.cells.get(cell_of(t)[1], 0))  # a never-stored cell reads 0 (its empty row), as in ref.py
            q = [SPL] + mdig(s[-1]) + [END]
        elif cell_of(t):
            c, v = cell_of(t)[1], self.pop()
            old = self.cells.get(c)
            self.cells[c] = v
            q = [CLRC] + (mdig(old) if old is not None else []) + [SPC] + mdig(v) + [CLR1] + mdig(v) + [END]
        elif t == CR:
            q = [NL_TEXT]
        elif t == DEPTH:
            self.push(len(s))
            q = [SPL] + mdig(s[-1]) + [END]
        elif t == SPACEOP:
            q = [SPACE]
        elif t in (IF, WHILE):
            f = self.pop()
            q = [CLR1] + mdig(f) + [END]
            if step and f == 0:
                self.jump = self.code[pc]["tgt"]
        elif t in (ELSE, REPEAT) and step:
            self.jump = self.code[pc]["tgt"]
        elif t == CALL and step:
            self.rpush(self.npc)
            self.jump = self.dict[self.code[pc]["key"]]
        elif t in LEAVES.values() and step:  # leave: drop N helper frames, jump past the loop's end
            for _ in range(t - LEAVES[0]):
                self.rs.pop()
                self.rlo -= 1
            self.jump = self.code[pc]["tgt"]
        elif (t == EXIT or t in EXITN.values()) and step:
            for _ in range(1 if t == EXIT else t - SPACEOP + 1):
                r = self.rs.pop()
                self.rlo -= 1
            self.jump = "done" if r == TOPRET else r
        elif t in (SPL, SPX, SPC, CLR0, CLR1, CLR2, CLRC, END, SPACE):
            return
        self.queue = q

    def mstep(self):
        """One digit of |x| per step through the scratch row: * r = 10 r + x_j |y|; MOD r = (10 r + x_j) mod |y|;
        / keeps the quotient beside the remainder in the scratch row, S = Q 10^11 + r (|x|, |y| <= 10 digits)."""
        op, x, y, j, r = self.mop
        dx = str(abs(x))
        last = j == len(dx) - 1
        rold = r
        if op == MUL:
            r = 10 * r + int(dx[j]) * abs(y)
            self.err = r >= 10**ND
        else:
            t = 10 * (r % QP) + int(dx[j])
            r = t % abs(y) + (10 * (r // QP) + t // abs(y)) * QP * (op == DIV)
        self.mop = (op, x, y, j + 1, r)
        q = ([CLR2] + mdig(rold)) if j else []
        if not last:
            return q + [SPX] + mdig(r) + [MSTEP]
        sgn = -1 if (x < 0) != (y < 0) else 1
        res = r * sgn if op == MUL else r // QP * sgn if op == DIV else r * (-1 if x < 0 else 1)
        self.stack[-2:] = [res]
        return q + [CLR1] + mdig(y) + [CLR0] + mdig(x) + [SPL] + mdig(res) + [END]


# ---------------- driving ----------------
def run(machine, user_runs):
    """user_runs: list of user token lists; after each, the kernel runs until it hands back. -> [(user, kernel)]."""
    out = []
    for user in user_runs:
        for t in user:
            machine.feed(t)
        model = []
        while (t := machine.nxt()) is not None:
            model.append(t)
            machine.feed(t, kernel=True)
            assert len(model) < 10**6
            if t == OVF:
                break  # the kernel repeats OVF forever
        out.append((user, model))
    return out


def printed(runs, tok):
    return tok.decode([t for _, m in runs for t in m], skip_special_tokens=True)


def source_tokens(src, tok, case=str.upper):
    """Forth source (ref.py dialect) -> kernel source tokens. Names are tokenized as ' ' + case(NAME); a top-level
    reference is its first token only (the kernel takes over right after it, so the rest is never written)."""
    words = re.sub(r"\( [^)]*\)", " ", src).split()
    name = lambda w: tok(" " + case(w), add_special_tokens=False).input_ids
    out, i, depth = [], 0, 0
    while i < len(words):
        w = words[i].upper()
        nx = words[i + 1] if i + 1 < len(words) else None
        ref = lambda w: name(w)[:1] if not depth else name(w)
        if w == ":":
            out += [DEF] + name(nx)
            depth, i = 1, i + 2
            continue
        if w == "VARIABLE":
            out += [VAR] + name(nx)
            i += 2
            continue
        if w == ";":
            out.append(SEMI)
            depth = 0
        elif re.fullmatch(r"-?\d+", w):
            out += [LIT] + [MINUS] * w.startswith("-") + [DIGITS[int(c)] for c in w.lstrip("-")]
        elif w in WORDS:
            out.append(WORDS[w])
        elif nx in ("@", "!"):
            out += [FETCHM if nx == "@" else STOREM] + ref(words[i])
            i += 2
            continue
        else:
            out += [CALL] + ref(words[i])
        i += 1
    return out
