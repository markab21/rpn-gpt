"""Stored-program Forth kernel compiled into a stock NemotronH (token protocol + reference machine: sp_vm.py).

The model writes source as tokens (definitions, variables, literals, calls); the kernel stores definitions in Mamba
state at definition time (code cells, name dictionary, back-patched jump targets) and runs calls by REPLAY: it emits
each code cell as a token, executes it when fed back, and keeps return addresses in a row-addressed return stack.
Data stack: row-addressed (1024 entries, 8 heads x 128 rows), every popped value cleared by re-emitting its digits in
clear mode; * and MOD run digit-serially (one <mstep> per digit of x through a scratch row); / runs as MOD with the
quotient kept beside the remainder in the scratch row (Q 10^11 + r); SWAP = OVER, then kernel-emitted <sw2> (pop 2:
the lower row gets b, b's row is cleared) and <sw3> (push the held copy one row lower); NEGATE copies with the sign
flipped. A name is keyed by its first text token (PREVTXT / FIRSTTXT); later tokens of a name do nothing.

Conventions (as s2/v2): residual feature f is read by every layer as f (rms of the 2^20 ballast dim = 2^14 = SCALE);
per-group ballast head makes every gated norm a fixed 2^-51; every head has A_log = -60, so dt <= 4096 decays by 1
and the only wipe, dt = 2^94, gives |dt*A| = 173 (exp = 0 in fp32), the vLLM-safe range. No residual value above 256
(bf16): depths are lo + 128 * hi with kernel-emitted carry / borrow tokens.
Data store row (MSD-first): channels 0-19 digits (x = 32 + d), 20-39 marker (x = 32), 40 sign. Write: dt 32, B 32.
Clear: dt 4096, B = silu(-BETA) = -1/4 (cancels the write exactly).

Layers (PLAN): L0 registers | L1 MLP compile-cancel | L2 counters | L3 address decode 1 + compile registers |
L4 address decode 2 + code / return addresses | L5 data store (3 read replicas) | L6 align + control | L7 add/sub g/p,
products d*y, patch row | L8 MLP carries | L9 result select + return stack + branch latch + d*y g/p | L10 MLP d*y carries
| L11 T -/+ d*y g/p | L12 MLP borrows / carries | L13 * / MOD step + code memory | L14 result register + name key |
L15 zero / length flags + dictionary | L16 MLP length | L17 print position + jump / cell emission | L18 digit select |
L19 MLP emission logic | L20 digit decode | L21 attention (inert, generate()'s cache needs one).
Text front end (text_vm.py; piece 2a: fence, numbers, single-token words, hidden prints, output block): two layers
before L0 (F0: fence state FON, its opening 'forth'; F1: text tokens inside the fence made neutral for the kernel, the
closing '```') and one MLP after L20 (FIN: hidden prints, the output block, carries only before other kernel output);
the rest sits in spare heads / units of L0-L20 (word registers L2, closed-word delays L3, digit registers L4, print
memory L7, close-time emission L19). Stateless (D-skip) heads never share a group with B / C rows: prefill clamps dt
to time_step_min, so such a head would accumulate state there.
Piece 2b (multi-token words, more built-in words, names, definitions, variables, comments; text_vm.py): a word's key
(text_vm.name_key) is summed over its tokens (L4 left-edge pair lookup, L6 sums, L7 at the close, L8 / L9 mod-P decodes
into the dictionary read row / column); L11 classifies the closed word (built-in by key + length + last char, number,
CLT's empty word) and latches KCUR (the last nonempty word's key; KPREV at L13 the one before); a name is one <nm>
(KCUR) / <nmp> (KPREV: a pending variable) token that drives the protocol name machinery (PENDA at L13, key decodes at
L16); the dictionary (L17) also holds each name's kind, length and last char, read at the close; L13-L20 latch the
word's class, decl word before (EXPK), pending variable (VCUR), comment state (LC, COMMENT); L21 emits the translation
table (tr_rules) at position TPC while the machine is silent (L23 suppresses it otherwise). <cr> / <space> are one-char
prints; <depth> pushes the depth (< 192).
Piece 2c (text_vm.py; one level of control flow): a fence opens at the line start after ```forth (FON / FOPEN: conv
taps t-2 / t-1 + conv bias); comment state ends with its block. if else then begin while repeat until again do loop
+loop i exit are op sequences in place (CTLSEQ); the loop cells and rot -rot min max /mod */'s scratch are hidden cells
32..36 run by their own tokens (<@h> <!h>: code cells, l_cancel; l_addr1's cell decodes); a top-level opener first
opens the hidden top-level definition (<nmtop>, TOPOPEN), closed with ; + its call at : variable value constant or at the
block's end phase (l_f_final feeds its ; once the block's work is done, EJ counts the rest); gcd lcm ?dup are hidden
words defined at first use (top level: in place; in a definition: after its ;, USED / PENDL; DECL sticky). A hidden name
token carries its key in KCUR's place (the latched KCUR gated off at E_NMH). Positions: TPL / TPH (TPC % 8, // 8), up
to 64. Word classes over two heads (WCOH).
Piece 2d-minimal (text_vm kerr): a stack underflow (l_e7: the depth after the op, per-op shift), / or MOD by zero (l_e9)
or an unknown word (a rule) emits <err..> (undoing the op's depth change) and the kernel says nothing more (SUPP: an lm
bias on its hidden tokens) until the output block, which ends with 'error: <why>' (l_e22); the next block starts with
the reset (RST): <done> (a replay), ; (a half-written definition), <drop>s (the stack).
Piece 2d (rebind, limits; text_vm): a bound name declared again (KVR at its close) is preceded by <rbh_hi> <rbl_kind_lo>,
its old dictionary value / kind (DVR, latched at the close, decoded at L20); the dictionary cancels the add-only entry at
<rbl> (dt 4096, B -1/4: CDHROW / CDCOL from KCUR); a constant binds at <cst>. 32 named variable cells (<@x> <!x> for 8..).
Errors at the failing token, each undoing its op's counter change (u+ u- ... the 'undo' table in base): toval (a rule),
mulop / divop (l_e11), + - 1+ 1- over 20 digits (OVX in text mode, l_e13), * by digit count (l_e11), a literal over 20
digits (DIG21, a rule), code memory full (CFULL: the cell is not written, WA undone), the 129th print (NPRF), a data /
return stack overflow (FULLP / FULLM / RFULL, l_ctrl6; the return stack is emptied at each block start), the token cap
(machine tokens MACH counted in TC1 + 128 TC2 + 16384 TC3 with hidden <tick> / <tick2> where a carry could go; at TC3 = 12
the next replayed cell is the error). OVFO stays for protocol mode only. Output blocks end with '```\n' (one token).
Residual 6144 (BAL2, BAL3: ballast).
End phase digits: a literal compiled after the closing ``` (FON off: the end phase's hidden gcd / lcm, abs's 0) compiles
its machine digits too (ENDPH, L2: from the closing ``` to the next text token); before, the 0 ran as data: wrong values.
a1 (text_vm.freeat): a block whose last top-level hidden definition ran with nothing compiled after it and no error
frees its code from 16 * ceil(first cell / 16): at the next block's first token (FOPEN, as RST) the kernel emits
<free_h> (FREEOK, L17, from ENDGO until a DEF / an error / <free_h>; HCA latched at <nmtop>; WAH >= h), which sets WA
= 16 h (WA's counter: wipe + set) and wipes the code heads >= h (FTH, l_code: dt BIG).
Nesting (text_vm: in place): CD counts the structures open while compiling (L4); its level one-hots (LVW written now,
LVC closed now, L7) gate per-level registers in their own layer (l_nest: wiped only where an event AND its level is on,
dt 2^93): the if / while cell, the else cell, the <begin> address, loop / do flags; l_ctrl6 selects the patch row and
<repeat>'s / <lv0>'s targets from them (<lv0>: the innermost loop's <begin>, back to its entry <while>). Each level's
kind (if else begin while do) for the closers' checks (KDL, l_kd17; the innermost: SKOH); do loops open (LDC) pick the
loop cells (3 deep); 4 structures open (CDFC): 'deep'.
Overflow (sp_vm.py): OVX = a value over ND digits made at this token (L3 a literal's digit ND + 1, L9 an addition's
carry out of the top digit, L13 a * step's carry out of the top digit or T's dropped digit ND), latched at L14 into
OVFO: OVF at every position from then on.
Run on the GPU box: python sp_test.py OUT all   (and sp_vllm.py OUT eager|compile all; text mode: sp_test.py OUT text
[programs.json])
"""

import itertools
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from mamba_kernel_test import ALPHA, BETA  # bf16(silu(ALPHA)) = 1/4, bf16(silu(-BETA)) = -1/4
from sp_vm import *  # noqa: F403 (token map)
import text_vm as TV  # text mode reference (word table, name keys)

LIGHTNING = os.environ.get("SP_LIGHTNING", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16")  # (SP_LIGHTNING: its config + tokenizer elsewhere)
HIDDEN, H, G, P, N = 6144, 64, 8, 64, 128  # (6144 = 4096 + 2048: two more ballast dims keep every norm's rms 2^14)
I = H * P
XO, BO, CO, DO = I, 2 * I, 2 * I + G * N, 2 * I + 2 * G * N
BAL, SCALE, BIG = 0, 2.0**14, 2.0**94
WIDTH = 4096
ND = 20  # digits per number
NPOS = 1024  # data stack positions (8 heads x 128 rows); head 8 = variable cells
MA, ML, MP = "linear_attention", "mlp", "full_attention"  # transformers >= 5.16 names
_n = [1]


def dims(w=1):
    d = _n[0]
    _n[0] += w
    return d


def heads(groups):
    return [8 * g + i for g in groups for i in range(7)]


# ---------------- residual features ----------------
ONE = dims()  # constant 1 (second bias input: keeps every bias weight bf16-exact)
BAL2, BAL3 = dims(), dims()  # more ballast, 2^19 each, never read: rms = sqrt((2^40 + 2 * 2^38) / 6144) = 2^14
(F_NRD, F_RDIG, DV, F_MINUS, F_LIT, F_PR, F_DUP, F_DROP, F_OVER, F_PICK, F_ADD, F_SUB, F_MUL, F_MOD, F_INC, F_DECR,
 F_ZEQ, F_LT, F_GT, F_IF, F_ELSE, F_WHILE, F_REPEAT, F_EXIT, F_CALL, F_FETCH, F_STORE, FV, F_SPL, F_SPX, F_SPC, F_CLR0,
 F_CLR1, F_CLR2, F_CLRC, F_END, F_MSTEP, F_CARRY, F_BORROW) = (dims() for _ in range(39))
F_SWAP, F_SW2, F_SW3, F_NEG, F_DIV = (dims() for _ in range(5))  # '/' also carries F_MOD: it runs as MOD
F_CR, F_SPACEOP, F_DEPTH = (dims() for _ in range(3))  # <cr> <space> (one-char prints), <depth> (a push)
F_LV, F_DOB = dims(), dims()  # <lv0>: a jump back to the innermost loop's entry <while>; <dob>: a do loop's <begin>
TOKF = {LIT: F_LIT, PR: F_PR, DUP: F_DUP, DROP: F_DROP, OVER: F_OVER, PICK: F_PICK, ADD: F_ADD, SUB: F_SUB, MUL: F_MUL,
        MOD: F_MOD, INC: F_INC, DECR: F_DECR, ZEQ: F_ZEQ, LT: F_LT, GT: F_GT, IF: F_IF, ELSE: F_ELSE, WHILE: F_WHILE,
        REPEAT: F_REPEAT, EXIT: F_EXIT, CALL: F_CALL, SPL: F_SPL, SPX: F_SPX, SPC: F_SPC, CLR0: F_CLR0, CLR1: F_CLR1,
        CLR2: F_CLR2, CLRC: F_CLRC, END: F_END, MSTEP: F_MSTEP, CARRY: F_CARRY, BORROW: F_BORROW, SWAP: F_SWAP,
        SW2: F_SW2, SW3: F_SW3, NEG: F_NEG, DIV: F_DIV, CR: F_CR, SPACEOP: F_SPACEOP, DEPTH: F_DEPTH, LEAVES[0]: F_LV}
TOKF3 = {}  # filled after the stage-3 dims exist
CELLF = [F_LIT, F_PR, F_DUP, F_DROP, F_OVER, F_PICK, F_ADD, F_SUB, F_MUL, F_MOD, F_INC, F_DECR, F_ZEQ, F_LT, F_GT, F_IF,
         F_ELSE, F_WHILE, F_REPEAT, F_EXIT, F_CALL, F_FETCH, F_STORE, F_SWAP, F_NEG, F_CR, F_SPACEOP,
         F_DEPTH, F_LV]  # executed code cells (op tokens)
OPS = CELLF + [F_MSTEP, F_SW2, F_SW3]
STARTS = [F_SPL, F_SPX, F_SPC, F_CLR0, F_CLR1, F_CLR2, F_CLRC]
KINDS = {"CP": [F_DUP, F_OVER, F_FETCH, F_DEPTH], "PICK": [F_PICK], "DROP": [F_DROP, F_IF, F_WHILE],
         "BIN": [F_ADD, F_SUB, F_LT, F_GT], "UN": [F_INC, F_DECR, F_ZEQ, F_NEG], "ST": [F_STORE], "MM": [F_MUL, F_MOD],
         "MS": [F_MSTEP], "SW1": [F_SWAP], "SW2": [F_SW2], "SW3": [F_SW3]}
PUSH = [F_LIT, F_DUP, F_OVER, F_FETCH, F_SWAP, F_SW3, F_DEPTH]
POP = [F_DROP, F_PR, F_ADD, F_SUB, F_LT, F_GT, F_IF, F_WHILE, F_STORE, F_MUL, F_MOD]
RESULT = [F_DUP, F_OVER, F_PICK, F_FETCH, F_PR, F_ADD, F_SUB, F_LT, F_GT, F_INC, F_DECR, F_ZEQ, F_STORE, F_MUL, F_MOD,
          F_MSTEP, F_SWAP, F_SW2, F_SW3, F_NEG, F_DEPTH]
COPY = [F_DUP, F_OVER, F_PICK, F_FETCH, F_PR, F_STORE, F_SWAP, F_SW2, F_SW3]

# L0 registers / counters
INLIT, PMODE, M_SPL, M_SPX, M_SPC, M_CLR0, M_CLR1, M_CLR2, M_CLRC, SEGST, LASTDIG, VARC, J = (dims() for _ in range(13))
K = {k: dims() for k in KINDS}
RMODES = [PMODE, M_SPL, M_SPX, M_SPC]  # segments emitted from the result register
EMODES = [M_CLR0, M_CLR1, M_CLR2, M_CLRC]  # clear segments, emitted from the store row (role X)
CLEARM = EMODES + [PMODE]  # store writes in clear mode (a print clears the popped row)
MODES = [INLIT] + RMODES + EMODES
# L2 counters
LO, HI, MJ = dims(), dims(), dims()
# L3/L4 addresses: per role row one-hot (128), block offset, head one-hot (9)
ROLES = ("X", "Y", "Z", "W")
ROW = {r: dims(N) for r in ROLES}
BOFF = {r: dims() for r in ROLES}
HD = {r: dims(9) for r in ROLES}
CROW = dims(N)  # write row in clear mode
WCH, JH = dims(ND), dims(ND + 1)
CDUE, BLO, BHI, CARRYO, BORROWO = (dims() for _ in range(5))
# store reads (MSD-first digits, thermometer, sign) and aligned (LSD-first)
RA = {r: dims(ND) for r in "XYZ"}
RW = {r: dims(ND) for r in "XYZ"}
RS = {r: dims() for r in "XYZ"}
AL = {r: dims(ND) for r in "XYZ"}
SELE, EWJ = dims(), dims()
R, RSG = dims(ND), dims()  # result digits (LSD-first) and sign
RV, RVS = dims(ND), dims()  # result register
ZC, ZERO, FINAL, FINR = dims(ND), dims(), dims(), dims()  # FINR: FINAL latched at the step token
LR, POS, ANYPOS, DONER, SELD = dims(), dims(ND), dims(), dims(), dims()
O = {k: dims() for k in ("MINUS", "MMINUS", "SPACE", "END", "MSTEP", "SPL", "SPX", "SPC", "CLR0", "CLR1", "CLR2",
                         "CLRC", "PD", "MDG", "SW2", "SW3", "CRNL")}
PDIG, MDIG = dims(10), dims(10)
# stage 2: add / sub / compare (v2's one-pass g/p decode + carry MLP)
GA, PA, G1, P1, G2, CA, B1, B2 = (dims(ND) for _ in range(8))  # CA/B1/B2[i] = carry/borrow out of position i
EFSB, ADDB, RSB, ZX = (dims() for _ in range(4))
LTM, GTM = B1 + ND - 1, B2 + ND - 1  # |x| < |y|, |x| > |y|
ADDL, SUBL = [F_ADD, F_INC], [F_SUB, F_DECR]
BINA = ADDL + SUBL
# stage 3: stored program
(F_TXT, F_NTXT, KA, KB, F_DEF, F_SEMI, F_THEN, F_BEGIN, F_VAR, F_FETCHM, F_STOREM, F_A, AV, F_DONE, F_RCARRY,
 F_RBORROW) = (dims() for _ in range(16))
DIGH = dims(10)
COMPILE, NM_DEF, NM_VAR, NM_CALL, NM_FETCH, NM_STORE, REPLAY, BASE, NHA, NHB, K_IFW = (dims() for _ in range(11))
NHK = 9  # hidden cells the kernel runs (32 33 / 37 38 / 39 40: i, limit at loop depth 1..3; 34..36 scratch):
# <@h> / <!h> tokens
HCT = [HFETCH0 + k for k in range(NHK)] + [HSTORE0 + k for k in range(NHK)]
CODES = [LIT, PR, ADD, SUB, MUL, MOD, INC, DECR, ZEQ, LT, GT, DUP, DROP, OVER, PICK, IF, ELSE, WHILE, REPEAT, EXIT, CALL,
         FETCHM, STOREM, MINUS, SWAP, NEG, DIV, CR, SPACEOP, DEPTH, LEAVES[0]] + [("d", d) for d in range(10)] + HCT
CI = {c: i for i, c in enumerate(CODES)}
OPC, OPN = dims(len(CODES)), dims(len(CODES))  # op code of the cell being stored / of the next cell (replay)
STEPW, DNDEF, DNVAR, CNAME, UN_CALL, UN_FETCH, UN_STORE = (dims() for _ in range(7))
RLO, RHI, STEPS, WA, NVAR = (dims() for _ in range(5))
OPEN, BEGINA, PENDA, PENDB, PREVDEF, PREVVAR, PREVEXIT, OPEND = (dims() for _ in range(8))
NL, NH_ = 16, NADDR // 16  # code address = 16 * head + row
PCL, PCH, NPCL, NPCH, WAL, WAH, PATCHL, PATCHH = (dims(n) for n in (NL, NH_) * 4)
RROW = {k: dims(N) for k in "WRC"}  # return-stack row: write (call), read (exit), clear (token after exit)
RBOFF = {k: dims() for k in "WRC"}
RHD = {k: dims(8) for k in "WRC"}
RCDUE, RBLO, RBHI, RETV, CALLR, DVAL, PROW, RET, BR, HSA, HSB, TGT, IFJ, DVCALL, DVCELL = (dims() for _ in range(15))
NHROW, NB1, NB0 = TV.P1, 19, 6  # name key (a, b): dictionary row a, column b = NB0 * b1 + b0 (head b1, channels b0)
DHROW, DCOL, HROW, HCOL = dims(NHROW), dims(NB1 + NB0), dims(NHROW), dims(NB1 + NB0)  # write / read row, column
NVC = NCELL + NXCELL  # named variable cells (text mode: 32; cells 8.. use <@x> / <!x>)
AOH, DONEO, FETCHO, STOREO = dims(NADDR), dims(), dims(NVC), dims(NVC)
RCARRYO, RBORROWO, HI7, HI8, LO129, LO1, RHI7, RHI8, RLO129, RLO1, OVFO = (dims() for _ in range(11))
# stage 4: * and MOD, digit-serial (one <mstep> per digit of x, scratch row at depth + 2)
K_MOD, XJ, XJM = dims(), dims(), dims()
K_DIV, DIVFIN = dims(), dims()  # '/' (runs as MOD) and its last step
PREVTXT, FIRSTTXT = dims(), dims()  # previous token is text; first text token of a name (its key)
MJH, XJH = dims(ND), dims(10)
S = {d: dims(11) for d in range(1, 10)}  # digits of d * |y| (|y| <= 10 digits)
GS, PS, BG, BP, AG, AP = ({d: dims(11) for d in range(1, 10)} for _ in range(6))
APH = dims(ND)  # [T_i == 9] for i >= 11
BB = {d: dims(11) for d in range(1, 10)}  # borrow out of position i of T - d|y|
CC = {d: dims(ND) for d in range(1, 10)}  # carry out of position i of T + d|y|
OVX, ZOV = dims(), dims()  # a value over ND digits made at this token; [T's digit ND (dropped) != 0]
STEPT = [F_MUL, F_MOD, F_MSTEP]  # tokens that compute a step
CALLNESS = [CALLR, UN_CALL]
CELLNESS = [OPN + CI[FETCHM], OPN + CI[STOREM], UN_FETCH, UN_STORE]
JMPF = [F_ELSE, F_REPEAT, F_EXIT, CALLR, UN_CALL, IFJ, F_LV]
PATCHF = [F_THEN, OPC + CI[ELSE], OPC + CI[REPEAT]]
# stage 5: plain-text front end (text_vm.py), piece 2a: fence, numbers, single-token words, hidden prints, output block
(E_T, E_TICK, E_FORTH, E_SW, E_EW, E_NE, E_DIG, E_DV, E_MIN, E_ISLIT, E_ISMD, E_ISMM, E_PSP, E_PCODE, E_NL,
 E_OUTPUT, E_SPACE, E_VDM, E_NZ, E_ZD) = (dims() for _ in range(20))  # embedding (never cancelled)
FON, FOPEN, CLT, EWR, EWP, CL, WRES, WNEW, MTNE, SND, IND, SMIN, IMIN, DIGMT, PCHAR = (dims() for _ in range(15))
W_ND, NMIN, W_NE, NPR, JP, QP, QJ, NLC, NLCK, FTICK, BLKP = (dims(4 if i == 8 else 1) for i in range(11))
CW_ND, CW_NEG, CW_NE, EMW_ND, EMW_NEG, CWZ, JLT, LASTP, MV, UNW, FLD = (dims() for _ in range(11))
NZD, INZ, W_NZ, CW_NZ, EMW_NZ, EMWZ = (dims() for _ in range(6))  # nonzero digits: '-0' is written as 0
ZDC, IZD, ZNZ, LZINC, LZ, CW_LZ, CW_Z, EMW_S = (dims() for _ in range(8))  # leading zeros are skipped
JS = dims(ND)  # word digit to emit next: J + EMW_S
WR, DREG, CJ, CODE = dims(ND), dims(ND), dims(22), dims(13)  # print char codes: digit, '-' 10, ' ' 11, '\n' 12
NPRINT = 128  # prints per output block (print memory rows)
RIDX = dims(NPRINT)
FE_NL, FE_TICKS, FE_OUTPUT, O_PSP, O_PNL, FE_TICKNL = (dims() for _ in range(6))
E_TNL, CLNL, PRE, KTICK = (dims() for _ in range(4))  # a token ends with '\n'; the closing ``` did (no leading '\n')  # (FE_TICKNL: the block's last '```\n')
TXT_NL, TXT_TICKS, TXT_OUTPUT = 1010, 1975, 10114  # Lightning's '\n' '```' 'output' (text_vm.NL / TICKS / OUTPUT)
# stage 6: text front end, piece 2b (multi-token words, built-in words, names, definitions, variables, comments)
RUNE = ("mulop", "divop")  # errors at a running * / MOD op (its pop undone, as / or MOD by zero)
EALIAS = {"digits2": "digits", "full2": "full", "prints1": "prints", "code2": "code"}  # error tokens with another undo:
# their class's line (code2: at a ':' name, no cell to undo)
ENOW = ("u+", "u-", "u0", "up", "zero") + RUNE + ("digits2", "code", "prints", "prints1", "full", "full2", "rfull") + \
    ("digits",)  # error tokens emitted at the failing token (not by a translation rule): never SUPP-suppressed
TCHARS = sorted({chr(c).lower() for c in range(33, 127)})  # chars a name's tokens may meet at (all printable)
CHC = {c: i for i, c in enumerate(TCHARS)}  # first-char one-hot index (E_FQ)
CHCA = {c: i + 1 for i, c in enumerate(sorted({chr(c).lower() for c in range(33, 127)}))}  # last-char code (0: none)
NCH = len(TCHARS)
B2W = ("+ - * / mod . dup drop over pick swap negate < > 1+ 1- 0= = <> <= >= 0<> 0> 0< not invert true false and or "
       "2dup nip tuck 2* 2/ abs 2drop depth cr space").split()  # built-in words: their op sequences (text_vm.WORDS)
SPECW = [":", "to", "variable", "value", "constant", ";", "@", "!", "+!", "(", "\\"]  # decl words first (EXPK 1..5)
CTLW = "if begin do else then while repeat until again loop +loop i exit leave".split()  # control (not shadowed)
SCRW, SHWC = ["rot", "-rot", "min", "max", "/mod", "*/"], ["gcd", "lcm", "?dup"]  # 2c: built from ops / hidden words
KCLS = ("underflow", "zero", "unknown", "strings", "semi", "semiopen", "noopener", "unterm", "unstruct", "iout", "defin",
        "varword", "toval", "mulop", "divop", "digits", "code", "prints", "full", "rfull", "tokens", "deep", "leave",
        "vars", "collide")  # (text_vm.ERRS; collide: a control word declared as a name)
STRW = ['."', ".(", "type", 's"', "emit"]  # (errors: not supported)
WCLS = B2W + SPECW + CTLW + SCRW + SHWC + STRW + ["#", ""]  # word classes (code 1 + index): built-ins, special words, a
WC = {w: i for i, w in enumerate(WCLS)} | {"?do": WCLS.index("do")}  # number, CLT's empty word (?do: do's class)
NUMC, EMPTYC = WC["#"] + 1, WC[""] + 1
LOOPI, LOOPL = HFETCH0 - HCELL0, HSTORE0 - HCELL0  # cell c's <@h> / <!h>: LOOPI + c / LOOPL + c
CTLSEQ = {"if": [IF], "else": [ELSE], "then": [THEN], "exit": [EXIT]} | TV.LOOPS  # text_vm.control (in place)
LSEQ = {d: {"i": [LOOPI + i], "do": [LOOPL + i, LOOPL + m, LOOPI + i, LOOPI + m, SUB, ZEQ, ZEQ, DOB, WHILE],
            "loop": [LOOPI + i, INC, LOOPL + i, LOOPI + i, LOOPI + m, LT, REPEAT],
            "+loop": [LOOPI + i, LOOPI + m, LT, SWAP, LOOPI + i, ADD, DUP, LOOPL + i, LOOPI + m, LT, SUB, ZEQ, REPEAT]}
        for d, (i, m) in TV.LOOPC.items()}  # do loops by loop depth (their hidden i / limit cells)
SHDEF = {w: [DEF, NMSH[w]] + TV.SHARED[w] + [SEMI] for w in SHWC}  # a hidden shared word's definition
TRTOK = sorted({t for w in B2W for t in TV.WORDS[w]} | {LIT, DEF, SEMI, VAR, CALL, FETCHM, STOREM, NM, NMP, ADD} |
               {t for q in CTLSEQ.values() for t in q} | {t for w in SCRW for t in TV.scratch_seq(w)} |
               {t for q in LSEQ.values() for qq in q.values() for t in qq} |
               {t for q in SHDEF.values() for t in q} | {NMTOP} | {ERRT[c] for c in KCLS[2:]} |
               {CST, ERRT["code2"]} | set(RBH) | {t for r in RBL for t in r} | set(FREE))
NTPC, NNOTE = 64, 20  # translation positions 0..63 (TPL / TPH: position % 8, // 8); the empty-output note's tokens
(E_LCC, E_LEN, E_HNL, E_NXS, E_NMT, E_NMPT, E_TROP, E_PNL, E_NOO) = (dims() for _ in range(9))
E_FQ, E_K1, E_INT = dims(NCH), dims(6), dims(6)  # first char; key parts (keyparts): 1-token key, internal pairs
LCR, LCP, HNLR, EHNLP, PREVOUT, MT = (dims() for _ in range(6))
START, CONT, CONTNX, TT, NLQ = (dims() for _ in range(5))
NKP = 6  # key parts (keyparts: base 8)
EOUT, LENC, SACC, SLEN, WNX = dims(NKP), dims(), dims(NKP), dims(), dims()
CSACC, CWLEN, CWNX, CLNE, CLX, CLNEP = dims(NKP), dims(), dims(), dims(), dims(), dims()
MAB, KEYA, KEYB = dims(2), dims(), dims()
BWC, DECLC, BSLF, RPW = (dims() for _ in range(4))
KCUR, KPREV = dims(4), dims(2)  # KCUR: a, b, length, last char of the last nonempty word; KPREV: the word before
LC, LCSET, INLCR, WCC, CCLS, EXPK, TPC = (dims() for _ in range(7))
TPL, TPH, WCOH, EXPKOH = dims(8), dims(8), dims(len(WCLS)), dims(5)
EVT, EVTP, COMMENT, CMB = (dims() for _ in range(4))
KDEF, KVARX, KCONST, DVLEN, DVLCP, KV = (dims() for _ in range(6))  # KV: the entry's length / last char match
KINDR, VAL, VALR, VCUR, KVR = dims(3), dims(), dims(), dims(), dims()
TRD, NOTED = dims(len(TRTOK)), dims(NNOTE)
EMPTYB, NOTEJ, LEF, LEP, K_CR, K_SP = (dims() for _ in range(6))
# stage 7: text front end, piece 2c (control flow, hidden cells, hidden top-level / shared words)
E_SNL, E_NMH, E_NMTOPF, E_NLX = (dims() for _ in range(4))  # a token starting with '\n'; a hidden name; <nmtop>
E_HC, E_NMSHF = dims(len(HCT)), dims(3)  # each hidden cell token; <nmgcd> <nmlcm> <nm?dup>
DNSH, USE, DECL, USEDSET, USED, PENDL, DECLL = (dims(3) for _ in range(7))  # per hidden shared word
DNTOP, DNUSR, TOPOPEN, TOPC, CMPC = (dims() for _ in range(5))
ENDGO, ENDACT, TTE, EJ, EJL, EJH = dims(), dims(), dims(), dims(), dims(8), dims(8)  # the block's end phase
# piece 2d-minimal: the kernel stops a block at a stack underflow, / or MOD by zero, an unknown word (text_vm kerr)
F_ERR, F_EUP, F_EUM, F_EPRD, ERRST, RST, RSTD, SUPP, DZ, LEND, ETJ = (dims() for _ in range(11))
STRUCT, OPENEV, CLOSEV, STRUCTC, NMCL, NLEND = (dims() for _ in range(6))  # a structure is open (the top-level definition's)
E_EK, ERRK, EEK, EPK = (dims(len(KCLS)) for _ in range(4))  # error kind (its line)
UNDK = dims(4)  # an underflow at this op: undo + (a pop), - (a push), 0, a print (+, and not counted)
UNDT = ("u+", "u-", "u0", "up")
ERRTXT = ["error: " + TV.err_text(k) for k in KCLS]  # the block's error lines (text_vm: ERRS, RETRY)
NETX = (33, 32, 35, 33, 16, 18, 18, 16, 16, 17, 19, 40, 21, 22, 19, 19, 15, 21, 41, 31, 43, 16, 16, 17, 22)  # tokens of each error text (checked in build)
SKW = ("if", "else", "begin", "while", "do")  # the open structure's kind (one-hot)
SKOH, SKC, TCK = dims(len(SKW)), dims(len(SKW)), dims(len(SKW) + 1)  # live / latched at a close; a structure token fed
TOPA = dims()  # the top-level definition's first cell (its calls resolve here: the dictionary can't rebind a name)
INLCRP, EXPKX = dims(), dims(5)  # the decl word before was in a \ comment; EXPK one-hot without those (the rules')
WERR = dims(len(KCLS))  # the closed word is an error of that class (l_e22: for l_f_final, after a top-level run)
ETX = [dims(n) for n in NETX]
SHW = ("gcd", "lcm", "?dup")
HKEY = {NMTOP: TV.name_key("\x01top")} | {NMSH[w]: TV.name_key("\x01" + w) for w in SHW}  # hidden names' keys
# rebind (text_vm.rebind): a bound name declared again; <rbh_hi> <rbl_kind_lo> carry its old entry, cancelled at <rbl>
E_RBH, E_CH, E_RBL, E_CL, CK_DEF, CK_VAR, CK_CONST, F_CST, CHR, DVR = (dims() for _ in range(10))
MULOP, DIVOP = dims(), dims()  # * / MOD top operand over 10 digits; / dividend over 10 digits (text_vm sp.Checked)
VARSFULL, VARSC, WA0F, NOFREE = (dims() for _ in range(4))  # all 32 variable cells taken (live; VARSC: latched at the
# close, as a declaration's own <var> fills the 32nd); WA = 0; no code cell free (WA >= 191)
TXTM = dims()
F_EWA, F_EUR, LO126, DIG21, CFULL, NPRF, NPRF1, FULLP, FULLM, RFULL, DIGB, DIGU, DIGM = (dims() for _ in range(13))
F_TICK, F_TICK2, MACH, TC1, TC2, TC3, TICKD, TICK2D, CAPF, TICKO, TICK2O, ERRCAPO = (dims() for _ in range(12))
NLV = 4  # control stack levels (text_vm.MAXNEST)
CD, CD0, CDF, STRUCTD, LDV, LPV, BEGSEL, BLOOP, CDFC, LPC = (dims() for _ in range(10))
LVW, LVC = dims(NLV), dims(NLV)  # the level written now (= CD) / closed now (CD + a closer)
OPENL, ELSEL, HASEL, BEGL, LOOPL_, DOL = (dims(NLV) for _ in range(6))  # per-level registers (layer NEST)
KEV, KDL = dims(5), dims(5 * NLV)  # a structure word's kind event (if else begin while do); per-level kinds
LDOH, LDC = dims(NLV + 1), dims(NLV + 1)  # do loops open (0..4), latched at a close
LPF = dims()  # inside a loop (live)
ENOWF = (CFULL, NPRF, NPRF1, FULLP, FULLM, RFULL, DIGB, DIGU, DIGM)  # (an error at this token: the kernel stops)  # a fence has opened (text mode): only in-fence digits / '-' compile (not an output block's, after an error)
CDHROW, CDCOL, CDCOL2, VHI, VLO = dims(NHROW), dims(NB1 + NB0), dims(NB1), dims(12), dims(16)  # cancel row / column
# a1 (text_vm.freeat): the last block's dead top-level code freed at the next block's first token by <free_h>
FREEOK, F_FREE, FREEV, ENDOK = dims(), dims(), dims(), dims()  # nothing compiled since a top-level ; and no error; <free_h>, its h
HCA, FTH, ENDPH, E_HID = dims(13), dims(12), dims(), dims()  # E_HID: an id < 1000  # ceil(the top-level definition's first cell / 16) (latched at <nmtop>); code heads >= h
assert _n[0] <= HIDDEN, _n[0]
OTOK = {"MINUS": MINUS, "MMINUS": MMINUS, "SPACE": SPACE, "END": END, "MSTEP": MSTEP, "SPL": SPL, "SPX": SPX,
        "SPC": SPC, "CLR0": CLR0, "CLR1": CLR1, "CLR2": CLR2, "CLRC": CLRC, "SW2": SW2, "SW3": SW3}


# ---------------- primitives (s2/v2 recipes, sized for this config) ----------------
def base(types):
    cfg = AutoConfig.from_pretrained(LIGHTNING, hidden_size=HIDDEN, num_hidden_layers=len(types),
                                     layers_block_type=types, mamba_num_heads=H, mamba_head_dim=P, n_groups=G,
                                     ssm_state_size=N, intermediate_size=WIDTH, num_nextn_predict_layers=0,
                                     tie_word_embeddings=False, use_mamba_kernels=False, eos_token_id=DONE,
                                     chunk_size=64)  # (prefill: a chunk's cumsum of wipes (dt A = -173 each) must
    # stay small for exp(segsum) to be exact: at 128, a register read 0.996 after ~250 wipes in one chunk)
    m = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    for p in m.parameters():
        p.requires_grad_(False).zero_()
    e = m.model.embeddings.weight
    e[:, BAL], e[:, F_NRD], e[:, ONE], e[:, BAL2], e[:, BAL3] = 2.0**20, 1, 1, 2.0**19, 2.0**19
    for d in range(10):
        for t in (DIGITS[d], MD[d]):
            e[t, F_RDIG], e[t, DV], e[t, F_NRD] = 1, d, 0
    for t in (MINUS, MMINUS):
        e[t, F_MINUS], e[t, F_NRD] = 1, 0
    for t, f in TOKF.items():
        e[t, f] = 1
    e[DIV, F_MOD] = 1
    e[INC, AL["Y"]] = e[DECR, AL["Y"]] = 1  # 1+ / 1- : operand y = 1 (row above the top is clean: reads 0)
    for c in range(NVC):
        e[fetch_tok(c), F_FETCH], e[fetch_tok(c), FV] = 1, c
        e[store_tok(c), F_STORE], e[store_tok(c), FV] = 1, c
    for t, f in ((DEF, F_DEF), (SEMI, F_SEMI), (THEN, F_THEN), (BEGIN, F_BEGIN), (VAR, F_VAR), (FETCHM, F_FETCHM),
                 (STOREM, F_STOREM), (RCARRY, F_RCARRY), (RBORROW, F_RBORROW), (DONE, F_DONE)):
        e[t, f] = 1
    e[TICK, F_TICK] = e[TICK2, F_TICK2] = 1
    for t in (CARRY, BORROW, RCARRY, RBORROW, TICK, TICK2):
        e[t, F_NRD] = 0  # transparent: no register wipes
    for d in range(10):
        e[DIGITS[d], DIGH + d] = e[MD[d], DIGH + d] = 1
    e[SEMI, OPC + CI[EXIT]] = e[SEMI, STEPW] = 1  # ';' stores the EXIT cell
    for x in range(NADDR):
        e[A0 + x, F_A], e[A0 + x, AV] = 1, x
    e[DONE, AV] = TOPRET
    txt = torch.tensor([is_text(t) for t in range(e.shape[0])])
    kap = torch.tensor([kappa(t) for t in range(e.shape[0])], dtype=torch.float32)
    e[:, F_TXT] = txt.to(e.dtype)
    e[:, F_NTXT] = (~txt).to(e.dtype)
    e[:, KA] = (kap[:, 0] * txt).to(e.dtype)
    e[:, KB] = (kap[:, 1] * txt).to(e.dtype)
    for t in (NM, NMP, *HKEY):  # text mode: a name is one kernel-emitted text token, keyed by the front end (KCUR / KPREV)
        e[t, F_TXT], e[t, F_NTXT], e[t, KA], e[t, KB] = 1, 0, 0, 0
    for t, k in HKEY.items():  # a hidden name: its key in KCUR's place (l_f_cls gates the latched KCUR off here)
        e[t, KCUR], e[t, KCUR + 1], e[t, E_NMH] = k[0], k[1], 1
    e[NMTOP, E_NMTOPF] = 1
    for i, w in enumerate(SHW):
        e[NMSH[w], E_NMSHF + i] = 1
    undo = {"u+": [F_EUP], "u-": [F_EUM], "zero": [F_EUP], "up": [F_EUP], "mulop": [F_EUP], "divop": [F_EUP],
            "digits2": [F_EUP], "code": [F_EWA], "prints": [F_EUP, F_EPRD], "prints1": [F_EPRD], "full": [F_EUM],
            "full2": [F_EUP], "rfull": [F_EUR]}
    for name in ["u+", "u-", "u0", "up", "zero"] + list(KCLS[2:]) + list(EALIAS):
        t = ERRT[name]  # stop the block; undo the failed op's depth change
        kind = 1 if name == "zero" else 0 if name in UNDT else KCLS.index(EALIAS.get(name, name))
        e[t, F_ERR], e[t, E_EK + kind] = 1, 1
        for f in undo.get(name, []):
            e[t, f] = 1
    e[ERRT["up"], F_EPRD] = 1
    e[CST, F_CST] = 1
    e[DOB, F_BEGIN] = e[DOB, F_DOB] = 1
    for h, t in enumerate(FREE):  # <free_h>: WA = 16 h, code heads h.. wiped
        e[t, F_FREE], e[t, FREEV] = 1, h
        e[t, FTH + h:FTH + NADDR // 16] = 1
    for h, t in enumerate(RBH):
        e[t, E_RBH], e[t, E_CH] = 1, h
    for k, row in enumerate(RBL):
        for lo, t in enumerate(row):  # (F_NRD 0: CHR, the hi part, holds through <rbl>)
            e[t, E_RBL], e[t, E_CL], e[t, F_NRD] = 1, lo, 0
            e[t, (CK_DEF, CK_VAR, CK_VAR)[k]] = 1
            if k == 2:
                e[t, CK_CONST] = 1
    for i, t in enumerate(HCT):  # hidden cells 32 + k: run as <@c> / <!c>
        e[t, F_FETCH if i < NHK else F_STORE], e[t, FV], e[t, E_HC + i] = 1, HCELL0 + i % NHK, 1
    for blk in m.model.layers:
        blk.norm.weight.fill_(SCALE)
        blk.norm.weight[BAL] = SCALE / 2**20
        blk.norm.weight[BAL2] = blk.norm.weight[BAL3] = 0
        if blk.block_type == MA:
            mx = blk.mixer
            W = mx.in_proj.weight
            W[DO:DO + H, BAL] = -200
            mx.conv1d.weight[:, 0, -1] = 1
            mx.A_log.fill_(-60.0)
            mx.norm.weight.fill_(1)
            for g in range(G):
                b = 8 * g + 7
                W[b * P:b * P + 32, BAL] = 32
                W[XO + b * P:XO + b * P + 32, BAL] = 32
                mx.D[b] = 2.0**43
    m.model.norm_f.weight.copy_(m.model.layers[0].norm.weight)
    return m


def rows0(mx, groups):
    for g in groups:
        mx.in_proj.weight[BO + g * N, BAL] = mx.in_proj.weight[CO + g * N, BAL] = 32


def register(mx, h, sets, wipes, out, c=0):
    """Wiped by any wipe feature, then holds sum(w * f) of sets [(f, w)] (exact for 8-bit values). Group needs rows0."""
    W = mx.in_proj.weight
    for f in wipes:
        W[DO + h, f] = BIG
    for f, w in sets:
        W[XO + h * P + c, f] = w * 2.0**-83
    W[h * P + c, BAL] = 32
    mx.norm.weight[h * P + c] = 2.0**26
    mx.out_proj.weight[out, h * P + c] = 1


def counter(mx, h, inc, dec, resets, out, steps=()):
    """+1 / -1 per inc / dec feature; steps: (feature, k) adds k (|k| a power of two <= 128). Group needs rows0."""
    W = mx.in_proj.weight
    for f in inc:
        W[XO + h * P, f], W[DO + h, f] = ALPHA, 232
    for f in dec:
        W[XO + h * P, f], W[DO + h, f] = -BETA, 232
    for f, k in steps:
        W[XO + h * P, f], W[DO + h, f] = (ALPHA if k > 0 else -BETA), 200 + 32 * abs(k)
    for f in resets:
        W[DO + h, f] = BIG
    W[h * P, BAL] = 32
    mx.norm.weight[h * P] = 2.0**33
    mx.out_proj.weight[out, h * P] = 1


def dec(mx, h, c0, terms, off, outs, gate=(), neg=()):
    """Channel c0+i of head h = [v == off + i] (v = sum w*f) -> outs[i] = [(dim, weight)]; zero unless one gate
    feature is 1 (if gate given) and every neg feature is 0."""
    W = mx.in_proj.weight
    for i, ds in enumerate(outs):
        if not ds:
            continue
        c = h * P + c0 + i
        assert c0 + i < P
        m, r = divmod(i + off, 16)  # bias -256 (i + off) + 128 split so every weight is bf16-exact
        W[XO + c, BAL] = -4096 * m - (16384 if gate else 0)
        W[XO + c, ONE] = -256 * r + 128
        W[c, BAL], W[c, ONE] = 4096 * m, 256 * r + 128
        for f, w in terms:
            f = ONE if f == BAL else f
            W[XO + c, f] += 256 * w
            W[c, f] -= 256 * w
        for g in gate:
            W[XO + c, g] += 16384
        for f in neg:
            W[XO + c, f] -= 16384
        mx.norm.weight[c] = 2.0**37
        for d, w in ds:
            mx.out_proj.weight[d, c] += w
    mx.D[h] = 1


def decw(mx, hs, terms, lo, hi, out, gate=(), neg=()):
    """dec over values lo..hi spread across heads hs (64 channels each); out(v) -> [(dim, weight)]."""
    vals = list(range(lo, hi + 1))
    for k, h in enumerate(hs):
        chunk = vals[64 * k:64 * (k + 1)]
        if not chunk:
            break
        dec(mx, h, 0, terms, chunk[0], [out(v) for v in chunk], gate, neg)
    assert len(vals) <= 64 * len(hs)


def zand(pos, neg=()):
    """z terms of a gated channel that is on iff every pos feature is 1 and every neg feature is 0."""
    return [(f, 160) for f in pos] + [(f, -160) for f in neg], -160 * len(pos) + 32


class Gated:
    """Sequential channel allocator for D-skip gated channels in a list of (stateless) heads."""

    def __init__(self, mx, hs):
        self.mx, self.hs, self.i = mx, hs, 0

    def __call__(self, x_terms, x_const, z_terms, z_const, outs):
        h, c = self.hs[self.i // P], self.i % P
        self.i += 1
        W, ch = self.mx.in_proj.weight, h * P + c
        for f, w in x_terms:
            W[XO + ch, f] += w
        W[XO + ch, BAL] += x_const
        for f, w in z_terms:
            W[ch, f] += w
        W[ch, BAL] += z_const
        self.mx.norm.weight[ch] = 2.0**46
        self.mx.D[h] = 1
        for d, w in outs:
            self.mx.out_proj.weight[d, ch] += w

    def delay(self, src, out):
        """out = previous token's src (conv tap t-1); src is 0 or an integer 1..255."""
        h, c = self.hs[self.i // P], self.i % P
        self.i += 1
        mx, ch = self.mx, h * P + c
        mx.in_proj.weight[XO + ch, src] = 32
        mx.conv1d.weight[ch, 0, -1], mx.conv1d.weight[ch, 0, -2] = 0, 1
        mx.in_proj.weight[ch, BAL] = 32
        mx.norm.weight[ch] = 2.0**41
        mx.D[h] = 1
        mx.out_proj.weight[out, ch] = 1

    def gdelay(self, src, pos, out):
        """out = previous token's src (0 / 1..255) if every pos feature is 1 now (zand gate), else 0."""
        h, c = self.hs[self.i // P], self.i % P
        self.i += 1
        mx, ch = self.mx, h * P + c
        mx.in_proj.weight[XO + ch, src] = 32
        mx.conv1d.weight[ch, 0, -1], mx.conv1d.weight[ch, 0, -2] = 0, 1
        zt, zc = zand(pos)
        for f, w in zt:
            mx.in_proj.weight[ch, f] += w
        mx.in_proj.weight[ch, BAL] += zc
        mx.norm.weight[ch] = 2.0**41
        mx.D[h] = 1
        mx.out_proj.weight[out, ch] = 1


def rcounter(mx, h, inc, wipes, sets, out):
    """+1 per inc feature; a wipe feature sets the state to sum(v * f) of sets [(f, v)] (v integer, |v| < 256) instead.
    inc and wipe features never fire together; at most one wipe feature at a time. Group needs rows0."""
    W = mx.in_proj.weight
    for f in inc:
        W[XO + h * P, f], W[DO + h, f] = ALPHA, 232
    for f in wipes:
        W[DO + h, f] = BIG
    for f, v in sets:
        W[XO + h * P, f] = v * 2.0**-90  # silu halves it; dt 2^94 * B 32 -> state 256 v, as an increment's
    W[h * P, BAL] = 32
    mx.norm.weight[h * P] = 2.0**33
    mx.out_proj.weight[out, h * P] = 1


def andl(mlp, u, lits, outs):
    """relu^2 AND of literals ('+', dims) = sum(dims), ('-', dims) = 1 - sum(dims), each 0/1; outs: (dim, weight)."""
    bias = 1 - 2 * len(lits)
    for sign, ds in lits:
        for d in ds:
            mlp.up_proj.weight[u, d] += 2 if sign == "+" else -2
        bias += 2 if sign == "-" else 0
    mlp.up_proj.weight[u, BAL] = bias
    for d, w in outs:
        mlp.down_proj.weight[d, u] += w
    return u + 1


# ---------------- layers ----------------
def l_registers(mx):
    rows0(mx, range(6))
    hs = iter(heads(range(5)) + heads([5])[1:])
    for f, out in ((F_LIT, INLIT), (F_PR, PMODE), (F_SPL, M_SPL), (F_SPX, M_SPX), (F_SPC, M_SPC), (F_CLR0, M_CLR0),
                   (F_CLR1, M_CLR1), (F_CLR2, M_CLR2), (F_CLRC, M_CLRC)):
        register(mx, next(hs), [(f, 1)], [F_NRD], out)
    kw = OPS + [F_DEF, F_SEMI, F_VAR, F_FETCHM, F_STOREM, F_A, F_DONE, F_ERR]
    register(mx, next(hs), [(f, 1) for f in STARTS], STARTS + kw, SEGST)
    for k, fs in KINDS.items():
        register(mx, next(hs), [(f, 1) for f in fs], kw, K[k])
    register(mx, next(hs), [(F_IF, 1), (F_WHILE, 1)], kw, K_IFW)
    register(mx, next(hs), [(DV, 1)], [F_RDIG], LASTDIG)
    register(mx, next(hs), [(FV, 1)], [F_FETCH, F_STORE], VARC)
    counter(mx, next(hs), [F_RDIG], [], [F_LIT, F_PR] + STARTS, J)
    register(mx, next(hs), [(F_DEF, 1)], [F_DEF, F_SEMI], COMPILE)
    for f, out in ((F_DEF, NM_DEF), (F_VAR, NM_VAR), (F_CALL, NM_CALL), (F_FETCHM, NM_FETCH), (F_STOREM, NM_STORE)):
        register(mx, next(hs), [(f, 1)], [F_NTXT], out)
    register(mx, next(hs), [(F_A, 1)], [F_A, F_DONE], REPLAY)
    register(mx, next(hs), [(AV, 1)], [F_A], BASE)
    register(mx, heads([5])[0], [(F_MOD, 1)], [F_MUL, F_MOD], K_MOD)
    register(mx, next(hs), [(F_DIV, 1)], [F_MUL, F_MOD], K_DIV)
    register(mx, next(hs), [(F_CR, 1)], kw + [E_PNL, E_NL], K_CR)  # <cr>: its newline next (after any carry)
    register(mx, next(hs), [(F_SPACEOP, 1)], kw + [E_PSP, E_SPACE], K_SP)
    Gated(mx, heads([6])[:1]).delay(F_TXT, PREVTXT)
    W = mx.in_proj.weight
    for k, out in ((KA, NHA), (KB, NHB)):  # name hash: sum of kappa over the text tokens since the last non-text token
        for cnt in (False, True):
            h = next(hs)
            W[XO + h * P, F_TXT], W[XO + h * P, F_NTXT] = 32, -256
            if not cnt:
                W[XO + h * P, k] = 1
            W[DO + h, F_TXT], W[DO + h, F_NTXT] = 232, BIG
            W[h * P, BAL] = 32
            mx.norm.weight[h * P] = 2.0**31
            mx.out_proj.weight[out, h * P] = -1 if cnt else 1


def l_cancel(mlp):
    """Inside a definition (COMPILE) a codeable token is stored, not run: its data features are cancelled and its op
    code (OPC) + a cell step (STEPW) are raised. Name-token flags."""
    u = 0
    feat = {LIT: F_LIT, PR: F_PR, ADD: F_ADD, SUB: F_SUB, MUL: F_MUL, MOD: F_MOD, INC: F_INC, DECR: F_DECR, ZEQ: F_ZEQ,
            LT: F_LT, GT: F_GT, DUP: F_DUP, DROP: F_DROP, OVER: F_OVER, PICK: F_PICK, IF: F_IF, ELSE: F_ELSE,
            WHILE: F_WHILE, REPEAT: F_REPEAT, EXIT: F_EXIT, CALL: F_CALL, FETCHM: F_FETCHM, STOREM: F_STOREM,
            MINUS: F_MINUS, SWAP: F_SWAP, NEG: F_NEG, DIV: F_DIV, CR: F_CR, SPACEOP: F_SPACEOP, DEPTH: F_DEPTH,
            LEAVES[0]: F_LV}
    for k, t in enumerate(HCT):
        feat[t] = E_HC + k
    for code, i in CI.items():
        f = DIGH + code[1] if isinstance(code, tuple) else feat[code]
        outs = [(f, -1), (OPC + i, 1), (STEPW, 1)] + ([(F_RDIG, -1)] if isinstance(code, tuple) else [])
        outs = [(F_FETCH if code in HCT[:NHK] else F_STORE, -1), (OPC + i, 1), (STEPW, 1)] if code in HCT else outs
        outs += [(F_MOD, -1)] if code == DIV else []
        lits = [("+", [f]), ("+", [COMPILE])] + ([("-", [F_DIV])] if code == MOD else [])
        if isinstance(code, tuple) or code == MINUS:  # (protocol mode: any; text mode: the kernel's own, in the fence)
            u = andl(mlp, u, lits + [("-", [TXTM])], outs)
            u = andl(mlp, u, lits + [("+", [ENDPH]), ("+", [E_ISMM if code == MINUS else E_ISMD])], outs)  # (the end
            # phase after the closing ```: its machine digits, e.g. abs's 0 in a hidden gcd / lcm defined there)
            lits = lits + [("+", [FON])]
        u = andl(mlp, u, lits, outs)
    u = andl(mlp, u, [("+", [F_TXT]), ("-", [PREVTXT])], [(FIRSTTXT, 1)])
    u = andl(mlp, u, [("+", [F_TXT]), ("+", [NM_DEF])], [(DNDEF, 1)])
    u = andl(mlp, u, [("+", [F_TXT]), ("+", [NM_VAR])], [(DNVAR, 1)])
    # a name is its first token: later name tokens compile / resolve nothing
    u = andl(mlp, u, [("+", [F_TXT]), ("-", [PREVTXT]), ("+", [COMPILE]), ("+", [NM_CALL, NM_FETCH, NM_STORE])],
             [(CNAME, 1)])
    for nm, out in ((NM_CALL, UN_CALL), (NM_FETCH, UN_FETCH), (NM_STORE, UN_STORE)):
        u = andl(mlp, u, [("+", [F_TXT]), ("-", [PREVTXT]), ("+", [nm]), ("-", [COMPILE])], [(out, 1)])
    # piece 2c: hidden names defined (a shared word, the top-level definition, any other) / a shared word called in one
    for i in range(3):
        u = andl(mlp, u, [("+", [F_TXT]), ("+", [NM_DEF]), ("+", [E_NMSHF + i])], [(DNSH + i, 1)])
        u = andl(mlp, u, [("+", [F_TXT]), ("+", [NM_CALL]), ("+", [COMPILE]), ("+", [E_NMSHF + i])], [(USE + i, 1)])
    u = andl(mlp, u, [("+", [F_TXT]), ("+", [NM_DEF]), ("+", [E_NMTOPF])], [(DNTOP, 1)])
    u = andl(mlp, u, [("+", [F_TXT]), ("+", [NM_DEF]), ("-", list(range(E_NMSHF, E_NMSHF + 3)))], [(DNUSR, 1)])


def l_counters(mx):
    rows0(mx, [0, 1])
    hs = iter(heads([0, 1]))
    counter(mx, next(hs), PUSH + [F_EUP], POP + [F_EUM], [], LO, [(F_CARRY, -128), (F_BORROW, 128), (F_SW2, -2)])
    counter(mx, next(hs), [F_CARRY], [F_BORROW], [], HI)
    counter(mx, next(hs), [F_MSTEP], [], [F_MUL, F_MOD], MJ)
    counter(mx, next(hs), [F_CALL], [F_EXIT, F_EUR], [FOPEN], RLO, [(F_RCARRY, -128), (F_RBORROW, 128)])
    counter(mx, next(hs), [F_RCARRY], [F_RBORROW], [FOPEN], RHI)  # (the return stack is emptied at a block start)
    counter(mx, next(hs), CELLF, [], [F_A], STEPS)
    h = next(hs)  # literal digits / '-' of a replayed literal are steps too
    W = mx.in_proj.weight
    W[XO + h * P, BAL], W[XO + h * P, ONE] = -512, ALPHA  # (ALPHA - 512 is not a bf16 weight)
    W[XO + h * P, F_RDIG] = W[XO + h * P, F_MINUS] = W[XO + h * P, INLIT] = 256
    W[DO + h, F_RDIG] = W[DO + h, F_MINUS] = 232
    W[DO + h, F_A] = BIG
    W[h * P, BAL] = 32
    mx.norm.weight[h * P] = 2.0**33
    mx.out_proj.weight[STEPS, h * P] = 1
    h = next(hs)
    counter(mx, h, [STEPW], [F_EWA], [F_FREE], WA)
    mx.in_proj.weight[XO + h * P, FREEV] = 2.0**-86  # a1: <free_h> sets WA = 16 h (rcounter's set: 256 v per 2^-90 v)
    counter(mx, next(hs), [F_VAR], [], [], NVAR)


def l_addr1(mx):
    """Row one-hot + block offset of each role's position p = lo + off - 1, cells, write channel, carry flags;
    compile-time registers; delays."""
    hs = iter(heads(range(6)) + heads([6])[1:])
    take = lambda n: [next(hs) for _ in range(n)]
    out = lambda r: (lambda v: [(ROW[r] + v % N, 1), (BOFF[r], v // N)] if v // N else [(ROW[r] + v % N, 1)])
    lo = [(LO, 1), (BAL, -1)]  # position of depth lo
    x_off = [(F_DUP, -1), (F_OVER, -2), (F_PR, 1), (F_IF, 1), (F_WHILE, 1), (F_STORE, 1), (M_CLR1, 1), (M_CLR2, 2),
             (F_SWAP, -2), (F_SW2, 1), (F_SW3, 1)]
    decw(mx, take(5), lo + x_off, 0, 257, out("X"), neg=[F_PICK, F_FETCH, M_CLRC])
    decw(mx, take(5), lo + [(LASTDIG, -1), (BAL, -1)], 0, 257, out("X"), gate=[F_PICK])
    decw(mx, take(5), lo + [(BAL, 1)], 0, 257, out("Y"))
    decw(mx, take(5), lo + [(BAL, 2)], 0, 257, out("Z"))
    w_off = [(M_SPX, 2), (M_CLR2, 2), (M_CLR1, 1), (PMODE, 1)]
    decw(mx, take(5), lo + w_off, 0, 257, out("W"), neg=[M_SPC, M_CLRC] + CLEARM)  # write modes: B = 32
    decw(mx, take(5), lo + w_off, 0, 257, lambda v: [(CROW + v % N, 1)] + ([(BOFF["W"], v // N)] if v // N else []),
         gate=CLEARM, neg=[M_CLRC])  # clear modes: B = silu(-BETA) = -1/4
    # variable cells: head 8, row c
    cells = lambda c: c < NVC or HCELL0 <= c < HCELL0 + NHK  # (named cells, and the hidden cells the kernel runs)
    for src, rw, hd, g in ((FV, ROW["X"], "X", F_FETCH), (VARC, ROW["X"], "X", M_CLRC), (VARC, ROW["W"], "W", M_SPC),
                           (VARC, CROW, "W", M_CLRC)):
        dec(mx, next(hs), 0, [(src, 1)], 0, [[(rw + c, 1), (HD[hd] + 8, 1)] * cells(c) for c in range(HCELL0 + NHK)],
            gate=[g])
    dec(mx, next(hs), 0, [(J, 1)], 1, [[(WCH + k, 1)] for k in range(ND)], gate=[F_RDIG])
    h = next(hs)
    dec(mx, h, 0, [(J, 1)], 0, [[(JH + k, 1)] for k in range(ND + 1)])
    dec(mx, h, ND + 1, [(J, 1)], ND + 1, [[(OVX, 1)]], gate=[INLIT], neg=[COMPILE])  # a run literal's digit ND + 1
    dec(mx, next(hs), 0, [(MJ, 1)], 0, [[(MJH + k, 1)] for k in range(ND)])
    h = next(hs)
    dec(mx, h, 0, [(HI, 1)], 1, [[(BHI, 1)]] * 8)
    dec(mx, h, 8, [(HI, 1)], 7, [[(HI7, 1)], [(HI8, 1)]])
    dec(mx, h, 10, [(RHI, 1)], 1, [[(RBHI, 1)]] * 8)
    dec(mx, h, 18, [(RHI, 1)], 7, [[(RHI7, 1)], [(RHI8, 1)]])
    for src, flags in ((LO, (CDUE, BLO, LO129, LO1, LO126)), (RLO, (RCDUE, RBLO, RLO129, RLO1, None))):
        decw(mx, take(4), [(src, 1)], 0, 255, lambda v, fl=flags: [(fl[0], 1)] * (v >= 192) + [(fl[1], 1)] * (v < 64)
             + [(fl[2], 1)] * (v >= 129) + [(fl[3], 1)] * (v >= 1) + [(fl[4], 1)] * (v >= 126 and fl[4] is not None))
    gd = Gated(mx, heads([6])[:1])
    gd.delay(DNDEF, PREVDEF)
    gd.delay(DNVAR, PREVVAR)
    gd.delay(F_EXIT, PREVEXIT)
    rows0(mx, [7])
    hr = iter(heads([7]))


def l_addr2(mx):
    """Head one-hot = hi + block offset (stack heads 0..7); code addresses; return-stack rows; call frame value."""
    hs = iter(heads(range(7)))
    take = lambda n: [next(hs) for _ in range(n)]
    for r, neg in (("X", [F_FETCH, M_CLRC]), ("Y", []), ("Z", []), ("W", [M_SPC, M_CLRC])):
        dec(mx, next(hs), 0, [(HI, 1), (BOFF[r], 1)], 0, [[(HD[r] + k, 1)] for k in range(8)], neg=neg)
    ca = lambda lo_, hi_: (lambda v: [(lo_ + v % NL, 1), (hi_ + v // NL, 1)])
    decw(mx, take(3), [(BASE, 1), (STEPS, 1), (BAL, -1)], 0, NADDR - 1, ca(PCL, PCH))
    decw(mx, take(3), [(BASE, 1), (STEPS, 1)], 0, NADDR - 1, ca(NPCL, NPCH))
    decw(mx, take(3), [(WA, 1), (BAL, -1)], 0, NADDR - 1, ca(WAL, WAH))
    rout = lambda k: (lambda v: [(RROW[k] + v % N, 1)] + ([(RBOFF[k], v // N)] if v // N else []))
    decw(mx, take(5), [(RLO, 1), (BAL, -1)], 0, 257, rout("W"), gate=[F_CALL])  # frame of depth RLO
    decw(mx, take(5), [(RLO, 1)], 0, 257, rout("R"), gate=[F_EXIT])  # depth RLO + 1 (popped at this exit)
    decw(mx, take(5), [(RLO, 1)], 0, 257, rout("C"), gate=[PREVEXIT])  # cleared on the token after the exit


def l_addr2b(mx):
    """Call frame value, CALLR, OPEND (l_addr2's gated head, one layer later)."""
    gd = Gated(mx, [56])
    zt, zc = zand([F_CALL, REPLAY])
    gd([], 32, zt, zc, [(CALLR, 1 / 32)])
    gd([(BASE, 1), (STEPS, 1)], 32, zt, zc, [(RETV, 1)])  # return address = next cell
    zt, zc = zand([F_CALL], [REPLAY])
    gd([], 32 + TOPRET, zt, zc, [(RETV, 1)])  # a user call returns to the user
    gd([], 32, *zand([F_CALL]), [(RETV, -1)])


def l_store(mx):
    """Data stack + cells, 3 read replicas (X, Y, Z: groups 0-1, 2-3, 4-5), identical writes at role W."""
    W, o = mx.in_proj.weight, mx.out_proj.weight
    for rep, r in enumerate("XYZ"):  # groups 0-5
        gs = (2 * rep, 2 * rep + 1)
        for g in gs:
            for n in range(N):
                W[BO + g * N + n, ROW["W"] + n] = 32
                W[BO + g * N + n, CROW + n] = -BETA
                W[CO + g * N + n, ROW[r] + n] = 32
        for hh, h in enumerate(heads(gs)[:9]):
            W[DO + h, F_RDIG] = W[DO + h, F_MINUS] = 232
            for f in CLEARM:
                W[DO + h, f] = 4064
            for c in range(2 * ND + 1):
                ch = h * P + c
                if c < 2 * ND:
                    k = c % ND
                    for f in [HD["W"] + hh, WCH + k, F_RDIG] + MODES:
                        W[XO + ch, f] = 192
                    W[XO + ch, BAL] = -736
                    if c < ND:
                        W[XO + ch, DV] = 1
                        o[RA[r] + k, ch] = 1
                    else:
                        o[RA[r] + k, ch] = -1
                        o[RW[r] + k, ch] = 2.0**-5
                else:
                    for f in [HD["W"] + hh, F_MINUS] + MODES:
                        W[XO + ch, f] = 192
                    W[XO + ch, BAL] = -544
                    o[RS[r], ch] = 2.0**-5
                W[ch, HD[r] + hh] = 160
                W[ch, BAL] = -128
                mx.norm.weight[ch] = 2.0**31


def l_ctrl6(mx):
    """Return-stack heads; dictionary write key / value; patch row."""
    hs = iter(heads(range(3, 7)))
    for k, g in (("W", F_CALL), ("R", F_EXIT), ("C", PREVEXIT)):
        dec(mx, next(hs), 0, [(RHI, 1), (RBOFF[k], 1)], 0, [[(RHD[k] + i, 1)] for i in range(8)], gate=[g])
    for _ in range(4):
        next(hs)  # (the dictionary write-key decode: l_f_keys, after PENDA / PENDB moved to layer 13)
    gd = Gated(mx, [next(hs), 27])  # (33 is l_f_l6's; 27-30 are free here)
    for j in range(ND):  # x_j = MSD-first digit j of x (j = step index)
        gd([(RA["X"] + j, 1)], 32, *zand([MJH + j]), [(XJ, 1)])
        gd([(RA["X"] + j, 1)], 32, *zand([MJH + j, K_MOD]), [(XJM, 1)])
    gd([], 32, [], 32, [(XJ, -1)])
    gd([], 32, *zand([K_MOD]), [(XJM, -1)])
    gd([(WA, 1), (STEPW, -1)], 32, *zand([PREVDEF]), [(DVAL, 1)])  # a definition binds its first cell
    gd([(NVAR, 1), (F_VAR, -1)], 31, *zand([PREVVAR]), [(DVAL, 1)])  # a variable binds its cell index
    gd([], 32, [(PREVDEF, 160), (PREVVAR, 160)], -128, [(DVAL, -1)])
    rep, lvo = OPC + CI[REPEAT], OPC + CI[LEAVES[0]]  # patch rows / jump targets from the control stack (l_nest)
    for d in range(NLV):
        gd([(OPENL + d, 1)], 32, *zand([OPC + CI[ELSE], LVW + d]), [(PROW, 1)])  # else: its if's cell
        gd([(ELSEL + d, 1)], 32, *zand([F_THEN, LVC + d, HASEL + d]), [(PROW, 1)])  # then: the else's, or the if's
        gd([(OPENL + d, 1)], 32, *zand([F_THEN, LVC + d], [HASEL + d]), [(PROW, 1)])
        gd([(OPENL + d, 1)], 32, *zand([rep, LVC + d]), [(PROW, 1)])  # repeat: the loop's entry <while>
        gd([(BEGL + d, 1)], 32, *zand([rep, LVC + d]), [(BEGSEL, 1)])  # ... and its <begin> (its own target)
        gd([(DOL + d, 32)], 0, [(LVW + c, 160) for c in range(d, NLV)], -128, [(LDV, 1 / 32)])  # do loops open
        gd([(LOOPL_ + d, 32)], 0, [(LVW + c, 160) for c in range(d, NLV)], -128, [(LPV, 1 / 32)])  # loops open
        for c in range(d, NLV):  # <lv0>: the innermost loop's <begin> (level d, nothing deeper up to the current c)
            gd([(BEGL + d, 1)], 32, *zand([lvo, LVW + c, LOOPL_ + d], [LOOPL_ + e for e in range(d + 1, c + 1)]),
               [(BLOOP, 1)])
    gd([], 32, [(f, 160) for f in PATCHF], -128, [(PROW, -1)])
    gd([], 32, *zand([rep]), [(BEGSEL, -1)])
    gd([], 32, *zand([lvo]), [(BLOOP, -1)])
    for pos_ in ([HI8, LO1], [HI7, LO129], [RHI8, RLO1], [RHI7, RLO129]):  # depth > 1024: defined error token
        gd([], 32, *zand(pos_, [TXTM]), [(OVFO, 1 / 32)])  # (protocol mode; text mode: an error, text_vm)
    grow = [F_LIT, F_DUP, F_OVER, F_FETCH, F_DEPTH, F_SWAP]  # text mode: an op needing a row past 1024 (text_vm GROW)
    for pos_ in ([HI8, LO1], [HI7, LO129]):
        gd([(f, 32) for f in grow], 0, *zand(pos_ + [TXTM], [F_MUL, F_MOD]), [(FULLP, 1 / 32)])
    for pos_ in ([HI8], [HI7, LO126]):  # * / MOD: depth after the op >= 1022 (the scratch row above the top)
        gd([(F_MUL, 32), (F_MOD, 32)], 0, *zand(pos_ + [TXTM]), [(FULLM, 1 / 32)])
    for pos_ in ([RHI8, RLO1], [RHI7, RLO129]):
        gd([], 32, *zand(pos_ + [TXTM, F_CALL]), [(RFULL, 1 / 32)])


def l_align(mx):
    """LSD-first operands AL[r][k] = RA[r][L-1-k] (thermometer RW gives L); clear-emission digit / marker at J."""
    gd = Gated(mx, heads(range(3)))
    for r in "XYZ":
        for n in range(ND):  # length n + 1
            zt = [(RW[r] + n, 160)] + ([(RW[r] + n + 1, -160)] if n < ND - 1 else [])
            for k in range(ND):
                gd([(RA[r] + n - k, 1)] if k <= n else [], 32, zt, -128, [(AL[r] + k, 1)])
        for k in range(ND):
            gd([], 32, [(RW[r], 160)], -128, [(AL[r] + k, -1)])
    for j in range(ND + 1):
        jz = [(JH + j, 160)] + [(f, 160) for f in EMODES]
        if j < ND:
            gd([(RA["X"] + j, 1)], 32, jz, -288, [(SELE, 1)])
            gd([(RW["X"] + j, 1)], 32, jz, -288, [(EWJ, 1)])
        else:
            gd([], 32, jz, -288, [(SELE, 1), (EWJ, 1)])
    gd([], 32, [(f, 160) for f in EMODES], -128, [(SELE, -1), (EWJ, -1)])


def l_dgp(mx):
    """Per digit: generate / propagate of x + y and of x - y; ZX = [x == 0]."""
    hs = heads(range(2))
    for c in range(ND):
        h, c0 = hs[c // 2], 29 * (c % 2)
        dec(mx, h, c0, [(AL["X"] + c, 1), (AL["Y"] + c, 1)], 9, [[(PA + c, 1)]] + [[(GA + c, 1)]] * 9)
        dec(mx, h, c0 + 10, [(AL["X"] + c, 1), (AL["Y"] + c, -1)], -9,
            [[(G1 + c, 1)]] * 9 + [[(P1 + c, 1)]] + [[(G2 + c, 1)]] * 9)
    dec(mx, hs[ND // 2], 0, [(AL["X"] + c, 1) for c in range(ND)], 0, [[(ZX, 1)]])


def l_ctrl7(mx):
    hs = heads([2])
    decw(mx, hs[:3], [(PROW, 1)], 0, NADDR - 1, lambda v: [(PATCHL + v % NL, 1), (PATCHH + v // NL, 1)], gate=PATCHF)
    dec(mx, hs[3], 0, [(XJ, 1)], 0, [[(XJH + v, 1)] for v in range(10)])
    dec(mx, hs[3], 10, [(RW["X"] + k, 1) for k in range(ND)] + [(MJ, -1)], 1, [[(FINAL, 1)]])  # last step of * / MOD
    for i in range(10):  # products d * y_i -> S[d][i] (low digit), S[d][i+1] (high digit): S in 0..17
        out = lambda v: [(S[d] + i, (d * v) % 10) for d in range(1, 10) if (d * v) % 10] + \
            [(S[d] + i + 1, (d * v) // 10) for d in range(1, 10) if (d * v) // 10]
        dec(mx, hs[4 + i // 6], 10 * (i % 6), [(AL["Y"] + i, 1)], 0, [out(v) for v in range(10)])


def l_carry(mlp):
    u = 0
    for gen, prop, out in ((GA, PA, CA), (G1, P1, B1), (G2, P1, B2)):
        for kk in range(1, ND + 1):
            for j in range(kk):
                u = andl(mlp, u, [("+", [gen + j])] + [("+", [prop + i]) for i in range(j + 1, kk)], [(out + kk - 1, 1)])
    sx, sy, sb = RS["X"], RS["Y"], SUBL
    for sg in ("+--", "-+-", "--+", "+++"):  # odd parity of (sign x, sign y, subtract): effective subtraction
        u = andl(mlp, u, [("+", BINA)] + [(q, f) for q, f in zip(sg, ([sx], [sy], sb))], [(EFSB, 1)])
    for sg in ("---", "++-", "+-+", "-++"):
        u = andl(mlp, u, [("+", BINA)] + [(q, f) for q, f in zip(sg, ([sx], [sy], sb))], [(ADDB, 1)])
    u = andl(mlp, u, [("+", [sy]), ("-", sb)], [(RSB, 1)])
    u = andl(mlp, u, [("-", [sy]), ("+", sb)], [(RSB, 1)])
    mach = [("+", [FON, BLKP]), ("-", [E_T, TT, F_TICK, F_TICK2, F_ERR])]  # a machine token (text_vm: fed as the
    u = andl(mlp, u, mach + [("-", [F_DONE])], [(MACH, 1)])                 # kernel's; not the reset's <done>)
    u = andl(mlp, u, mach + [("+", [F_DONE]), ("-", [RST])], [(MACH, 1)])
    assert u <= WIDTH


def l_sel(mx):
    """Result digits R (LSD-first) and sign RSG: copies, add / sub (v2), flags (-1 / 0)."""
    gd = Gated(mx, heads([0]))
    X, Y = AL["X"], AL["Y"]
    cp = [(f, 160) for f in COPY]
    cpn = cp + [(F_NEG, 160)]  # NEGATE copies the digits, flips the sign
    for c in range(ND):
        cin = lambda d, w: [(d + c - 1, w)] if c > 0 else []
        gd([(X + c, 1), (Y + c, 1)] + cin(CA, 1) + [(CA + c, -10)], 32, [(ADDB, 160)], -128, [(R + c, 1)])
        gd([(X + c, 1), (Y + c, -1)] + cin(B1, -1) + [(B1 + c, 10)], 32, [(EFSB, 160), (LTM, -160)], -128, [(R + c, 1)])
        gd([(Y + c, 1), (X + c, -1)] + cin(B2, -1) + [(B2 + c, 10)], 32, [(EFSB, 160), (LTM, 160)], -288, [(R + c, 1)])
        gd([(X + c, 1)], 32, cpn, -128, [(R + c, 1)])
    sx, sy = RS["X"], RS["Y"]
    gd([(sx, 1)], 32, [(ADDB, 160)], -128, [(RSG, 1)])
    gd([(sx, 1)], 32, [(EFSB, 160), (LTM, -160)], -128, [(RSG, 1)])
    gd([(RSB, 1)], 32, [(EFSB, 160), (LTM, 160)], -288, [(RSG, 1)])
    gd([(sx, 1)], 32, cp, -128, [(RSG, 1)])
    gd([(sx, -1)], 33, [(F_NEG, 160)], -128, [(RSG, 1)])
    gd([], 32, [(f, 160) for f in COPY + BINA + [F_NEG]], -128, [(R + k, -1) for k in range(ND)] + [(RSG, -1)])
    gd([], 32, *zand([ADDB, CA + ND - 1]), [(OVX, 1 / 32)])  # |x| + |y| over ND digits
    flag = [(R, 1 / 32), (RSG, 1 / 32)]
    for pos, neg in (([F_ZEQ, ZX], []),
                     ([F_LT, sx], [sy]), ([F_LT, LTM], [sx, sy]), ([F_LT, sx, sy, GTM], []),
                     ([F_GT, sy], [sx]), ([F_GT, GTM], [sx, sy]), ([F_GT, sx, sy, LTM], [])):
        zt, zc = zand(pos, neg)
        gd([], 32, zt, zc, flag)


def l_rstack(mx):
    """Return stack (groups 1-2): frame of depth d at row (d-1) % 128 of head (d-1) // 128; value 32 + address.
    Written by <call>, read by <exit>, cleared exactly by the next token (<A_ret> / DONE carries the address)."""
    W, o = mx.in_proj.weight, mx.out_proj.weight
    for g in (1, 2):
        for n in range(N):
            W[BO + g * N + n, RROW["W"] + n] = 32
            W[BO + g * N + n, RROW["C"] + n] = -BETA
            W[CO + g * N + n, RROW["R"] + n] = 32
    for hh, h in enumerate(heads((1, 2))[:8]):
        W[DO + h, F_CALL], W[DO + h, PREVEXIT], W[DO + h, FOPEN] = 232, 4296, BIG
        for c in range(2):
            ch = h * P + c
            for f in (RHD["W"] + hh, F_CALL, RHD["C"] + hh, PREVEXIT):
                W[XO + ch, f] = 512
            W[XO + ch, BAL] = -992
            if c == 0:
                W[XO + ch, RETV] = W[XO + ch, AV] = 1
            W[ch, RHD["R"] + hh] = W[ch, F_EXIT] = 160
            W[ch, BAL] = -288
            mx.norm.weight[ch] = 2.0**31
            o[RET, ch] = 1 if c == 0 else -1
    rows0(mx, [3])
    register(mx, heads([3])[0], [(ZX, 1)], [F_IF, F_WHILE], BR)  # IF / WHILE: branch iff the flag is 0
    register(mx, heads([3])[1], [(FINAL, 1)], STEPT, FINR)  # (the X role moves during the step's clear segments)


def l_res(mx):
    rows0(mx, [0])
    h = heads([0])[0]
    for k in range(ND + 1):
        register(mx, h, [(R + k if k < ND else RSG, 1)], RESULT, RV + k if k < ND else RVS, c=k)
    register(mx, heads([0])[1], [(OVX, 1), (TXTM, -1)], [OVX], OVFO)  # sticky: never wiped again (protocol mode)


def zsum(lits, const=0):
    """z of a gated channel on iff every literal holds: ('+', [exclusive dims]) sums to 1, ('-', dims) all 0."""
    terms = [(f, 160 if sg == "+" else -160) for sg, fs in lits for f in fs]
    return terms, -160 * sum(sg == "+" for sg, _ in lits) + 32


def l_mulgp(mx):
    """Generate / propagate of the column sums S of d * |y| (groups 4-6)."""
    hs = heads([4, 5, 6])
    for idx, (d, i) in enumerate((d, i) for d in range(1, 10) for i in range(11)):
        dec(mx, hs[idx // 7], 9 * (idx % 7), [(S[d] + i, 1)], 9, [[(PS[d] + i, 1)]] + [[(GS[d] + i, 1)]] * 8)


def l_mulcarry(mlp):
    """Carries of d * |y| in place: S[d] becomes its digits."""
    u = 0
    for d in range(1, 10):
        for k in range(1, 11):
            for j in range(k):
                u = andl(mlp, u, [("+", [GS[d] + j])] + [("+", [PS[d] + i]) for i in range(j + 1, k)],
                         [(S[d] + k - 1, -10), (S[d] + k, 1)])
    u = andl(mlp, u, [("+", [K_DIV]), ("+", [FINR])], [(DIVFIN, 1)])
    assert u <= WIDTH


def tdim(i):
    """Step operand T = 10 * (previous step result, scratch row Z) + (MOD: x_j)."""
    return (XJM, 1) if i == 0 else (AL["Z"] + i - 1, 1)


def l_tm(mx):
    """Per digit: borrow g/p of T - d|y| (MOD, 11 digits) and carry g/p of T + d|y| (*, 20 digits)."""
    hs = heads(range(7))
    k = 0
    for d in range(1, 10):
        for i in range(11):
            dec(mx, hs[k // 6], 10 * (k % 6), [tdim(i), (S[d] + i, -1)], -9, [[(BG[d] + i, 1)]] * 9 + [[(BP[d] + i, 1)]])
            k += 1
            dec(mx, hs[k // 6], 10 * (k % 6), [tdim(i), (S[d] + i, 1)], 9, [[(AP[d] + i, 1)]] + [[(AG[d] + i, 1)]] * 9)
            k += 1
    h = hs[k // 6 + 1]
    for i in range(11, ND):
        dec(mx, h, i, [tdim(i)], 9, [[(APH + i, 1)]])
    dec(mx, h, ND, [(AL["Z"] + ND - 1, 1)], 1, [[(ZOV, 1)]] * 9)  # T = 10 r drops r's digit ND - 1


def l_tmcarry(mlp):
    u = 0
    for d in range(1, 10):
        for k in range(1, 12):
            for j in range(k):
                u = andl(mlp, u, [("+", [BG[d] + j])] + [("+", [BP[d] + i]) for i in range(j + 1, k)],
                         [(BB[d] + k - 1, 1)])
        prop = lambda i: AP[d] + i if i <= 10 else APH + i
        for k in range(1, ND + 1):
            for j in range(min(k, 11)):
                u = andl(mlp, u, [("+", [AG[d] + j])] + [("+", [prop(i)]) for i in range(j + 1, k)], [(CC[d] + k - 1, 1)])
    assert u <= WIDTH, u


def l_step(mx):
    """One step of * / MOD into R (group 0): MOD R = T - q|y| (q = #{d: d|y| <= T}); * R = T + x_j |y|."""
    gd = Gated(mx, heads([0]))
    st = ("+", STEPT)
    lt = lambda d: BB[d] + 10  # T < d|y|
    for q in range(10):
        cond = [st, ("+", [K_MOD]), ("-", [DIVFIN])] + ([("+", [lt(q + 1)])] if q < 9 else []) + \
            ([("-", [lt(q)])] if q > 0 else [])
        for i in range(11):
            xt = [tdim(i)] + ([(S[q] + i, -1), (BB[q] + i, 10)] + ([(BB[q] + i - 1, -1)] if i else []) if q else [])
            gd(xt, 32, *zsum(cond), [(R + i, 1)])
    for x in range(10):
        cond = [st, ("+", [XJH + x]), ("-", [K_MOD])]
        for i in range(ND):
            xt = [tdim(i)]
            if x:
                xt += ([(S[x] + i, 1)] if i <= 10 else []) + ([(CC[x] + i - 1, 1)] if i else [])
                xt += [(CC[x] + i, -10)] if i < ND - 1 else []
            gd(xt, 32, *zsum(cond), [(R + i, 1)])
    gd([], 32, *zsum([st, ("+", [K_MOD]), ("-", [DIVFIN])]), [(R + i, -1) for i in range(11)])
    # '/': scratch S = Q 10^11 + r. Not last: R = remainder (above) + (10 Q + q) at 11..; last: R = 10 Q + q
    qv, mid, fin = [(lt(d), -1) for d in range(1, 10)], [st, ("+", [K_DIV]), ("-", [FINR])], [st, ("+", [DIVFIN])]
    gd(qv, 41, *zsum(mid), [(R + 11, 1)])  # q = 9 - #{d: T < d|y|}
    for i in range(12, ND):
        gd([(AL["Z"] + i - 1, 1)], 32, *zsum(mid), [(R + i, 1)])
    gd([], 32, *zsum(mid), [(R + i, -1) for i in range(11, ND)])
    gd(qv, 41, *zsum(fin), [(R, 1)])
    for i in range(1, 10):
        gd([(AL["Z"] + 10 + i, 1)], 32, *zsum(fin), [(R + i, 1)])
    gd([], 32, *zsum(fin), [(R + i, -1) for i in range(10)])
    gd([], 32, *zsum([st, ("-", [K_MOD])]), [(R + i, -1) for i in range(ND)])
    for x in range(1, 10):  # * step over ND digits (one OVX: the carry only when T itself fits)
        gd([], 32, *zsum([st, ("+", [XJH + x]), ("+", [CC[x] + ND - 1]), ("-", [K_MOD, ZOV])]), [(OVX, 1 / 32)])
    gd([], 32, *zsum([st, ("+", [ZOV]), ("-", [K_MOD])]), [(OVX, 1 / 32)])
    sx, sy = RS["X"], RS["Y"]
    for lits in ([("+", [sx]), ("-", [sy]), ("-", [K_MOD])], [("-", [sx]), ("+", [sy]), ("-", [K_MOD])],
                 [("+", [sx]), ("+", [K_MOD]), ("-", [K_DIV])], [("+", [sx]), ("-", [sy]), ("+", [K_DIV])],
                 [("-", [sx]), ("+", [sy]), ("+", [K_DIV])]):
        gd([], 32, *zsum([st, ("+", [FINR])] + lits), [(RSG, 1 / 32)])


def l_code(mx):
    """Code memory (address a = 16 * head + row): groups 1-2 op code + name key, read at the next cell (NPC);
    groups 3-4 name key + REPEAT's own target, read at the current cell (PC); groups 5-6 forward targets patched by
    THEN / ELSE / REPEAT, read at PC. Cells written once (dt 32, x = 32 + value, marker 32)."""
    W, o = mx.in_proj.weight, mx.out_proj.weight
    for gs, brow, crow in (((1, 2), WAL, NPCL), ((3, 4), WAL, PCL), ((5, 6), PATCHL, PCL)):
        for g in gs:
            for n in range(NL):
                W[BO + g * N + n, brow + n] = 32
                W[CO + g * N + n, crow + n] = 32
    def chan(h, c, conds, value, ztr, zc, outs):
        ch = h * P + c
        for f in conds:
            W[XO + ch, f] += 512
        W[XO + ch, BAL] += -512 * len(conds) + 32
        for f, w in value:
            W[XO + ch, f] += w
        for f, w in ztr:
            W[ch, f] += w
        W[ch, BAL] += zc
        mx.norm.weight[ch] = 2.0**31
        for d, w in outs:
            o[d, ch] += w
    cell = lambda hh: [WAH + hh]
    assert len(CODES) + 5 <= P, len(CODES)  # (op codes + name-key channels share one head per address head)
    for hh in range(NH_):
        ha, hb, hc = heads((1, 2))[hh], heads((3, 4))[hh], heads((5, 6))[hh]
        for h in (ha, hb):
            W[DO + h, STEPW] = W[DO + h, CNAME] = 232
        W[DO + hc, F_THEN] = W[DO + hc, OPC + CI[ELSE]] = W[DO + hc, OPC + CI[REPEAT]] = 232
        for h in (ha, hb, hc):  # a1: <free_h> wipes the address heads >= h (op codes, keys, patched targets)
            W[DO + h, FTH + hh] = BIG
        zn = zand([NPCH + hh, REPLAY])
        for i in range(len(CODES)):
            chan(ha, i, cell(hh) + [OPC + i], [], *zn, [(OPN + i, 1 / 32)])
        zh = (zn[0] + [(F_CALL, -160)], zn[1])
        zp = zand([PCH + hh, REPLAY, F_CALL])
        for h, c0, cp, zz in ((ha, len(CODES), len(CODES) + 3, zh), (hb, 0, 7, zp)):
            chan(h, c0, cell(hh) + [CNAME], [(KA, 1), (KCUR, 1), (E_NMPT, -512)], *zz, [(HSA, 1)])
            chan(h, c0 + 1, cell(hh) + [CNAME], [(KB, 1), (KCUR + 1, 1), (E_NMPT, -512)], *zz, [(HSB, 1)])
            chan(h, c0 + 2, cell(hh) + [CNAME], [], *zz, [(HSA, -1), (HSB, -1)])
            chan(h, cp, cell(hh) + [CNAME, E_NMPT], [(KPREV, 1)], *zz, [(HSA, 1)])  # <nmp>: the word before
            chan(h, cp + 1, cell(hh) + [CNAME, E_NMPT], [(KPREV + 1, 1)], *zz, [(HSB, 1)])
        zj = ([(PCH + hh, 160), (F_ELSE, 160), (F_REPEAT, 160), (F_LV, 160)], -288)  # ELSE / REPEAT / <lv0> jump now
        zi = zand([PCH + hh, F_END, K_IFW, BR])  # IF / WHILE with a zero flag: jump at the end of its clear run
        for c, zz in ((3, zj), (5, zi)):
            chan(hb, c, cell(hh) + [OPC + CI[REPEAT]], [(BEGSEL, 1)], *zz, [(TGT, 1)])
            chan(hb, c + 1, cell(hh) + [OPC + CI[REPEAT]], [], *zz, [(TGT, -1)])
        chan(hb, 9, cell(hh) + [OPC + CI[LEAVES[0]]], [(BLOOP, 1)], *zj, [(TGT, 1)])  # <lv0>: its loop's <begin>
        chan(hb, 10, cell(hh) + [OPC + CI[LEAVES[0]]], [], *zj, [(TGT, -1)])
        for c, zz in ((0, zj), (2, zi)):
            chan(hc, c, [PATCHH + hh] + [], [(WA, 1)] + [(f, 512) for f in PATCHF], *zz, [(TGT, 1)])
            chan(hc, c + 1, [PATCHH + hh], [(f, 512) for f in PATCHF], *zz, [(TGT, -1)])
        for c in range(4):  # patch channels: conds = head + any patch feature (exclusive)
            W[XO + hc * P + c, BAL] += -512
    gd = Gated(mx, heads([7]))
    gd([], 32, *zand([F_END, K_IFW, BR]), [(IFJ, 1 / 32)])


def l_zero(mx):
    hs = iter(heads([0]))
    h = next(hs)
    for k in range(ND):
        dec(mx, h, k, [(RV + k, 1)], 0, [[(ZC + k, 1)]])
    dec(mx, next(hs), 0, [(RV + k, 1) for k in range(ND)], 0, [[(ZERO, 1)]])


def l_hash(mx):
    """Dictionary key of the name being resolved: user name (NHA/NHB) or the code cell's (HSA/HSB)."""
    hs, neg = heads([1]), [E_NMT, E_NMPT, CLNE]  # (text names and closed words: l_f_keys, stage 2)
    decw(mx, hs[0:2], [(NHA, 1), (HSA, 1)], 0, NHROW - 1, lambda v: [(HROW + v, 1)], neg=neg)
    decw(mx, hs[2:4], [(NHB, 1), (HSB, 1)], 0, TV.P2 - 1, colout(HCOL), neg=neg)


def l_len(mlp):
    u = 0
    for n in range(1, ND + 1):
        u = andl(mlp, u, [("+", [ZC + i]) for i in range(n, ND)] + ([("-", [ZC + n - 1])] if n > 1 else []),
                 [(LR, n)])


def l_pos(mx):
    hs = iter(heads([0]))
    dec(mx, next(hs), 0, [(LR, 1), (J, -1)], 1, [[(POS + q, 1), (ANYPOS, 1)] for q in range(ND)], gate=RMODES)
    dec(mx, next(hs), 0, [(LR, 1), (J, -1)], 0, [[(DONER, 1)]], gate=RMODES)


def colout(base):
    """Column b of the dictionary as one-hot (head b // NB0, channel group b % NB0)."""
    return lambda v: [(base + (v % TV.P2) // NB0, 1), (base + NB1 + (v % TV.P2) % NB0, 1)]


DICT = [  # per column (b0): (binding features, value, read gate, outputs)
    ([PREVDEF, PREVVAR], [(DVAL, 1)], CALLNESS + [CLNE], [(DVCALL, 1)]),  # (at a close: the old value, for rebind)
    ([PREVDEF, PREVVAR], [], CALLNESS + [CLNE], [(DVCALL, -1)]),
    ([PREVDEF, PREVVAR], [(DVAL, 1)], CELLNESS, [(DVCELL, 1)]), ([PREVDEF, PREVVAR], [], CELLNESS, [(DVCELL, -1)]),
    ([PREVDEF, PREVVAR], [(KCUR + 2, 1)], [CLNE], [(DVLEN, 1)]), ([PREVDEF, PREVVAR], [(KCUR + 3, 1)], [CLNE], [(DVLCP, 1)]),
    ([PREVDEF, PREVVAR], [], [CLNE], [(DVLEN, -1), (DVLCP, -1)]),
    ([PREVDEF], [], [CLNE], [(KDEF, 1 / 32)]), ([PREVVAR], [], [CLNE], [(KVARX, 1 / 32)]),
    ([PREVVAR], [(F_CST, 512), (BAL, -512), (CK_CONST, 512)], [CLNE], [(KCONST, 1 / 32)])]  # constant: bound at <cst>
DCLR = [E_RBL] * 7 + [CK_DEF, CK_VAR, CK_VAR]  # per column: the cancel's binding feature (<rbl>: the old entry's kind)
DCV = {0: [(CHR, 16), (E_CL, 1)], 2: [(CHR, 16), (E_CL, 1)]}  # the cancel's value (the write's DVAL)


def l_dict(mx):
    """Dictionary (groups 1-3): key (a, b) -> row a, head b // NB0, channels b % NB0; per entry: value 32 + (first cell |
    cell index) read for calls (DVCALL) and fetch / store (DVCELL), and for text mode the name's length / last char and
    kind (def, variable or constant, constant), read at a word's close. Written on the first token after a definition /
    variable name (text mode: after its <nm>)."""
    W, o = mx.in_proj.weight, mx.out_proj.weight
    for g in range(1, 4):
        for n in range(NHROW):
            W[BO + g * N + n, DHROW + n] = 32
            W[CO + g * N + n, HROW + n] = 32
            W[BO + g * N + n, CDHROW + n] = -BETA
    for b1, h in enumerate(heads(range(1, 4))[:NB1]):
        W[DO + h, PREVDEF] = W[DO + h, PREVVAR] = W[DO + h, F_NTXT] = W[DO + h, DCOL + b1] = 1024
        W[DO + h, BAL] = 32 - 3072  # (bf16-exact: dt 32 exactly when written)
        W[DO + h, CDCOL + b1], W[DO + h, CDCOL2 + b1] = 4064, 2048  # a cancel: dt 4096, B -1/4 (= -(32 * 32))
        for b0 in range(NB0):
            for j, (prev, value, gate, outs) in enumerate(DICT):
                ch = h * P + len(DICT) * b0 + j
                for f in [DCOL + NB1 + b0, F_NTXT] + prev:
                    W[XO + ch, f] = 512
                W[XO + ch, BAL] = -1504
                for f, w in value:
                    W[XO + ch, f] += w
                for f, w in [(CDCOL + NB1 + b0, 512), (DCLR[j], 512)] + DCV.get(j, []):  # the cancel: the same x
                    W[XO + ch, f] += w
                for f in [HCOL + b1, HCOL + NB1 + b0] + gate:
                    W[ch, f] = 160
                W[ch, BAL] = -448
                if CALLR in gate:
                    W[ch, E_NMTOPF] = -160  # (<nmtop>: TOPA instead, l_f_c17)
                mx.norm.weight[ch] = 2.0**31
                for d, w in outs:
                    o[d, ch] = w


def l_seld(mx):
    gd = Gated(mx, heads([0]))
    for q in range(ND):
        gd([(RV + q, 1)], 32, [(POS + q, 160)], -128, [(SELD, 1)])
    gd([], 32, [(ANYPOS, 160)], -128, [(SELD, -1)])


def l_jump(mx):
    """Control emission: <A_x> / DONE for jumps, calls, returns; <@c> / <!c> for variable cells; carries / borrows
    (never while a literal / segment is being written or at a jump); overflow."""
    hs = heads(range(1, 5))
    decw(mx, hs[0:4], [(TGT, 1), (RET, 1), (DVCALL, 1)], 0, TOPRET,
         lambda v: [(AOH + v, 1)] if v < NADDR else [(DONEO, 1)] if v == TOPRET else [], gate=JMPF)
    hs = heads([5])
    dec(mx, hs[0], 0, [(DVCELL, 1)], 0, [[(FETCHO + c, 1)] for c in range(NVC)], gate=[OPN + CI[FETCHM], UN_FETCH])
    dec(mx, hs[1], 0, [(DVCELL, 1)], 0, [[(STOREO + c, 1)] for c in range(NVC)], gate=[OPN + CI[STOREM], UN_STORE])
    gd = Gated(mx, heads([6]))
    blk = MODES + JMPF + [UN_FETCH, UN_STORE]  # (no carry / tick between a name and its cell token)
    for pos_, out in (([CDUE], CARRYO), ([BLO, BHI], BORROWO), ([RCDUE], RCARRYO), ([RBLO, RBHI], RBORROWO)):
        gd([], 32, *zand(pos_, blk), [(out, 1 / 32)])
    gd([], 32, *zand([TICK2D], blk), [(TICK2O, 1 / 32)])  # machine-token count: as a carry (sp_vm.due)
    gd([], 32, *zand([TICKD], blk + [TICK2D]), [(TICKO, 1 / 32)])


def l_emit(mlp):
    """Next-token logic of the data path (0/1 dims; logits x4). Silent inside a definition: the layer-0 kind / mode
    registers see op tokens before l_cancel, and the kernel never speaks while compiling."""
    u = 0
    rule = lambda u, lits, outs: andl(mlp, u, lits + [("-", [COMPILE])], outs)
    P_, S_, C_ = ("+", [F_PR]), ("+", [F_SPL, F_SPX, F_SPC]), ("+", [F_CLR0, F_CLR1, F_CLR2, F_CLRC])
    nP_, nS_, nC_ = ("-", [F_PR]), ("-", [F_SPL, F_SPX, F_SPC]), ("-", [F_CLR0, F_CLR1, F_CLR2, F_CLRC])
    neg, pos_, zero = ("+", [RVS]), ("-", [RVS]), ("+", [ZERO])
    nz = ("-", [ZERO])
    nd = ("-", [DONER])
    u = rule(u, [P_, neg, nz], [(O["MINUS"], 1)])
    u = rule(u, [S_, neg, nz], [(O["MMINUS"], 1)])
    u = rule(u, [C_, ("+", [RS["X"]])], [(O["MMINUS"], 1)])
    for mode, start, nstart, out in ((("+", [PMODE]), P_, nP_, "PD"), (("+", [M_SPL, M_SPX, M_SPC]), S_, nS_, "MDG")):
        u = rule(u, [mode, nd, nstart], [(O[out], 1)])
        u = rule(u, [mode, nd, start, pos_], [(O[out], 1)])
        u = rule(u, [mode, nd, start, neg, zero], [(O[out], 1)])
    em, ew, ne = ("+", EMODES), ("+", [EWJ]), ("-", [EWJ])
    u = rule(u, [em, ew, nC_], [(O["MDG"], 1)])
    u = rule(u, [em, ew, C_, ("-", [RS["X"]])], [(O["MDG"], 1)])
    dn = ("+", [DONER])
    sw = lambda k: ("+", [K[k]])
    for lits, out in (([("+", [PMODE]), dn], "SPACE"), ([("+", [M_SPL]), dn, ("-", [K["SW1"], K["SW2"]])], "END"),
                      ([("+", [M_SPL]), dn, sw("SW1")], "SW2"), ([("+", [M_SPL]), dn, sw("SW2")], "CLR1"),
                      ([("+", [M_CLR1]), ne, sw("SW2")], "SW3"), ([("+", [M_CLR1]), ne, sw("SW3")], "SPL"),
                      ([("+", [M_SPX]), dn], "MSTEP"),
                      ([("+", [M_SPC]), dn], "CLR1"), ([("+", [M_CLR0]), ne], "SPL"),
                      ([("+", [M_CLR1]), ne, ("+", [K["BIN"], K["MM"], K["MS"]])], "CLR0"),
                      ([("+", [M_CLR1]), ne, ("+", [K["DROP"], K["ST"]])], "END"),
                      ([("+", [M_CLR2]), ne, ("+", [FINR])], "CLR1"), ([("+", [M_CLR2]), ne, ("-", [FINR])], "SPX"),
                      ([("+", [M_CLRC]), ne], "SPC")):
        u = rule(u, lits, [(O[out], 1)])
    u = rule(u, [("+", [K_CR])], [(O["CRNL"], 1)])
    u = rule(u, [("+", [K_SP])], [(O["SPACE"], 1)])
    first = ("-", [SEGST])
    for lits, out in (([("+", [K["CP"]])], "SPL"), ([("+", [K["PICK"]])], "CLR0"), ([("+", [K["DROP"]])], "CLR1"),
                      ([("+", [K["BIN"]])], "CLR1"), ([("+", [K["UN"]])], "CLR0"), ([("+", [K["ST"]])], "CLRC"),
                      ([("+", [K["MM"]]), ("+", [FINR])], "CLR1"), ([("+", [K["MM"]]), ("-", [FINR])], "SPX"),
                      ([("+", [K["MS"]])], "CLR2"), ([sw("SW1")], "SPL"), ([sw("SW2")], "CLR0"),
                      ([sw("SW3")], "CLR1")):
        u = rule(u, lits + [first], [(O[out], 1)])
    assert u <= WIDTH


def l_digit(mx):
    hs = iter(heads([0]))
    dec(mx, next(hs), 0, [(SELD, 1), (SELE, 1)], 0, [[(PDIG + d, 1)] for d in range(10)], gate=[O["PD"]])
    dec(mx, next(hs), 0, [(SELD, 1), (SELE, 1)], 0, [[(MDIG + d, 1)] for d in range(10)], gate=[O["MDG"]])


# ---------------- text front end (text_vm.py), piece 2a ----------------
# The model writes plain Forth text inside a ```forth fence. Its text tokens are made neutral for the kernel (F0/F1);
# the front end tracks the word being written and, at the token that closes it, emits its protocol translation (a
# number: <lit> ['-'] machine digits replayed from per-digit registers; a single-token word: its op token). Prints are
# hidden (machine digits / <psp>) and kept in a print memory (row = print, channel = char); at the closing ``` the
# kernel finishes the last word's run, then emits the output block '\n' '```' 'output' '\n' line '\n' '```'.
def keyparts(k):
    """A key (a, b) as base-8 parts (a // 64, a // 8 % 8, a % 8, the same of b): a word's sums stay small enough that
    every intermediate (with the accumulators' offset 8 per token) is <= 256, exact in bf16, for names up to 9 tokens."""
    return [k[0] // 64, k[0] // 8 % 8, k[0] % 8, k[1] // 64, k[1] // 8 % 8, k[1] % 8]


def front_embed(e):
    import itertools
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(LIGHTNING)
    feats = (E_T, E_SW, E_EW, E_NE, E_TICK, E_FORTH, E_DIG, E_DV, E_MIN, E_NZ, E_ZD, E_LCC, E_LEN, E_HNL, E_NXS, E_SNL,
             E_TNL)
    rows = {f: [] for f in feats + tuple(range(E_FQ, E_FQ + NCH)) + tuple(range(E_K1, E_K1 + NKP)) +
            tuple(range(E_INT, E_INT + NKP))}
    for t in range(1000, e.shape[0]):
        s = tok.decode([t])
        st = s.strip()
        sl = st.lower()
        dig = len(st) == 1 and st in "0123456789"
        ia = ib = 0
        for p, q in itertools.pairwise(sl):  # the token's internal character pairs (text_vm.name_key)
            x, y = TV.pair_code(p, q)
            ia, ib = ia + x, ib + y
        for f, v in ((E_T, 1), (E_SW, s[:1].isspace()), (E_EW, s[-1:].isspace()), (E_NE, bool(st)),
                     (E_TICK, st == "```"), (E_FORTH, st.lower() == "forth"), (E_DIG, dig),
                     (E_DV, int(st) if dig else 0), (E_MIN, st == "-"), (E_NZ, dig and st != "0"), (E_ZD, st == "0"),
                     (E_LCC, CHCA.get(sl[-1:], 0)), (E_LEN, min(len(sl), 255)), (E_HNL, "\n" in s),
                     (E_NXS, bool(sl) and not dig and sl != "-"), (E_SNL, s[:1] == "\n"), (E_TNL, s[-1:] == "\n")):
            if v:
                rows[f].append((t, v))
        if sl[:1] in CHC:
            rows[E_FQ + CHC[sl[0]]].append((t, 1))
        if sl:
            for i, v in enumerate(keyparts(TV.name_key(sl))):
                if v:
                    rows[E_K1 + i].append((t, v))
            for i, v in enumerate(keyparts((ia % TV.P1, ib % TV.P2))):
                if v:
                    rows[E_INT + i].append((t, v))
    for f, tv in rows.items():
        if tv:
            e[[t for t, _ in tv], f] = torch.tensor([float(v) for _, v in tv], dtype=e.dtype)
    e[LIT, E_ISLIT] = e[MMINUS, E_ISMM] = e[PSP, E_PSP] = e[TXT_NL, E_NL] = e[TXT_OUTPUT, E_OUTPUT] = 1
    e[SPACE, E_SPACE] = e[MINUS, E_VDM] = 1
    e[MMINUS, E_PCODE], e[PSP, E_PCODE], e[PNL, E_PCODE] = 10, 11, 12  # print char codes: digit d, '-', ' ', '\n'
    e[PNL, E_PNL] = e[NM, E_NMT] = e[NMP, E_NMPT] = 1
    for t in HKEY:
        e[t, E_NMT] = 1
    for t in TRTOK:
        if t not in MD and t != MMINUS:
            e[t, E_TROP] = 1  # a translation token (literal digits count by INLIT)
    e[1000:, E_NOO] = 1  # text tokens only: NOTEJ restarts at every hidden token (E_HID), so a long run never grows it
    e[TXT_OUTPUT, E_NOO] = 0
    e[:1000, E_HID] = 1
    for d in range(10):
        e[MD[d], E_ISMD], e[MD[d], E_PCODE], e[DIGITS[d], E_VDM] = 1, d, 1


def fence_x(mx, ch):
    """xBC channel ch = 32 at a token right after '```' 'forth' (conv taps t-2 / t-1, conv bias), else 0 after silu."""
    W = mx.in_proj.weight
    W[XO + ch, E_TICK], W[XO + ch, E_FORTH] = 160, -160
    mx.conv1d.weight[ch, 0, -1], mx.conv1d.weight[ch, 0, -2], mx.conv1d.weight[ch, 0, -3] = 0, -1, 1
    mx.conv1d.bias[ch] = -288


def l_front0(mx):
    """FON: inside a fence after this token (+1 at a token starting with '\n' right after '```' 'forth', wiped at
    '```'); FOPEN: that token (the block's first)."""
    W, h = mx.in_proj.weight, 0
    W[CO, BAL] = 32
    fence_x(mx, I)  # conv channel I = in_proj row XO + I = BO (group 0's B row 0): B = 32 only right after '```'
    # 'forth' (no state input elsewhere: prefill's dt floor)
    W[XO + h * P, E_SNL] = ALPHA  # +1 at a line start there
    W[DO + h, E_SNL], W[DO + h, E_TICK] = 232, BIG
    W[h * P, BAL] = 32
    mx.norm.weight[h * P] = 2.0**33
    mx.out_proj.weight[FON, h * P] = 1
    h = heads([1])[0]  # FOPEN: D-skip, gated on the line start
    fence_x(mx, h * P)
    zt, zc = zand([E_SNL])
    for f, w in zt:
        W[h * P, f] += w
    W[h * P, BAL] += zc
    mx.norm.weight[h * P] = 2.0**41
    mx.D[h] = 1
    mx.out_proj.weight[FOPEN, h * P] = 1
    rows0(mx, [2])
    register(mx, 16, [(E_LCC, 1)], [E_NE], LCR)  # last char of the last nonempty text token
    register(mx, 17, [(E_HNL, 1)], [E_T], HNLR)  # the last text token held a '\n'


def l_front1(mx):
    """Text tokens inside the fence become neutral for the kernel (as its print space: F_NRD, F_NTXT, nothing else);
    EWR: the last text token ended with whitespace; CLT: the closing '```'."""
    gd = Gated(mx, heads([0]))
    on = lambda f: zand([E_T, FON, f])
    gd([], 32, *on(F_RDIG), [(F_RDIG, -1 / 32), (F_NRD, 1 / 32), (DV, 1)])
    gd([(DV, 1)], 32, *on(F_RDIG), [(DV, -1)])
    for d in range(10):
        gd([], 32, *on(DIGH + d), [(DIGH + d, -1 / 32)])
    gd([], 32, *on(F_MINUS), [(F_MINUS, -1 / 32), (F_NRD, 1 / 32)])
    gd([], 32, *on(F_TXT), [(F_TXT, -1 / 32), (F_NTXT, 1 / 32), (KA, 1), (KB, 1)])
    gd([(KA, 1)], 32, *on(F_TXT), [(KA, -1)])
    gd([(KB, 1)], 32, *on(F_TXT), [(KB, -1)])
    gd([], 32, *zand([E_T, FON]), [(MT, 1 / 32)])  # a model token inside the fence
    gd([], 32, *zand([E_NL], [FOPEN]), [(E_NLX, 1 / 32)])  # a '\n' token that is not the block's first
    g3 = Gated(mx, heads([3]))
    g3.delay(LCR, LCP)  # the last char of the text token before this one (a closed word's last char)
    g3.delay(HNLR, EHNLP)
    g3.delay(E_OUTPUT, PREVOUT)
    rows0(mx, [1])
    register(mx, heads([1])[0], [(E_EW, 1)], [E_T], EWR)
    Gated(mx, heads([2])).gdelay(FON, [E_TICK], CLT)


def l_f_l0(mx):
    Gated(mx, [49]).delay(EWR, EWP)
    rows0(mx, [7])
    rcounter(mx, 56, [E_NLX], [FOPEN, CLT], [(FOPEN, 100), (E_TNL, 1)], NLC)  # '\n's since the closing '```' (100 +
    # inside a fence; a closing '```\n' counts as the output block's leading '\n')
    register(mx, 59, [(E_TNL, 1)], [CLT], CLNL)
    register(mx, 60, [(CLT, 1)], [E_T], ENDPH)  # from the closing ``` (a text token: one wipe, so the set is 1) to the
    # next text token (the end phase)


def l_f_l1(mlp):
    """Word close (CL) and the word registers' wipe / set / increment features; print chars."""
    u = 1024
    mt = [("+", [E_T]), ("+", [FON]), ("-", [FOPEN])]
    sw, nsw, ewp, newp = ("+", [E_SW]), ("-", [E_SW]), ("+", [EWP]), ("-", [EWP])
    u = andl(mlp, u, mt + [sw], [(CL, 1), (WRES, 1)])
    u = andl(mlp, u, mt + [nsw, ewp], [(CL, 1), (WRES, 1)])
    u = andl(mlp, u, [("+", [CLT])], [(CL, 1), (WRES, 1), (WNEW, 1)])
    u = andl(mlp, u, [("+", [FOPEN])], [(WRES, 1), (WNEW, 1)])
    u = andl(mlp, u, mt + [("+", [E_NE])], [(WNEW, 1), (MTNE, 1)])
    u = andl(mlp, u, [("+", [E_TICK]), ("-", [CLT])], [(KTICK, 1)])  # the kernel's own ``` (not the closing one)
    u = andl(mlp, u, mt + [("-", [E_NE]), sw], [(WNEW, 1)])
    u = andl(mlp, u, mt + [("-", [E_NE]), nsw, ewp], [(WNEW, 1)])
    u = andl(mlp, u, mt + [("+", [E_DIG])], [(DIGMT, 1)])
    u = andl(mlp, u, mt + [("+", [E_DIG]), nsw, ewp], [(SND, 1)])
    u = andl(mlp, u, mt + [("+", [E_DIG]), nsw, newp], [(IND, 1)])
    u = andl(mlp, u, mt + [("+", [E_MIN]), sw], [(SMIN, 1)])
    u = andl(mlp, u, mt + [("+", [E_MIN]), nsw, ewp], [(SMIN, 1)])
    u = andl(mlp, u, mt + [("+", [E_MIN]), nsw, newp], [(IMIN, 1)])
    u = andl(mlp, u, mt + [("+", [E_NZ]), nsw, ewp], [(NZD, 1)])
    u = andl(mlp, u, mt + [("+", [E_NZ]), nsw, newp], [(INZ, 1)])
    u = andl(mlp, u, mt + [("+", [E_ZD]), nsw, ewp], [(ZDC, 1)])
    u = andl(mlp, u, mt + [("+", [E_ZD]), nsw, newp], [(IZD, 1)])
    u = andl(mlp, u, [("+", [PMODE]), ("+", [E_ISMD, E_ISMM])], [(PCHAR, 1)])
    u = andl(mlp, u, [("+", [E_PSP, E_PNL])], [(PCHAR, 1)])
    ne = ("+", [E_NE])
    u = andl(mlp, u, mt + [ne, sw], [(START, 1)])  # a word's first token
    u = andl(mlp, u, mt + [ne, nsw, ewp], [(START, 1)])
    u = andl(mlp, u, mt + [ne, nsw, newp], [(CONT, 1)])  # a word's later token
    u = andl(mlp, u, mt + [ne, nsw, newp, ("-", [E_DIG])], [(CONTNX, 1)])
    u = andl(mlp, u, [("+", [FON]), ("-", [REPLAY]), ("+", [E_TROP])], [(TT, 1)])  # a translation token was fed
    u = andl(mlp, u, [("+", [FON]), ("-", [REPLAY]), ("+", [E_ISMD, E_ISMM]), ("+", [INLIT])], [(TT, 1)])
    u = andl(mlp, u, [("+", [E_NL]), ("-", [PREVOUT])], [(NLQ, 1)])  # a '\n' in the output line: next print


def l_f_l2(mx):
    """Word registers (digit count, '-' count, nonempty, last op), print / block counters, block '\n' count flags."""
    rows0(mx, [2, 3])
    rcounter(mx, 16, [IND], [WRES], [(SND, 1)], W_ND)
    rcounter(mx, 17, [IMIN], [WRES], [(SMIN, 1)], NMIN)
    register(mx, 18, [(MTNE, 1)], [WNEW], W_NE)
    rcounter(mx, 25, [INZ], [WRES], [(NZD, 1)], W_NZ)
    prints = [F_PR, F_CR, F_SPACEOP]  # <cr> / <space>: one-char prints
    counter(mx, 20, prints, [F_EPRD], [FOPEN], NPR)  # (a print that underflowed: not counted)
    counter(mx, 21, [PCHAR], [], prints, JP)
    counter(mx, 22, [E_SPACE, NLQ], [], [E_OUTPUT], QP)  # the output line: print (row) being written
    counter(mx, 24, [E_VDM], [], [E_SPACE, E_NL, E_OUTPUT], QJ)
    counter(mx, 26, [E_NOO], [], [E_OUTPUT, E_HID], NOTEJ)  # tokens since 'output' (the empty-output note: no
    # hidden token comes after 'output'); unbounded, it grew the residual rms ~1 / token: at ~80k tokens in one run
    # every feature read ~0.2% low and the address decodes flipped
    dec(mx, 32, 0, [(NLC, 1)], 0, [[(NLCK + k, 1)] for k in range(4)])
    dec(mx, 32, 6, [(NLC, 1), (CLNL, -1)], 0, [[(PRE, 1)]])  # before the output block (its leading '\n' not yet out)
    dec(mx, 32, 4, [(NLC, 1)], 0, [[(FTICK, -1)], [(FTICK, -1)]], gate=[E_TICK])  # the block's last ``` (l_f_edge +1)


def l_f_l3(mx):
    gd = Gated(mx, [53])  # the closed word, as of the token before the close
    for src, out in ((W_ND, CW_ND), (NMIN, CW_NEG), (W_NE, CW_NE), (W_NZ, CW_NZ)):
        gd.delay(src, out)
    dec(mx, 54, 0, [(W_ND, 1), (BAL, -1)], 0, [[(WR + r, 1)] for r in range(ND)], gate=[DIGMT])
    dec(mx, 54, ND, [(W_NZ, 1)], 0, [[(ZNZ, 1)]])  # no nonzero digit yet
    dec(mx, 54, ND + 1, [(W_NZ, 1)], 0, [[(LZINC, 1)]], gate=[IZD])  # a leading zero
    register(mx, 60, [(CLT, 1)], [CLT, FTICK], BLKP)  # from the closing '```' to the block's last '```'


def l_f_l4(mx):
    rows0(mx, [5, 6, 7])  # (group 4 holds l_addr2's stateless head 32: no B / C rows there)
    for r, h in enumerate(heads([5, 6, 7])[:ND]):
        register(mx, h, [(E_DV, 1)], [WR + r], DREG + r)  # the word's digit r
    rcounter(mx, heads([7])[6], [LZINC], [WRES], [(ZDC, 1)], LZ)


def l_f_l5(mx):  # (stateless heads only in groups without B / C rows: the prefill scan clamps dt to a minimum)
    gd = Gated(mx, [50])
    gd.delay(LZ, CW_LZ)
    gd.delay(ZNZ, CW_Z)
    dec(mx, 48, ND, [(NPR, 1), (QP, -1), (BAL, -1)], 0, [[(LASTP, 1)]])  # the block is at the last print
    dec(mx, 48, ND + 1, [(NPR, 1)], 0, [[(EMPTYB, 1)]], gate=[BLKP])  # nothing printed: the note
    dec(mx, 49, 0, [(CW_ND, 1)], 0, [[(CWZ, 1)]])


def l_f_l6(mx):
    """Print memory address: row = print (written at its chars, read on the block line), channel = char."""
    decw(mx, [33, 34], [(NPR, 1), (BAL, -1)], 0, NPRINT - 1, lambda v: [(RIDX + v, 1)], gate=[PCHAR])
    decw(mx, [35, 36], [(QP, 1)], 0, NPRINT - 1, lambda v: [(RIDX + v, 1)], gate=[BLKP], neg=[NLCK, NLCK + 1])
    dec(mx, 37, 0, [(JP, 1), (BAL, -1)], 0, [[(CJ + j, 1)] for j in range(22)], gate=[PCHAR])
    dec(mx, 37, 22, [(QJ, 1)], 0, [[(CJ + j, 1)] for j in range(22)], gate=[BLKP], neg=[NLCK, NLCK + 1])


def l_f_l6b(mx):
    rows0(mx, [7])
    register(mx, 56, [(CW_LZ, 1), (CW_Z, -1)], [CL], EMW_S)  # first word digit to emit (an all-zero word: its last)
    register(mx, 57, [(CW_ND, 1)], [CL], EMW_ND)  # the literal being emitted: digit count, sign, nonzero digits
    register(mx, 58, [(CW_NEG, 1)], [CL], EMW_NEG)
    register(mx, 59, [(CW_NZ, 1)], [CL], EMW_NZ)


def l_f_l7(mx):
    dec(mx, 24, 0, [(J, 1), (EMW_S, 1)], 0, [[(JS + r, 1)] for r in range(ND)])
    dec(mx, 24, ND, [(EMW_ND, 1), (EMW_S, -1), (J, -1)], 1, [[(JLT, 1)]] * ND)  # digits left to emit
    dec(mx, 25, 0, [(EMW_NZ, 1)], 0, [[(EMWZ, 1)]])  # the literal is zero: no '-'


def l_f_mem(mx):
    """Print memory (group 6, head 48): written once per char (x = 32 + code), wiped at the fence's 'forth'."""
    W, h = mx.in_proj.weight, 48
    for n in range(NPRINT):
        W[BO + 6 * N + n, RIDX + n] = W[CO + 6 * N + n, RIDX + n] = 32
    W[DO + h, PCHAR], W[DO + h, FOPEN] = 232, BIG
    for j in range(22):
        ch = h * P + j
        W[XO + ch, CJ + j] = W[XO + ch, PCHAR] = 192
        W[XO + ch, BAL], W[XO + ch, E_PCODE] = -352, 1
        W[ch, CJ + j] = W[ch, BLKP] = 160
        W[ch, NLCK] = W[ch, NLCK + 1] = -160
        W[ch, BAL] = -288
        mx.norm.weight[ch] = 2.0**31
        mx.out_proj.weight[MV, ch] = 1


def l_f_code(mx):
    dec(mx, 2, 0, [(MV, 1)], 32, [[(CODE + c, 1)] for c in range(13)])
    dec(mx, 2, 13, [(MV, 1)], 0, [[(UNW, 1)]])


TMF = ("+", [FON, BLKP])  # text mode: inside the fence, or finishing it


def l_f_fld(mlp):
    u = 1024  # FLD: the front end's literal emits machine digit J next
    lit = [TMF, ("+", [INLIT]), ("-", [REPLAY]), ("+", [JLT]), ("+", [WCOH + NUMC - 1])]
    u = andl(mlp, u, lit + [("+", [E_ISLIT]), ("-", [EMW_NEG])], [(FLD, 1)])
    u = andl(mlp, u, lit + [("+", [E_ISLIT]), ("+", [EMW_NEG]), ("+", [EMWZ])], [(FLD, 1)])
    u = andl(mlp, u, lit + [("+", [E_ISMD, E_ISMM])], [(FLD, 1)])


def l_f_seld(mx):
    gd = Gated(mx, [8])
    for r in range(ND):
        gd([(DREG + r, 1)], 32, *zand([FLD, JS + r]), [(SELD, 1)])
    gd([], 32, *zand([FLD]), [(SELD, -1)])


def tr_rules():
    """The translation table (text_vm.translate's sequences): (literals, position, token). At position TPC of the closed
    word's sequence, with the machine silent, the kernel feeds that token. A pending variable (VCUR) prefixes <@> / <!>
    <nmp>; a name resolves through the dictionary (KINDR, its length / last char checked: KVR); a built-in word is its
    op sequence unless a definition of the same name shadows it."""
    w = lambda c: ("+", [WCOH + WC[c]])
    other = lambda *cs: ("-", [WCOH + WC[c] for c in cs])
    content, vp, nvp = ("-", [CMB, INLCR]), ("+", [VCUR]), ("-", [VCUR])
    noexp = ("-", list(range(EXPKX, EXPKX + 5)))
    R = []

    def seq(lits, toks, off=0):
        R.extend((lits, off + j, t) for j, t in enumerate(toks))
    pre = [vp, content, other("")]  # also at a comment's first word: a comment ends a pending name
    seq(pre + [w("!")], [STOREM, NMP])
    seq(pre + [w("+!")], [FETCHM, NMP, ADD, STOREM, NMP])
    seq(pre + [other("!", "+!")], [FETCHM, NMP])
    seq([vp, w("")], [FETCHM, NM])  # the closing ``` after a variable name
    kv, nkv = ("+", [KVR]), ("-", [KVR])
    kinds = ([("+", [KINDR])], [("+", [KINDR + 1]), ("-", [KINDR + 2])], [("+", [KINDR + 2])])  # def, var, const
    for k, toks in enumerate(([DEF, NM], [STOREM, NM], [VAR, NM], [VAR, NM, STOREM, NM], [VAR, NM, CST, STOREM, NM])):
        decl = [content, ("+", [EXPKX + k]), other("(", "\\", "", *(CTLW if k != 1 else ()))] + \
            [("-", [NOFREE if k == 0 else VARSC])] * (k != 1)  # (a control word as the name: 'collide', word_errors)
        # the name after : to variable value constant (not when no code cell / variable cell is free: word_errors)
        if k == 1:  # to: a value / variable (else: toval, word_errors)
            seq(decl + [kv, ("+", [KINDR + 1]), ("-", [KINDR + 2])], toks)
            continue
        seq(decl + [nkv], toks)
        for h in range(12):  # a bound name again (text_vm.rebind): its old value and kind first
            seq(decl + [kv, ("+", [VHI + h])], [RBH[h]])
        for kk, kl in enumerate(kinds):
            for lo in range(16):
                seq(decl + [kv] + kl + [("+", [VLO + lo])], [RBL[kk][lo]], 1)
        seq(decl + [kv], toks, 2)
    names = other(*[c for c in WCLS if c not in B2W + SCRW + SHWC])  # a name, or a built-in word's name (a definition
    kv = ("+", [KVR])                                                  # may shadow it)
    NS = ([("-", [KVR])], [kv, ("-", [KINDR, KINDR + 1])])  # no definition / variable of that name
    cm, ncm = ("+", [CMPC]), ("-", [CMPC])
    pend = lambda k, on: ("+" if on else "-", [PENDL + k])
    for b, pv in ((0, nvp), (2, vp)):
        body = [content, pv, noexp]
        seq(body + [w("#"), ("-", [DIG21])], [LIT], b)
        seq(body + [w(";"), cm, ("-", [TOPC, STRUCTC])], [SEMI], b)
        seq(body + [names, kv, ("+", [KINDR])], [CALL, NM], b)
        seq(body + [names, kv, ("+", [KINDR + 2])], [FETCHM, NM], b)  # constant
        seq(body + [names, kv, ("+", [KINDR + 1]), ("-", [KINDR + 2]), ("+", [BLKP])], [FETCHM, NM], b)  # at ```
        for c in B2W:
            for ns in NS:
                seq(body + [w(c)] + ns, TV.WORDS[c], b)
        # piece 2c: control in place (at top level an opener first opens the hidden top-level definition)
        ndeep, sk = ("-", [CDFC]), lambda *ks: ("+", [SKC + SKW.index(k) for k in ks])
        for c in CTLW:  # (in place; LDC: do loops open, CDFC: 4 structures open, LPC: inside a loop)
            if c in ("if", "begin"):
                seq(body + [w(c), cm, ndeep], CTLSEQ[c], b)
                seq(body + [w(c), ncm], [DEF, NMTOP] + CTLSEQ[c], b)
            elif c == "do":
                for k in range(len(LSEQ)):
                    seq(body + [w(c), cm, ndeep, ("+", [LDC + k])], LSEQ[k + 1][c], b)
                seq(body + [w(c), ncm], [DEF, NMTOP] + LSEQ[1][c], b)
            elif c in ("loop", "+loop", "i"):
                for k in range(1, len(LSEQ) + 1):
                    seq(body + [w(c), ("+", [LDC + k])] + [sk("do")] * (c != "i"), LSEQ[k][c], b)
            elif c == "while":
                seq(body + [w(c), sk("begin"), ndeep], CTLSEQ[c], b)
            elif c in ALLOW:  # a closer: only after its opener (else: error)
                seq(body + [w(c), sk(*ALLOW[c])], CTLSEQ[c], b)
            elif c == "leave":
                seq(body + [w(c), ("+", [LPC])], CTLSEQ[c], b)
            else:
                seq(body + [w(c)], CTLSEQ[c], b)
        for c in SCRW:  # through the hidden scratch cells
            for ns in NS:
                seq(body + [w(c)] + ns, TV.scratch_seq(c), b)
        for k, c in enumerate(SHWC):  # hidden shared words: a call (at top level the first use defines it there)
            for ns in NS:
                seq(body + [w(c)] + ns + [cm], [CALL, NMSH[c]], b)
                seq(body + [w(c)] + ns + [ncm, ("+", [DECLL + k])], [CALL, NMSH[c]], b)
                seq(body + [w(c)] + ns + [ncm, ("-", [DECLL + k])], SHDEF[c] + [CALL, NMSH[c]], b)
        # a definition ends with ;: the hidden shared words it called first are defined right after it (gcd lcm ?dup
        # order); the top-level one ends (: variable value constant, the closing ```, an error) with its ; and the end
        # phase (end_rules: the pending definitions, its call)
        ends = [w(";"), cm, ("-", [TOPC, STRUCTC])] + body
        tops = [("+", [WCOH + WC[c] for c in (":", "variable", "value", "constant")]), ("+", [TOPC])] + body
        seq(tops, [SEMI], b)
        for lits in (ends,):
            for k in range(3):  # word k's definition, after the pending ones before it
                for combo in itertools.product((0, 1), repeat=k):
                    at = b + 1 + sum(len(SHDEF[SHWC[j]]) * x for j, x in enumerate(combo))
                    seq(lits + [pend(j, x) for j, x in enumerate(combo)] + [pend(k, 1)], SHDEF[SHWC[k]], at)
        for c, lits in word_errors(body, b):  # an error at this word (finished top-level code runs first)
            seq(lits + [("-", [TOPC])], [ERRT[c]], b)
            seq(lits + [("+", [TOPC]), ("+", [STRUCTC])], [ERRT[c]], b)  # (a structure open: not run)
            seq(lits + [("+", [TOPC]), ("-", [STRUCTC])], [SEMI], b)  # (then the end phase; l_f_final: the error)
    assert max(i for _, i, _ in R) < NTPC
    return R


ALLOW = {"else": ["if"], "then": ["if", "else"], "while": ["begin"], "repeat": ["while", "begin"], "until": ["begin"],
         "again": ["begin"], "loop": ["do"], "+loop": ["do"]}  # a closer's openers (text_vm.CLOSERS; repeat after
# begin with no while: again)


def word_errors(body, b):
    """(class, literals) of the errors a closing word raises (text_vm's ERRS), for one prefix variant (b: VCUR's)."""
    w = lambda c: ("+", [WCOH + WC[c]])
    kv, NS = ("+", [KVR]), ([("-", [KVR])], [("+", [KVR]), ("-", [KINDR, KINDR + 1])])
    cm, ncm = ("+", [CMPC]), ("-", [CMPC])
    E = [("unknown", [("-", list(range(WCOH, WCOH + len(WCLS)))), ("-", [KVR]), ("+", [NMCL])] + body)]
    E += [("digits", [("+", [WCOH + NUMC - 1]), ("+", [DIG21])] + body)]  # a literal over 20 digits
    E += [("strings", [("+", [WCOH + WC[c] for c in STRW])] + ns + body) for ns in NS]
    E += [("semi", [w(";"), ncm] + body), ("semi", [w(";"), ("+", [TOPC])] + body),
          ("semiopen", [w(";"), cm, ("-", [TOPC]), ("+", [STRUCTC])] + body),
          ("defin", [("+", [WCOH + WC[c] for c in (":", "variable", "value", "constant")]), cm, ("-", [TOPC])] + body),
          ("iout", [w("i"), ("+", [LDC]), ("-", [KVR])] + body), ("leave", [w("leave"), ("-", [LPC])] + body),
          ("deep", [("+", [WCOH + WC[c] for c in ("if", "begin", "do")]), cm, ("+", [CDFC])] + body),
          ("deep", [w("while"), ("+", [SKC + SKW.index("begin")]), ("+", [CDFC])] + body),
          ("deep", [w("do"), cm, ("-", [CDFC]), ("+", [LDC + len(LSEQ)])] + body)]  # (do loops 3 deep: text_vm.LOOPC)
    E += [("noopener", [w(c), ("-", [SKC + SKW.index(k) for k in ALLOW[c]])] + body) for c in ALLOW]
    tb = [x for x in body if x != ("-", list(range(EXPKX, EXPKX + 5)))] + [("+", [EXPKX + 1])]  # the name after to
    E += [("toval", [("-", [KVR])] + tb), ("toval", [("+", [KVR]), ("+", [KINDR])] + tb),
          ("toval", [("+", [KVR]), ("+", [KINDR + 2])] + tb)]  # not a value / variable
    nd = [x for x in body if x != ("-", list(range(EXPKX, EXPKX + 5)))] + [("-", [WCOH + WC[c] for c in ("(", "\\", "")])]
    E += [("code2", [("+", [NOFREE]), ("+", [EXPKX])] + nd),  # a ':' name with no code cell free (before any token)
          ("vars", [("+", [VARSC]), ("+", [EXPKX + 2, EXPKX + 3, EXPKX + 4])] + nd),  # a 33rd variable cell
          ("collide", [("+", [WCOH + WC[c] for c in CTLW]), ("+", [EXPKX]), ("-", [NOFREE])] + nd),  # a control word
          ("collide", [("+", [WCOH + WC[c] for c in CTLW]), ("+", [EXPKX + 2, EXPKX + 3, EXPKX + 4]), ("-", [VARSC])] + nd)]
    if b == 0:  # (@ ! +! after a variable name: the pending variable's rules)
        E += [("varword", [("+", [WCOH + WC[c] for c in ("@", "!", "+!")]), ("-", [VCUR])] + body)]
    return E


def end_rules():
    """The block's end phase (positions by EJ: l_f_final fed the top-level definition's ; there): the hidden shared
    words it called first, then its call. (At the closing ``` the last word's translation comes first, any length.)"""
    R = []
    for combo in itertools.product((0, 1), repeat=3):
        at, lits = 1, [("+", [ENDACT])] + [("+" if x else "-", [PENDL + k]) for k, x in enumerate(combo)]
        for k, x in enumerate(combo):
            if x:
                R.extend((lits, at + j, t) for j, t in enumerate(SHDEF[SHWC[k]]))
                at += len(SHDEF[SHWC[k]])
        R.extend((lits, at + j, t) for j, t in enumerate([CALL, NMTOP]))
    assert max(i for _, i, _ in R) < 64
    return R


def l_f_emit(mlp):
    """A number's sign and digits (2a); the translation tokens (suppressed in l_f_final while the machine speaks)."""
    u = 1024
    num = ("+", [WCOH + NUMC - 1])
    u = andl(mlp, u, [("+", [FLD])], [(O["MDG"], 1)])
    u = andl(mlp, u, [TMF, num, ("+", [E_ISLIT]), ("+", [INLIT]), ("-", [REPLAY]), ("+", [EMW_NEG]), ("-", [EMWZ])],
             [(O["MMINUS"], 1)])
    for lits, i, t in tr_rules():
        u = andl(mlp, u, lits + [("+", [TPL + i % 8]), ("+", [TPH + i // 8]), TMF], [(TRD + TRTOK.index(t), 1)])
    for lits, i, t in end_rules():
        u = andl(mlp, u, lits + [("+", [EJL + i % 8]), ("+", [EJH + i // 8]), TMF], [(TRD + TRTOK.index(t), 1)])
    assert u <= WIDTH, u


OLD_OUT = list(O.values()) + [OVFO, DONEO] + list(range(AOH, AOH + NADDR)) + list(range(FETCHO, FETCHO + NVC)) + \
    list(range(STOREO, STOREO + NVC)) + list(range(OPN, OPN + len(CODES)))  # the kernel speaks (carries excluded)


def l_f_final(mlp):
    """Hidden prints in text mode; carries / borrows only before something else the kernel emits (as sp_vm); translation
    tokens only while the machine is silent; the output block once the kernel is silent after the closing '```'. The
    block line: per print row its chars, a <space> row a blank, a <cr> row a newline; the line ends at the last print's
    space or past the last row (LEF -> LEP), or after the empty-output note (l_f_note)."""
    u, quiet, b = 0, ("-", OLD_OUT), ("+", [BLKP])
    for d in range(10):
        u = andl(mlp, u, [("+", [PDIG + d]), TMF], [(PDIG + d, -1), (MDIG + d, 1)])
    u = andl(mlp, u, [("+", [O["MINUS"]]), TMF], [(O["MINUS"], -1), (O["MMINUS"], 1)])
    u = andl(mlp, u, [("+", [O["SPACE"]]), TMF], [(O["SPACE"], -1), (O_PSP, 1)])
    u = andl(mlp, u, [("+", [O["CRNL"]]), TMF], [(O["CRNL"], -1), (O_PNL, 1)])
    for f in (CARRYO, BORROWO, RCARRYO, RBORROWO, TICKO, TICK2O):
        u = andl(mlp, u, [("+", [f]), quiet], [(f, -1)])
    opn = list(range(OPN, OPN + len(CODES)))  # the token cap: the next replayed cell is the error instead
    u = andl(mlp, u, [("+", [CAPF]), ("+", opn), ("-", [d for d in OLD_OUT if d not in opn])], [(ERRCAPO, 1)])
    for d in range(len(TRTOK)):  # relu(8 TR + sum(OLD) - 8)^2: off unless the translation token is up (up to 8 others)
        mlp.up_proj.weight[u, TRD + d], mlp.up_proj.weight[u, BAL] = 8, -8
        for f in OLD_OUT:
            mlp.up_proj.weight[u, f] = 1
        mlp.down_proj.weight[TRD + d, u] = -1
        u += 1
    nl, silent = (lambda k: ("+", [NLCK + k])), ("-", OLD_OUT + list(range(TRD, TRD + len(TRTOK))))
    line = [b, ("-", [NLCK, NLCK + 1, LEP, LEND])]
    alltrd = ("-", list(range(TRD, TRD + len(TRTOK))))
    for k, c in enumerate(KCLS):  # an error word that first closed and ran the top-level definition: its error now
        if c in ("underflow", "zero"):
            continue
        u = andl(mlp, u, [("+", [WERR + k]), ("+", [TOPC]), ("+", [ENDACT]), ("-", [STRUCTC, ERRST]), quiet, alltrd, TMF],
                 [(TRD + TRTOK.index(ERRT[c]), 1)])
    pre = [("+", [PRE]), ("-", [KTICK, E_OUTPUT])]  # before the output block starts (a closing '```\n': no leading
    # '\n'; its '```' and 'output' already out are not 'before')
    u = andl(mlp, u, [b, silent, ("+", [COMPILE]), ("-", [TOPOPEN, ERRST])] + pre, [(TRD + TRTOK.index(ERRT["unterm"]), 1)])
    u = andl(mlp, u, [b, silent, ("+", [TOPOPEN]), ("+", [STRUCTD]), ("-", [ERRST])] + pre,
             [(TRD + TRTOK.index(ERRT["unstruct"]), 1)])
    u = andl(mlp, u, [b, silent, ("+", [TOPOPEN]), ("-", [ERRST, STRUCTD])] + pre, [(TRD + TRTOK.index(SEMI), 1)])  # the
    # block's end phase; after an error the output block at once; the reset after an error: ; / <drop>s (l_e20: <done>)
    u = andl(mlp, u, [b, ("+", [ERRST]), ("-", [CLNL])] + pre, [(FE_NL, 1)])
    u = andl(mlp, u, [b, ("+", [ERRST]), ("+", [CLNL])] + pre, [(FE_TICKS, 1)])
    u = andl(mlp, u, [("+", [RST]), ("+", [COMPILE]), ("-", [REPLAY]), quiet], [(TRD + TRTOK.index(SEMI), 1)])
    u = andl(mlp, u, [("+", [RST]), ("-", [COMPILE, REPLAY]), ("+", [LO1]), quiet], [(TRD + TRTOK.index(DROP), 1)])
    for h, t in enumerate(FREE):  # a1: at the block's first token, when its cut 16 h is below WA (WA - 1 in head >= h)
        u = andl(mlp, u, [("+", [FOPEN]), ("+", [FREEOK]), ("+", [HCA + h]), ("+", [WAH + k for k in range(h, NH_)])],
                 [(TRD + TRTOK.index(t), 1)])
    start = [b, silent, ("-", [TOPOPEN, ERRST, COMPILE] + list(range(WERR, WERR + len(KCLS))))] + pre
    for lits, out in ((start + [("-", [CLNL])], FE_NL), (start + [("+", [CLNL])], FE_TICKS),
                      ([b, ("+", [E_NL]), nl(1), ("-", [CLNL])], FE_TICKS),
                      ([b, ("+", [KTICK]), nl(1)], FE_OUTPUT), ([b, ("+", [E_OUTPUT])], FE_NL),
                      (line + [("+", [CODE + 10])], O["MINUS"]),
                      (line + [("+", [CODE + 11]), ("-", [LASTP])], O["SPACE"]),
                      (line + [("+", [CODE + 11]), ("+", [LASTP])], FE_NL),
                      (line + [("+", [UNW]), ("-", [EMPTYB])], FE_NL), (line + [("+", [CODE + 12])], FE_NL),
                      ([b, ("+", [E_NL]), ("+", [LEP]), ("-", [ERRST])], FE_TICKNL)):
        u = andl(mlp, u, lits, [(out, 1)])
    for d in range(10):
        u = andl(mlp, u, line + [("+", [CODE + d])], [(PDIG + d, 1)])


# ---------------- text front end, piece 2b ----------------
# A word's key (text_vm.name_key) is summed over its tokens: the first token's own key (embedding E_K1), each later
# token's internal pairs (E_INT) + the pair across its left edge (l_f_edge: last char of the token before x first char).
# Sums are kept as base-16 parts (<= 255 for words up to 9 tokens), taken at the close (CSACC) and reduced mod (127, 113)
# in two decodes (l_f_s1, l_f_s2) into the key's one-hot (dictionary read row / column) and values. The closed word's
# class (l_f_cls: built-in word by key, length, last char; a number; CLT's empty word) and the name keys (KCUR: the
# last nonempty word, latched the token after its close; KPREV: the word before) drive the translation table.
def gconst(c):
    """A constant as (BAL, ONE) weights, both bf16-exact."""
    hi = float(torch.tensor(float(c), dtype=torch.bfloat16))
    assert float(torch.tensor(c - hi, dtype=torch.bfloat16)) == c - hi, c
    return [(BAL, hi), (ONE, c - hi)]


def l_f_edge(mx):
    """Layer 4: a continuing token's key piece 32 + pc(LCP, first char) + internal pairs, per part (EOUT), its length
    (LENC); the block's last ``` (FTICK)."""
    gd = Gated(mx, [33, 34, 35, 36, 37, 38, 40])
    for p in CHC:
        pc = CHCA[p]
        codes = {q: keyparts(TV.pair_code(p, q)) for q in CHC}
        for part in range(NKP):
            xt = [(E_FQ + CHC[q], codes[q][part]) for q in CHC if codes[q][part]] + [(E_INT + part, 1), (LCP, 256)]
            xt += gconst(32 - 256 * pc)
            zt = [(LCP, -256), (CONT, 160)] + gconst(256 * pc - 128)
            gd(xt, 0, zt, 0, [(EOUT + part, 1)])
    gd([(E_LEN, 1)], 32, *zand([CONT]), [(LENC, 1)])
    gd([], 32, *zand([E_TICK]), [(FTICK, 1 / 32)])


def l_f_acc(mx):
    """Layer 6: the open word's sums (key parts, length): set to its first token's own values at START (x = 2^-88 v,
    dt 2^94), + (32 + piece) at each continuing token, the 32s off one count head; WNX: non-digit tokens (a number has
    none, a '-' first excepted)."""
    rows0(mx, [4])
    W, h = mx.in_proj.weight, 32  # one head, a channel per sum (same dt): parts, length, the count of the 8s
    chans = [(SACC + k, E_K1 + k, EOUT + k) for k in range(NKP)] + [(SLEN, E_LEN, LENC), (None, None, None)]
    for c, (out, first, piece) in enumerate(chans):
        ch = h * P + c
        if first is None:
            W[XO + ch, CONT] = 8
        else:  # x = 8 + piece at a continuing token (EOUT, LENC carry 32 +): silu's error there cancels the count's
            W[XO + ch, first], W[XO + ch, piece], W[XO + ch, CONT] = 2.0**-88, 1, -24
        W[ch, BAL] = 32
        mx.norm.weight[ch] = 2.0**31
        for o in ([out] if out is not None else [SACC + k for k in range(NKP)] + [SLEN]):
            mx.out_proj.weight[o, ch] = 1 if out is not None else -1
    W[DO + h, CONT], W[DO + h, START] = 232, BIG
    rcounter(mx, 38, [CONTNX], [START], [(E_NXS, 1)], WNX)


def l_e5(mx):
    register(mx, 58, [(WA, 1)], [DNTOP], TOPA)


def l_e2(mx):
    """Layer 2 (2d-minimal): ERRST: the kernel stopped this block (from its <err..> to the next block's first token);
    ERRK: why (underflow, / or MOD by zero, unknown word)."""
    register(mx, 57, [(F_ERR, 1)], [F_ERR, FOPEN], ERRST)  # (group 7: rows0 in l_f_l0)
    register(mx, 46, [(FOPEN, 1)], [FOPEN], TXTM)  # (group 5: rows0 in l_registers; never wiped after)
    for k in range(len(KCLS)):
        register(mx, 58, [(E_EK + k, 1)], [F_ERR], ERRK + k, c=k)


def l_e7(mx):
    """Layer 7: a stack underflow at this op (depths under 128): the depth after the op's own change, shifted per op so
    that v <= 0 means too few values: popping ops (+1 for a 1-pop), pushing ops (-need + 1), the others, PICK (its index
    from LASTDIG)."""
    for h, gate, terms, k in (
            (52, [F_DROP, F_IF, F_WHILE, F_STORE, F_ADD, F_SUB, F_LT, F_GT, F_MUL, F_MOD],
             [(F_DROP, 1), (F_IF, 1), (F_WHILE, 1), (F_STORE, 1)], 0), (54, [F_PR], [(BAL, 1)], 3),
            (53, [F_DUP, F_OVER, F_SWAP], [(F_DUP, -1), (F_OVER, -2), (F_SWAP, -2)], 1),
            (58, [F_INC, F_DECR, F_ZEQ, F_NEG], [], 2), (59, [F_PICK], [(LASTDIG, -1), (BAL, -1)], 2)):
        dec(mx, h, 0, [(LO, 1)] + terms, -12, [[(UNDK + k, 1)]] * 13, gate=gate, neg=[BHI])


def l_tc16(mx):
    """Layer 16: machine tokens per block (text_vm: sp_vm tc1 tc2 tc3), group 0 (rows0 in l_res)."""
    counter(mx, 4, [MACH], [], [FOPEN], TC1, [(F_TICK, -128)])
    counter(mx, 5, [F_TICK], [], [FOPEN], TC2, [(F_TICK2, -128)])
    counter(mx, 6, [F_TICK2], [], [FOPEN], TC3)


def l_tc17(mx):
    decw(mx, [2, 3], [(TC1, 1)], 128, 255, lambda v: [(TICKD, 1)])
    decw(mx, [4, 5], [(TC2, 1)], 128, 255, lambda v: [(TICK2D, 1)])
    dec(mx, 6, 0, [(TC3, 1)], TV.CAPT, [[(CAPF, 1)]] * 52)


def reg_and(mx, h, events, level, chans):
    """A register wiped (and set) only where one of events AND level is on: dt = softplus(2^94 (e + l) - 1.5 2^94) =
    2^93 there, else 0. chans: [(sets, out)] in channels 0.. (group needs rows0)."""
    W = mx.in_proj.weight
    for f in events + [level]:
        W[DO + h, f] = BIG
    W[DO + h, BAL] = -1.5 * BIG
    for c, (sets, out) in enumerate(chans):
        for f, w in sets:
            W[XO + h * P + c, f] = w * 2.0**-82  # (dt 2^93: twice register()'s x)
        W[h * P + c, BAL] = 32
        mx.norm.weight[h * P + c] = 2.0**26
        mx.out_proj.weight[out, h * P + c] = 1


def l_cd4(mx):
    """Layer 4: CD, structures open while compiling (if / begin +1, then / repeat -1; ; and a block start reset)."""
    counter(mx, 28, [OPC + CI[IF], F_BEGIN], [F_THEN, OPC + CI[REPEAT]], [F_SEMI, FOPEN], CD)


def l_cd7(mx):
    """Layer 7: CD's level one-hots: written now (LVW), closed now (LVC), any open (STRUCTD), all 4 open (CDF)."""
    dec(mx, 60, 0, [(CD, 1)], 0, [[(CD0, 1)]] + [[(LVW + d, 1), (STRUCTD, 1)] + [(CDF, 1)] * (d == NLV - 1)
                                                for d in range(NLV)])
    dec(mx, 60, 8, [(CD, 1), (F_THEN, 1), (OPC + CI[REPEAT], 1)], 1, [[(LVC + d, 1)] for d in range(NLV)])


def l_nest(mx):
    """Layer NEST: the compile-time control stack, per level d: the if / while cell (OPENL), the else cell and whether
    there is one (ELSEL, HASEL), the <begin> address, a loop / do loop there (BEGL, LOOPL_, DOL)."""
    rows0(mx, [0, 1, 2])
    hs = iter(heads([0, 1, 2]))
    ifc, wh, el = OPC + CI[IF], OPC + CI[WHILE], OPC + CI[ELSE]
    for d in range(NLV):
        reg_and(mx, next(hs), [ifc, wh], LVW + d, [([(WA, 1), (BAL, -1)], OPENL + d)])
        reg_and(mx, next(hs), [ifc, el], LVW + d, [([(WA, 1), (BAL, -1)], ELSEL + d), ([(el, 1)], HASEL + d)])
        reg_and(mx, next(hs), [ifc, F_BEGIN], LVW + d, [([(WA, 1)], BEGL + d), ([(F_BEGIN, 1)], LOOPL_ + d),
                                                        ([(F_DOB, 1)], DOL + d)])


def l_kd17(mx):
    """Layer 17: each level's structure kind (if else begin while do), from its word's own token (l_e16's KEV)."""
    for d in range(NLV):
        reg_and(mx, 41 + d, [KEV + k for k in range(len(SKW))], LVW + d,
                [([(KEV + k, 1)], KDL + 5 * d + k) for k in range(len(SKW))])


def l_ld11(mx):
    dec(mx, 52, 0, [(LDV, 1)], 0, [[(LDOH + k, 1)] for k in range(NLV + 1)])  # do loops open
    dec(mx, 52, 8, [(LPV, 1)], 1, [[(LPF, 1)]] * NLV)  # inside a loop


def l_ld13(mx):
    for k in range(NLV + 1):  # latched at a close, for the word's translation
        register(mx, 45, [(LDOH + k, 1)], [CLX], LDC + k, c=k)
    register(mx, 45, [(LPF, 1)], [CLX], LPC, c=NLV + 1)
    register(mx, 45, [(CDF, 1)], [CLX], CDFC, c=NLV + 2)
    register(mx, 45, [(VARSFULL, 1)], [CLX], VARSC, c=NLV + 3)


def l_e8(mx):
    """Layer 8: code memory full (the cell that leaves no room for a ;), the 129th print in a block (a '.' that does not
    underflow first, a cr / space)."""
    gd = Gated(mx, [49])  # WA >= 192 from l_addr2's address one-hots (WA - 1 over 0..191): no upper bound (each
    gd.i = 3  # reset's ; past full adds 1, and WA reads inexactly in bf16 above 256); channels 0..2: the dec's below
    gd([], 32, *zand([STEPW], [F_SEMI] + [WAH + h for h in range(NH_)]), [(CFULL, 1 / 32)])  # WA >= 193
    gd([], 32, *zand([STEPW, WAH + NH_ - 1, WAL + NL - 1], [F_SEMI]), [(CFULL, 1 / 32)])  # WA = 192
    dec(mx, 49, 5, [(NVAR, 1)], NVC, [[(VARSFULL, 1)]])  # every variable cell taken (the next declaration: 'vars')
    dec(mx, 49, 6, [(WA, 1)], 0, [[(WA0F, 1)]])  # (for NOFREE: WA - 1 = -1 has no address one-hot either)
    dec(mx, 49, 1, [(NPR, 1), (TXTM, 256)], NPRINT + 257, [[(NPRF, 1)]], gate=[F_PR], neg=[UNDK + 3])  # (text mode)
    dec(mx, 49, 2, [(NPR, 1), (TXTM, 256)], NPRINT + 257, [[(NPRF1, 1)]], gate=[F_CR, F_SPACEOP])


def l_e9b(mx):
    decw(mx, [12, 13], [(CW_ND, 1)], ND + 1, ND + 128, lambda v: [(DIG21, 1)])  # a literal over 20 digits


def l_e13(mx):
    Gated(mx, [48])([], 32, *zand([CFULL]), [(STEPW, -1)])  # code memory full: the cell is not written
    gd, und = Gated(mx, [49]), [UNDK + k for k in range(4)]
    for f, out in ((F_ADD, DIGB), (F_SUB, DIGB), (F_INC, DIGU), (F_DECR, DIGU)):  # + - 1+ 1- over 20 digits (l_sel)
        gd([], 32, *zand([OVX, TXTM, f], und), [(out, 1 / 32)])
    gd([], 32, *zand([], [WA0F] + [WAH + h for h in range(NH_)]), [(NOFREE, 1 / 32)])  # WA >= 193: no free cell
    for r in (NL - 2, NL - 1):  # WA = 191, 192 (a ':' name then is 'code memory full': text_vm.name_after)
        gd([], 32, *zand([WAH + NH_ - 1, WAL + r]), [(NOFREE, 1 / 32)])


def l_e11(mx):
    """Layer 11: * / MOD with a top operand over 10 digits; / with a dividend over 10 digits (after underflow and zero)."""
    dec(mx, 49, 0, [(RW["Y"] + 10, 1)], 1, [[(MULOP, 1)]], gate=[F_MUL, F_MOD], neg=[UNDK + k for k in range(4)])
    dec(mx, 49, 1, [(RW["X"] + 10, 1), (RW["Y"] + 10, -2)], 1, [[(DIVOP, 1)]], gate=[F_DIV],
        neg=[UNDK + k for k in range(4)] + [DZ])
    und = [UNDK + k for k in range(4)]
    dec(mx, 50, 0, [(RW["X"] + k, 1) for k in range(ND)] + [(RW["Y"] + k, 1) for k in range(ND)] + [(TXTM, 64)], 85,
        [[(DIGM, 1)]] * 20, gate=[F_MUL], neg=[F_MOD, RW["Y"] + 10] + und)  # * : digits(x) + digits(y) > 20



def l_e9(mx):
    """Layer 9: / or MOD by zero (its divisor, role Y, all zero digits)."""
    dec(mx, 57, 0, [(AL["Y"] + c, 1) for c in range(ND)], 0, [[(DZ, 1)]], gate=[F_MOD], neg=[UNDK + k for k in range(4)])


def l_e16(mx):
    register(mx, 3, [(LEP, 1)], [LEP, FOPEN], LEND)  # the output line ended (an error line may follow)
    gd = Gated(mx, [13])  # a structure word's own token fed: its kind at its level (if else begin while do)
    for k, (c, f) in enumerate(zip(SKW, (OPC + CI[IF], OPC + CI[ELSE], F_BEGIN, OPC + CI[ZEQ], F_DOB))):
        gd([], 32, *zand([TT, WCOH + WC[c], f]), [(KEV + k, 1 / 32)])


def l_e17(mx):
    Gated(mx, [35])([], 32, *zand([], [LEND]), [(NLEND, 1 / 32)])


def l_e19(mx):
    counter(mx, 19, [LEND], [], [NLEND], ETJ)  # positions since the output line ended (1 at its end)
    gd = Gated(mx, [2])  # the innermost open structure's kind (its level: CD; none at CD 0)
    for d in range(NLV):
        for k in range(len(SKW)):
            gd([], 32, *zand([KDL + 5 * d + k, LVW + d]), [(SKOH + k, 1 / 32)])


def l_e19b(mx):  # (layer 17, for l_f_m18's VAL)
    rows0(mx, [5])
    preg(mx, 40, [INLCR], [CLX], [INLCRP])


def l_e20c(mx):
    dec(mx, 4, 0, [(EXPK, 1)], 1, [[(EXPKX + i, 1)] for i in range(5)], neg=[INLCRP])


def l_e20b(mx):
    for k in range(len(SKW)):  # (as of the close: the word's own translation changes SKOH)
        register(mx, 21, [(SKOH + k, 1)], [CLX], SKC + k, c=k)


def l_e20(mx):
    """Layer 20: SUPP: the kernel says nothing (an error now or this block stopped: l_f_final's lm bias); RSTD: the
    reset ends a replay; the error line's case (no prints: in the note's place / after the printed line)."""
    gd = Gated(mx, [3])
    gd([(UNDK + k, 32) for k in range(4)] + [(f, 32) for f in (DZ, ERRST, MULOP, DIVOP) + ENOWF], 0, [], 32,
       [(SUPP, 1 / 32)])
    gd([], 32, *zand([RST, REPLAY]), [(RSTD, 1 / 32)])
    for k in range(len(KCLS)):
        gd([], 32, *zand([ERRK + k, EMPTYB, ERRST]), [(EEK + k, 1 / 32)])
        gd([], 32, *zand([ERRK + k, LEND, ERRST], [EMPTYB]), [(EPK + k, 1 / 32)])


def l_e22b(mx):
    gd = Gated(mx, [60, 61, 62])
    content, noexp = ("-", [CMB, INLCR]), ("-", list(range(EXPKX, EXPKX + 5)))
    for c, lits in word_errors([content, noexp], 0):
        pos = [ds for sg, ds in lits if sg == "+"]
        neg = [d for sg, ds in lits if sg == "-" for d in ds]
        for pick in itertools.product(*pos):  # ('+' over several dims: exactly one of them)
            gd([], 32, *zand(list(pick), neg), [(WERR + KCLS.index(EALIAS.get(c, c)), 1 / 32)])


def l_e22(mx):
    """Layer 22: the error line: 'error: <why>' then the block's end (no prints: right after 'output'; else after the
    printed line's newline)."""
    hs = [h for h in list(range(16, 60)) + [2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14] if h % 8 != 7]  # (60-62: l_e22b)
    for k in range(len(KCLS)):
        outs = [[(ETX[k] + j, 1)] for j in range(NETX[k])] + [[(FE_NL, 1)], [(FE_TICKNL, 1)]]
        dec(mx, hs[2 * k], 0, [(NOTEJ, 1)], 1, outs, gate=[EEK + k], neg=[NLCK, NLCK + 1])
        dec(mx, hs[2 * k + 1], 0, [(ETJ, 1)], 1, outs, gate=[EPK + k])


def l_f_c4(mx):
    """Layer 4 (piece 2c): DECL (a hidden shared word is defined: sticky), TOPOPEN (the hidden top-level definition is
    being compiled: from its <nmtop> to its ;)."""
    for i in range(3):
        register(mx, 10 + i, [(DNSH + i, 1)], [DNSH + i], DECL + i)
    register(mx, 13, [(DNTOP, 1)], [DNDEF, F_SEMI], TOPOPEN)
    preg(mx, 19, [ERRST], [E_T], [RST])  # the block after an error: from its first token (ERRST of the 'forth')


def l_f_c7(mx):
    """Layer 7 (piece 2c): a hidden shared word called in a definition while still undefined."""
    gd = Gated(mx, [51])
    for i in range(3):
        gd([], 32, *zand([USE + i], [DECL + i]), [(USEDSET + i, 1 / 32)])


def l_f_c13(mx):
    """Layer 13 (piece 2c): the state a word's translation depends on, as of its close: definitions pending after it
    (USED), shared words defined (DECL), the top-level definition open, compiling."""
    srcs = [(USED + i, PENDL + i) for i in range(3)] + [(DECL + i, DECLL + i) for i in range(3)] + \
        [(TOPOPEN, TOPC), (COMPILE, CMPC), (STRUCTD, STRUCTC), (CLNE, NMCL)]
    for c, (src, out) in enumerate(srcs):
        register(mx, 44, [(src, 1)], [CLX], out, c)


def l_f_c15(mx):
    """Layer 15: ENDGO: the top-level definition's ; fed after the closing ``` (the block's end phase starts)."""
    gd = Gated(mx, [60])
    gd([], 32, *zand([TOPC, F_SEMI], [RST]), [(ENDGO, 1 / 32)])  # (its own ; : the top-level one; not the reset's)
    gd([], 32, *zand([TOPC, F_SEMI], [RST] + [PENDL + k for k in range(3)]), [(ENDOK, 1 / 32)])  # a1: no shared word
    # defined after it (ENDGO fires again at those words' own ;, TOPC still latched)


def l_f_c16(mx):
    register(mx, 2, [(ENDGO, 1)], [ENDGO, CLX, FOPEN], ENDACT)  # (group 0: rows0 in l_res)


def l_a1_17(mx):
    """Layer 17 (a1): FREEOK from a top-level definition's ; with no shared word pending (ENDOK) until a DEF, an error or <free_h>; HCA: its first
    cell's head, rounded up (from the address decode of WA at <nmtop>: WAH + 1, or 0 at WA 0). Group 2 (rows0 in
    l_f_keys)."""
    register(mx, 17, [(ENDOK, 1)], [ENDOK, DNDEF, F_ERR, F_FREE], FREEOK)
    for h in range(13):
        register(mx, 18, [(WAH + h - 1, 1)] if h else [(WA0F, 1)], [DNTOP], HCA + h, c=h)


def l_f_c17(mx):
    gd = Gated(mx, [34])
    gd([], 32, *zand([TT, ENDACT]), [(TTE, 1 / 32)])  # a translation token of the end phase
    gd([(TOPA, 1)], 32, *zand([UN_CALL, E_NMTOPF]), [(DVCALL, 1)])  # <call> <nmtop>: its address from TOPA
    gd([], 32, *zand([UN_CALL, E_NMTOPF]), [(DVCALL, -1)])


def l_f_c19(mx):
    counter(mx, 18, [TTE], [], [CLX], EJ)  # (group 2: rows0 in l_f_l19)


def l_f_c20(mx):
    dec(mx, 1, 0, [(EJ, 1)], 0, [[(EJL + i % 8, 1), (EJH + i // 8, 1)] for i in range(64)])


def l_f_l7b(mx):
    """Layer 7: the closed word's sums (the token before the close); CLNE: a nonempty word closes; CLX: or the closing
    ``` (an empty word: its class is CLT's)."""
    gd = Gated(mx, [57])
    for src, out in [(SACC + k, CSACC + k) for k in range(NKP)] + [(SLEN, CWLEN), (WNX, CWNX)]:
        gd.delay(src, out)
    gd([], 32, *zand([CL, CW_NE]), [(CLNE, 1 / 32), (CLX, 1 / 32)])
    gd([], 32, *zand([CLT], [CW_NE]), [(CLX, 1 / 32)])
    for f in (E_TROP, E_ISMD, E_ISMM):  # translation tokens after the closing ``` (FON is off there; l_f_l1: inside)
        gd([], 32, *zand([BLKP, f] + ([] if f == E_TROP else [INLIT]), [REPLAY]), [(TT, 1 / 32)])


def l_f_s1(mx):
    """Layer 8: M = (64 * part 0 + 8 * part 1) mod P per key half; CLNEP; the translation position counter."""
    for half, pm, hs in ((0, TV.P1, [40, 41, 44]), (1, TV.P2, [42, 43, 45])):
        for k, mul, top, hh in ((0, 64, 17, hs[:1]), (1, 8, 119, hs[1:])):
            decw(mx, hh, [(CSACC + 3 * half + k, 1)], 0, top,
                 lambda v, mul=mul, pm=pm, half=half: [(MAB + half, mul * v % pm)] if mul * v % pm else [])
    Gated(mx, [48]).delay(CLNE, CLNEP)
    counter(mx, 60, [TT], [], [CLX], TPC, steps=[(FOPEN, 64)])  # (a fence opens: no position until a word closes; a
    # word whose translation an error stopped must not resume after the reset)


def l_f_s2(mx):
    """Layer 9: the closed word's key (M + low part) mod P: dictionary read row / column (HROW / HCOL), values; TPCH."""
    decw(mx, [32, 33, 34, 35, 36, 37], [(MAB, 1), (CSACC + 2, 1)], 0, 2 * 126 + 119,
         lambda v: [(HROW + v % TV.P1, 1)] + ([(KEYA, v % TV.P1)] if v % TV.P1 else []), gate=[CLNE])
    decw(mx, [38, 40, 41, 42, 43, 44], [(MAB + 1, 1), (CSACC + 5, 1)], 0, 2 * 112 + 119,
         lambda v: colout(HCOL)(v) + ([(KEYB, v % TV.P2)] if v % TV.P2 else []), gate=[CLNE])
    dec(mx, 56, 0, [(TPC, 1)], 0, [[(TPL + i % 8, 1), (TPH + i // 8, 1)] for i in range(NTPC)])


def preg(mx, h, srcs, wipes, outs):
    """Registers in channels 0.. of head h: wiped by any wipe feature, then holding the PREVIOUS token's src (conv tap
    t-1). Group needs rows0."""
    for c, (src, out) in enumerate(zip(srcs, outs)):
        register(mx, h, [(src, 1)], wipes, out, c)
        mx.conv1d.weight[h * P + c, 0, -1], mx.conv1d.weight[h * P + c, 0, -2] = 0, 1


def l_f_cls(mx):
    """Layer 11: the closed word's class code (BWC: a built-in word by key one-hot + length * 128 + last char, a number,
    CLT's empty word), its decl code (EXPK order), '\\'; ')'-ending; KCUR (latched the token after a nonempty close)."""
    gd = Gated(mx, [56, 57])
    W = mx.in_proj.weight
    for c in B2W + SPECW + CTLW + SCRW + SHWC + STRW + ["?do"]:
        i = WC[c]
        a, bb = TV.name_key(c)
        lc = CHCA[c[-1]]
        outs = [(BWC, (i + 1) / 32)] + ([(DECLC, (SPECW.index(c) + 1) / 32)] if c in SPECW[:5] else []) + \
            ([(BSLF, 1 / 32)] if c == "\\" else [])
        gd([(CWLEN, 256 * 128), (LCP, 256), (CLNE, 32), (BAL, -256 * 128 * len(c)), (ONE, -256 * lc)], 0,
           [(HROW + a, 160), (HCOL + bb // NB0, 160), (HCOL + NB1 + bb % NB0, 160), (CWLEN, -256 * 128), (LCP, -256),
            (BAL, 256 * 128 * len(c)), (ONE, 256 * lc), (CLNE, -448)], 0, outs)
    gd([(CWNX, -256)], 32, *zand([CLNE], [CWZ]), [(BWC, NUMC / 32)])
    gd([], 32, *zand([CLT], [CW_NE]), [(BWC, EMPTYC / 32)])
    gd([(LCP, 256)] + gconst(32 - 256 * CHCA[")"]), 0,
       [(LCP, -256), (CLNE, 160)] + gconst(256 * CHCA[")"] - 128), 0, [(RPW, 1 / 32)])
    preg(mx, 26, [KEYA, KEYB, CWLEN, LCP], [CLNEP], [KCUR, KCUR + 1, KCUR + 2, KCUR + 3])
    for c in (0, 1):  # a hidden name token (<nmtop> <nmgcd> ...) carries its own key in KCUR's place
        W[26 * P + c, E_NMH] = -160
    for i in range(3):  # a hidden shared word called in this definition before it is defined (its definition follows ;)
        register(mx, 27 + i, [(USEDSET + i, 1)], [DNUSR, USEDSET + i, DNSH + i], USED + i)


def l_depth(mx):
    """Layer 11: <depth>'s result digits: the depth before the push (LO - 1; depths under 192, where HI = 0)."""
    decw(mx, [3, 4, 5], [(LO, 1), (BAL, -1)], 0, 191,
         lambda v: [(R + k, int(c)) for k, c in enumerate(reversed(str(v))) if c != "0"], gate=[F_DEPTH])


def l_f_m12(mlp):
    """Layer 12: a '\\' word closes: a line comment from here (unless its token ended the line)."""
    andl(mlp, 3000, [("+", [CLNE]), ("+", [BSLF]), ("-", [EHNLP])], [(LCSET, 1)])


def l_f_l13(mx):
    """Layer 13: KPREV (at a nonempty close: the word before), PENDA / PENDB (a name token's dictionary write key: a
    protocol name's first token, a text <nm>'s KCUR), LC (in a line comment: from its '\\' to its line's end), the
    closed word's class / decl codes (latched for its translation); LEF: the kernel ends the output line here."""
    rows0(mx, [5])
    register(mx, 40, [(KCUR, 1)], [CLNE], KPREV, 0)
    register(mx, 40, [(KCUR + 1, 1)], [CLNE], KPREV + 1, 1)
    register(mx, 41, [(NHA, 1), (KCUR, 1)], [FIRSTTXT], PENDA, 0)
    register(mx, 41, [(NHB, 1), (KCUR + 1, 1)], [FIRSTTXT], PENDB, 1)
    register(mx, 42, [(LCSET, 1)], [EHNLP, LCSET, FOPEN], LC)  # (comment state ends with its block)
    register(mx, 43, [(BWC, 1)], [CLX], WCC, 0)
    register(mx, 43, [(DECLC, 1)], [CLX], CCLS, 1)
    gd = Gated(mx, [56])
    gd([], 32, *zand([BLKP, CODE + 11, LASTP], [NLCK, NLCK + 1]), [(LEF, 1 / 32)])
    gd([], 32, *zand([BLKP, UNW], [EMPTYB, NLCK, NLCK + 1]), [(LEF, 1 / 32)])
    dec(mx, 57, 0, [(NOTEJ, 1)], NNOTE + 1, [[(LEF, 1)]], gate=[EMPTYB], neg=[ERRST, NLCK, NLCK + 1])  # (in the line)


def l_f_l15(mx):
    """Layer 15: LEP (the line ended: next the block's last ```), the latched class as one-hot."""
    Gated(mx, [57]).delay(LEF, LEP)
    decw(mx, [58, 59], [(WCC, 1)], 1, len(WCLS), lambda v: [(WCOH + v - 1, 1)])


def l_rb4(mx):
    register(mx, 27, [(E_CH, 1)], [F_NRD], CHR)  # rebind: <rbh>'s hi part, held through <rbl> (F_NRD 0 there)


def l_rb19(mx):
    rows0(mx, [4])
    register(mx, 32, [(DVCALL, 1)], [CLX], DVR)  # the closed word's dictionary value (rebind: the old entry's)


def l_rb20(mx):
    decw(mx, [9, 10, 11], [(DVR, 1)], 0, NADDR - 1, lambda v: [(VHI + v // 16, 1), (VLO + v % 16, 1)])


def l_f_keys(mx):
    """Layer 16: EXPK (the decl word before the closed word), INLCR (the closed word is in a line comment); read keys of
    text names (<nm>: KCUR, <nmp>: KPREV); the dictionary write key (PENDA / PENDB)."""
    rows0(mx, [2])
    preg(mx, 16, [CCLS, LC], [CLX], [EXPK, INLCR])  # (as of the token before the close)
    rows = lambda v: [(HROW + v, 1)]
    for src, g, hs in ((KCUR, E_NMT, (24, 25, 26, 27)), (KPREV, E_NMPT, (28, 29, 30, 32))):
        decw(mx, hs[:2], [(src, 1)], 0, NHROW - 1, rows, gate=[g])
        decw(mx, hs[2:], [(src + 1, 1)], 0, TV.P2 - 1, colout(HCOL), gate=[g])
    decw(mx, [33, 34], [(PENDA, 1)], 0, NHROW - 1, lambda v: [(DHROW + v, 1)], neg=[E_RBL])
    decw(mx, [35, 36], [(PENDB, 1)], 0, TV.P2 - 1, colout(DCOL), neg=[E_RBL])
    decw(mx, [40, 41], [(KCUR, 1)], 0, NHROW - 1, lambda v: [(CDHROW + v, 1)], gate=[E_RBL])  # rebind: the old entry
    decw(mx, [42, 43], [(KCUR + 1, 1)], 0, TV.P2 - 1, lambda v: colout(CDCOL)(v) + [(CDCOL2 + v // NB0, 1)],
         gate=[E_RBL])


def l_f_l17(mx):
    """Layer 17: EXPK as one-hot; comment events at a close (outside line comments): '(' starts one, a ')'-ending word
    ends it."""
    dec(mx, 32, 0, [(EXPK, 1)], 1, [[(EXPKOH + i, 1)] for i in range(5)])
    gd = Gated(mx, [33])
    gd([], 32, *zand([CLNE, WCOH + WC["("]], [INLCR]), [(EVT, 1 / 32), (EVTP, 1 / 32)])
    gd([], 32, *zand([CLNE, RPW], [INLCR, WCOH + WC["("]]), [(EVT, 1 / 32)])


def l_f_m18(mlp):
    """Layer 18: VAL: the closed word is a variable's name (pending: fetched or stored at the next word)."""
    nonname = [WCOH + i for i in range(len(WCLS))]  # a word named like a built-in: pending only if KV (its own entry)
    base = [("+", [CLNE]), ("+", [KVARX]), ("-", [KCONST]), ("-", [INLCR]), ("-", [BLKP])]
    u = 2048
    for nm in ([("-", nonname)], [("+", nonname), ("+", [KV])]):  # (a built-in word's name: only its own entry, D2)
        u = andl(mlp, u, base + nm + [("-", list(range(EXPKOH, EXPKOH + 5)))], [(VAL, 1)])
        u = andl(mlp, u, base + nm + [("+", list(range(EXPKOH, EXPKOH + 5))), ("+", [INLCRP])], [(VAL, 1)])


def l_f_l19(mx):
    """Layer 19: VAL and the kind, latched for the word's translation; COMMENT; KV: the dictionary entry's length and
    last char are the closed word's (else a name with the same key)."""
    rows0(mx, [2])
    register(mx, 16, [(VAL, 1)], [CLX], VALR, 0)
    for i, f in enumerate((KDEF, KVARX, KCONST)):
        register(mx, 16, [(f, 1)], [CLX], KINDR + i, 1 + i)
    register(mx, 17, [(EVTP, 1)], [EVT, FOPEN], COMMENT)



def l_kv(mx):
    """Its own layer after the dictionary: KV, the entry's length and last char are the closed word's (else a name with
    the same key), for l_f_m18's VAL (D2: a variable named like a built-in word) and the rules."""
    d = [(DVLEN, 65536), (CWLEN, -65536), (DVLCP, 256), (LCP, -256)]
    Gated(mx, [0])(d, 32, [(f, -w) for f, w in d] + [(CLNE, 160)], -128, [(KV, 1 / 32)])


def l_f_l20(mx):
    """Layer 20: VCUR (the word before left a pending variable), CMB (in a ( comment before this word), KVR."""
    rows0(mx, [2])
    preg(mx, 16, [VALR, COMMENT], [CLX], [VCUR, CMB])
    register(mx, 17, [(KV, 1)], [CLX], KVR)


def l_f_note(mx):
    """Layer 22: the empty-output note (text_vm.EMPTY, written with ``` split), then the line's end."""
    dec(mx, 8, 0, [(NOTEJ, 1)], 1, [[(NOTED + j, 1)] for j in range(NNOTE)] + [[(FE_NL, 1)]], gate=[EMPTYB],
        neg=[NLCK, NLCK + 1, LEP, ERRST])


PLAN = [(MA, [l_front0]), (MA, [l_front1]), (MA, [l_registers, l_f_l0, l_e2]), (ML, [l_cancel, l_f_l1]),
        (MA, [l_counters, l_f_l2, l_f_edge, l_f_c4, l_rb4, l_cd4]), (MA, [l_addr1, l_f_l3, l_e5]), (MA, [l_addr2, l_f_l4, l_f_acc]),
        (MA, [l_store, l_f_l5, l_addr2b, l_f_l7b, l_f_c7, l_e7, l_cd7]), (MA, [l_nest]), (MA, [l_align, l_ctrl6, l_f_l6, l_f_l6b, l_f_s1, l_e8]),
        (MA, [l_dgp, l_ctrl7, l_f_mem, l_f_l7, l_f_s2, l_e9, l_e9b]), (ML, [l_carry]),
        (MA, [l_sel, l_rstack, l_mulgp, l_f_code, l_depth, l_f_cls, l_e11, l_ld11]), (ML, [l_mulcarry, l_f_m12]), (MA, [l_tm, l_f_l13, l_f_c13, l_e13, l_ld13]),
        (ML, [l_tmcarry]), (MA, [l_step, l_code, l_f_l15, l_f_c15]), (MA, [l_res, l_hash, l_f_keys, l_f_c16, l_e16, l_tc16, l_a1_17]),
        (MA, [l_zero, l_dict, l_f_l17, l_f_c17, l_e17, l_e19b, l_tc17, l_kd17]), (MA, [l_kv]), (ML, [l_len, l_f_fld, l_f_m18]), (MA, [l_pos, l_jump, l_f_l19, l_f_c19, l_e19, l_rb19]),
        (MA, [l_seld, l_f_seld, l_f_l20, l_f_c20, l_e20, l_e20b, l_e20c, l_rb20]), (ML, [l_emit, l_f_emit]), (MA, [l_digit, l_f_note, l_e22, l_e22b]),
        (ML, [l_f_final]), (MP, [])]
STAGE1 = PLAN


def build(plan=PLAN):
    m = base([t for t, _ in plan])
    front_embed(m.model.embeddings.weight)
    for blk, (t, fns) in zip(m.model.layers, plan):
        for fn in fns:
            fn(blk.mixer)
    lm = m.lm_head.weight
    for name, t in OTOK.items():
        lm[t, O[name]] = 4
    for d in range(10):
        lm[DIGITS[d], PDIG + d] = 4
        lm[MD[d], MDIG + d] = 4
    for t, d, w in ((OVF, OVFO, 16), (CARRY, CARRYO, 8), (BORROW, BORROWO, 8), (RCARRY, RCARRYO, 7),
                    (RBORROW, RBORROWO, 7), (DONE, DONEO, 2), (TICK, TICKO, 6), (TICK2, TICK2O, 6),
                    (ERRT["tokens"], ERRCAPO, 4)):
        lm[t, d] = w
    for x in range(NADDR):
        lm[A0 + x, AOH + x] = 2
    for t, d in ((TXT_NL, FE_NL), (TXT_TICKS, FE_TICKS), (TXT_OUTPUT, FE_OUTPUT), (PSP, O_PSP), (PNL, O_PNL), (TV.TICKNL, FE_TICKNL),
                 (NL_TEXT, O["CRNL"])):
        lm[t, d] = 4
    for k, t in enumerate(TRTOK):
        lm[t, TRD + k] = 4
    from transformers import AutoTokenizer
    note = TV.TextMachine(AutoTokenizer.from_pretrained(LIGHTNING)).text(TV.EMPTY)
    assert len(note) == NNOTE, len(note)
    for j, t in enumerate(note):
        lm[t, NOTED + j] = 4
    for c in range(NVC):
        lm[fetch_tok(c), FETCHO + c] = lm[store_tok(c), STOREO + c] = 1
    for k, name in enumerate(UNDT):  # 2d-minimal: the error token at the failing op, then silence
        lm[ERRT[name], UNDK + k] = 4
    lm[ERRT["zero"], DZ] = 4
    lm[ERRT["mulop"], MULOP] = lm[ERRT["divop"], DIVOP] = 4
    for name, d in (("digits2", DIGB), ("digits2", DIGM), ("digits", DIGU), ("code", CFULL), ("prints", NPRF),
                    ("prints1", NPRF1), ("full", FULLP), ("full2", FULLM), ("rfull", RFULL)):
        lm[ERRT[name], d] = 4
    lm[DONE, RSTD] = 8  # (over the replay's next cell)
    tok_ = AutoTokenizer.from_pretrained(LIGHTNING)
    for k, txt in enumerate(ERRTXT):
        ids = TV.TextMachine(tok_).text(txt)
        assert len(ids) == NETX[k], (txt, len(ids))
        for j, t in enumerate(ids):
            lm[t, ETX[k] + j] = 4
    for code, i in CI.items():  # replay: the next cell's op token
        t = MD[code[1]] if isinstance(code, tuple) else MMINUS if code == MINUS else code  # (hidden cells: own token)
        if code not in (FETCHM, STOREM):
            lm[t, OPN + i] = 1
    for t in range(1000):  # SUPP: every hidden token the kernel can emit drops below the model's turn
        if lm[t].any() and t not in [ERRT[k] for k in ENOW]:
            lm[t, SUPP] = -16
    return m


def save(m, out):
    """save_pretrained, with the layer list written as hybrid_override_pattern (loads in transformers 5.12 .. 5.19+,
    which renamed the block types)."""
    import json
    from transformers import AutoTokenizer
    m.save_pretrained(out)
    AutoTokenizer.from_pretrained(LIGHTNING).save_pretrained(out)
    p = f"{out}/config.json"
    c = json.load(open(p))
    c["hybrid_override_pattern"] = "".join({MA: "M", ML: "-", MP: "*"}[t] for t in c.pop("layers_block_type"))
    json.dump(c, open(p, "w"), indent=2)
