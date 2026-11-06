"""Reference Forth for the prize programs (Hanoi, the dubious blackjack table): ONLY the words they use.

Signed arbitrary-precision integers; `/` and MOD truncate toward zero (SM/REM, as the kernel); flags are -1/0.
Definitions are `: NAME ... ;` (recursion by name); VARIABLE NAME creates a cell; `@`/`!` fetch/store.
Also records the peak data/return stack depth, which sizes the kernel.
"""

import re
import sys

sys.setrecursionlimit(20_000)  # Forth calls nest as Python frames; the kernel's return stack holds 1024 frames

PROGRAMS = {
    # Pegs 1-3, so the spare peg is 6-from-to: 3 values per recursion level. Prints each move as "from to".
    "hanoi": """
: HANOI ( n from to -- )
  2 PICK 0= IF DROP DROP DROP EXIT THEN
  2 PICK 1-  2 PICK  6 3 PICK - OVER -  HANOI     ( move n-1 from -> spare )
  OVER . DUP .                                    ( move disk n from -> to )
  2 PICK 1-  6 2 PICK - 3 PICK -  2 PICK  HANOI   ( move n-1 spare -> to )
  DROP DROP DROP ;
""",
    # Infinite deck from a linear congruential generator; aces only count 1 and the dealer hits below 17 (dubious).
    # Output: DEAL prints player total, dealer up card; HIT prints card, player total; STAND prints dealer total
    # then the result (1 player wins, 0 push, -1 dealer wins).
    "blackjack": """
VARIABLE SEED  VARIABLE PLAYER  VARIABLE DEALER
: RAND ( -- n ) SEED @ 1103515245 * 12345 + 2147483648 MOD DUP SEED ! ;
: CARD ( -- v ) RAND 13 MOD 1+ DUP 10 > IF DROP 10 THEN ;
: DEAL ( -- ) CARD CARD + DUP PLAYER ! . CARD DUP DEALER ! . ;
: HIT ( -- ) CARD DUP . PLAYER @ + DUP PLAYER ! . ;
: RESULT ( -- )
  PLAYER @ 21 > IF -1 . EXIT THEN
  DEALER @ 21 > IF 1 . EXIT THEN
  PLAYER @ DEALER @ - DUP 0= IF . EXIT THEN
  0 > IF 1 ELSE -1 THEN . ;
: STAND ( -- ) BEGIN DEALER @ 17 < WHILE CARD DEALER @ + DEALER ! REPEAT DEALER @ . RESULT ;
""",
}


class Forth:
    def __init__(self):
        self.stack, self.rstack, self.out, self.words, self.vars, self.memory = [], [], [], {}, {}, []
        self.peak = {"data": 0, "return": 0}

    def push(self, v):
        self.stack.append(v)
        self.peak["data"] = max(self.peak["data"], len(self.stack))

    def pop(self):
        return self.stack.pop()

    def rpush(self, v):
        self.rstack.append(v)
        self.peak["return"] = max(self.peak["return"], len(self.rstack))

    def run(self, source):
        tokens = re.sub(r"\( [^)]*\)", " ", source).split()
        i = 0
        while i < len(tokens):
            t = tokens[i].upper()
            if t == ":":
                end = tokens.index(";", i)
                self.words[tokens[i + 1].upper()] = [w.upper() for w in tokens[i + 2:end]]
                i = end + 1
                continue
            if t == "VARIABLE":
                self.vars[tokens[i + 1].upper()] = len(self.memory)
                self.memory.append(0)
                i += 2
                continue
            self.execute([t])
            i += 1
        return " ".join(self.out)

    def execute(self, body):
        """Run a word list; returns 'exit' if EXIT was hit. Control flow: IF ELSE THEN, BEGIN UNTIL/WHILE REPEAT."""
        pc = 0
        while pc < len(body):
            w = body[pc]
            if w == "EXIT":
                return "exit"
            if w == "IF":
                if self.pop() == 0:  # false: continue after the matching ELSE or THEN
                    pc = self._skip(body, pc, ("ELSE", "THEN")) + 1
                    continue
            elif w == "ELSE":
                pc = self._skip(body, pc, ("THEN",)) + 1
                continue
            elif w in ("THEN", "BEGIN"):
                pass
            elif w == "UNTIL":
                if self.pop() == 0:
                    pc = self._back(body, pc)
                    continue
            elif w == "WHILE":
                if self.pop() == 0:
                    pc = self._skip(body, pc, ("REPEAT",)) + 1
                    continue
            elif w == "REPEAT":
                pc = self._back(body, pc)
                continue
            elif w in self.words:
                self.rpush(("ret", w))  # the kernel's return stack holds return addresses too
                self.execute(self.words[w])
                self.rstack.pop()
            else:
                self.prim(w)
            pc += 1

    @staticmethod
    def _skip(body, pc, stops):
        depth = 0
        for j in range(pc + 1, len(body)):
            if body[j] in ("IF", "BEGIN"):
                depth += 1
            elif body[j] in ("THEN", "UNTIL", "REPEAT") and depth:
                depth -= 1
            elif body[j] in stops and depth == 0:
                return j
        raise SyntaxError(f"unterminated control at {pc}")

    @staticmethod
    def _back(body, pc):
        depth = 0
        for j in range(pc - 1, -1, -1):
            if body[j] in ("UNTIL", "REPEAT"):
                depth += 1
            elif body[j] == "BEGIN":
                if depth == 0:
                    return j + 1
                depth -= 1
        raise SyntaxError("BEGIN not found")

    def prim(self, w):
        if re.fullmatch(r"-?\d+", w):
            return self.push(int(w))
        if w in self.vars:
            return self.push(self.vars[w])
        div = lambda a, b: abs(a) // abs(b) * (-1 if (a < 0) != (b < 0) else 1)
        binary = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b, "/": div,
                  "MOD": lambda a, b: a - b * div(a, b), "=": lambda a, b: -(a == b), "<": lambda a, b: -(a < b),
                  ">": lambda a, b: -(a > b)}
        if w in binary:
            b, a = self.pop(), self.pop()
            return self.push(binary[w](a, b))
        s = self.stack
        if w == "DUP":
            self.push(s[-1])
        elif w == "DROP":
            s.pop()
        elif w == "SWAP":
            s[-1], s[-2] = s[-2], s[-1]
        elif w == "OVER":
            self.push(s[-2])
        elif w == "ROT":
            s.append(s.pop(-3))
        elif w == "PICK":
            u = s.pop()
            self.push(s[-1 - u])
        elif w == "NEGATE":
            s[-1] = -s[-1]
        elif w == "1+":
            s[-1] += 1
        elif w == "1-":
            s[-1] -= 1
        elif w == "0=":
            s[-1] = -(s[-1] == 0)
        elif w == ">R":
            self.rpush(self.pop())
        elif w == "R>":
            self.push(self.rstack.pop())
        elif w == "R@":
            self.push(self.rstack[-1])
        elif w == "@":
            self.push(self.memory[self.pop()])
        elif w == "!":
            addr = self.pop()
            self.memory[addr] = self.pop()
        elif w == ".":
            self.out.append(str(self.pop()))
        else:
            raise NameError(w)
