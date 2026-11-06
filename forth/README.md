# The Forth dialect

Forth is a stack language. Numbers go onto a stack, and each word takes its inputs from the stack and leaves its outputs there.

This page lists what the machine supports, its limits, and its error lines. The spec is [`kernel/text_vm.py`](../kernel/text_vm.py). The model learns the dialect from the card, the short reference in its system prompt: [`training/data/cards/reference_v4.txt`](../training/data/cards/reference_v4.txt).

`tests/test_docs.py` runs every example on this page through the spec. Blocks under one heading share state, as blocks in one model turn do. Try your own:

```sh
uv run python -m forth.run '17 5 mod .' ': sq dup * ; 12 sq .'
```

## How a block runs

The model opens a block with a line that holds only ` ```forth `, and closes it with ` ``` `. Then the machine runs the block and writes an ` ```output ` block: what `.` printed, or an error. Words and variables stay defined for the later blocks of the turn. The stack starts empty in each block.

```forth
2 3 + 4 * .
```
```output
20
```

## Words

| Group | Words |
|---|---|
| Arithmetic | `+ - * / mod /mod */ negate abs 1+ 1- 2* 2/ min max gcd lcm` |
| Compare | `= <> < > <= >= 0= 0<> 0< 0> not invert and or true false` |
| Stack | `dup drop swap over rot -rot nip tuck 2dup 2drop ?dup pick depth` |
| Print | `. cr space` |
| Memory | `variable constant value @ ! +! to` |
| Words | `: ; exit` |
| Control | `if else then begin while repeat until again do ?do loop +loop i leave` |
| Comments | `( ... )` and `\` to the end of the line |

Integers only. `/` and `mod` truncate toward zero. True is -1 and false is 0.

```forth
17 5 mod .  17 5 / .  -17 5 / .
```
```output
2 3 -3
```

```forth
17 5 /mod . .
```
```output
3 2
```

```forth
1 2 3 rot . . .
```
```output
1 3 2
```

```forth
84 36 gcd .  4 6 lcm .
```
```output
12 12
```

```forth
3 4 < .  3 4 = .  true false or .
```
```output
-1 0 -1
```

## Memory

A `variable` is a named cell. A `constant` and a `value` hold a number from the start. `to` stores into a value.

```forth
variable total  0 total !  11 1 do i total +! loop  total @ .
```
```output
55
```

```forth
7 constant week  week 52 * .
```
```output
364
```

```forth
0 value best  42 to best  best .
```
```output
42
```

## Definitions and control

A definition starts with `:` and ends with `;`. A word calls itself by its own name. `exit` leaves the word, and its loop, at once.

```forth
: sq ( n -- n*n ) dup * ;  12 sq .
```
```output
144
```

```forth
5 0 do i . loop
```
```output
0 1 2 3 4
```

```forth
: first ( -- n ) 1000 1 do i i * 50 > if i exit then loop 0 ;  first .
```
```output
8
```

```forth
1 begin dup 100 < while 2* repeat .
```
```output
128
```

```forth
10 0 do i 3 = if leave then i . loop
```
```output
0 1 2
```

Fractions use a numerator and a denominator. This reduces 6/8:

```forth
6 8 2dup gcd tuck / -rot / . .
```
```output
3 4
```

## Limits

The limits come from the compiled kernel. The spec has the same limits, so the model trains on what the weights do.

| Limit | Value |
|---|---|
| Number size | 20 digits. `+` and `-` work on any of them. |
| `*` and `mod` | The top operand (pushed last) has at most 10 digits. A product must fit in 20 digits. |
| `/` and `/mod` | The dividend has at most 10 digits. `n d mod` takes a 20-digit `n`. |
| Variables | 32 named cells (`variable`, `value`, `constant`) per turn |
| Code memory | 192 cells per turn. Top-level code is freed after it runs. Definitions stay. |
| Stack | 1024 values, and 1024 nested calls |
| Prints | 128 per block (`.`, `cr` and `space` each count) |
| Steps | 196608 kernel steps per block |
| Nesting | 4 open structures (`if`, `begin`, `do`) at once |

A 21-digit result is an error, not a wrong number:

```forth
99999999999999999999 1 + .
```
```output
error: a number over 20 digits
fix the program and send the corrected block again
```

The top-operand rule matters. This factorial puts the big partial product on top, so `20 fact` fails:

```forth
: fact ( n -- n! ) dup 1 > if dup 1- fact * then ;  20 fact .
```
```output
error: * / MOD top operand over 10 digits
fix the program and send the corrected block again
```

The same word with the small loop index on top works:

```forth
: fact ( n -- n! ) 1 swap 1+ 1 do i * loop ;  20 fact .
```
```output
2432902008176640000
```

A big dividend works with `mod`:

```forth
12345678901234567890 9876543210 mod .
```
```output
7253086410
```

## Code memory

Top-level code (a loop outside a definition) runs once, so the machine frees it at the next block. A definition stays until the turn ends. If the model sends the same definition again, the new copy takes new cells.

This block can run 25 times in one turn, because its loop is top-level code. (One run shown.)

```forth
0 100 0 do i + loop .
```
```output
4950
```

This block defines `s` and then calls it. If the model sends it again and again in one turn, the eighth copy fills code memory. So the model must not re-send code that already worked. Training taught it that habit ([training](../training/README.md#round-3-clean-on-policy-rows)).

```forth
: s 0 100 0 do i + loop ; s .
```
```output
4950
```

## Errors

An error stops the block. The output shows what was printed before the error, then two fixed lines: `error: <why>` and what to do. Words and variables from earlier blocks stay.

```forth
1 0 /
```
```output
error: / or MOD by zero
the divisor was 0: check the values that reach / or mod; fix the program and send the corrected block again
```

```forth
drop
```
```output
error: stack underflow
a word needed more values than the stack held: check what each word takes and leaves; fix the program and send the corrected block again
```

```forth
x @ .
```
```output
error: unknown word
a word is not defined: check its spelling, or define it with : or declare it with variable; fix the program and send the corrected block again
```

```forth
." hello"
```
```output
error: strings and emit are not supported: the machine prints numbers only (. prints one; cr, space for layout)
fix the program and send the corrected block again
```

```forth
begin again
```
```output
error: the block ran too long (over 196608 kernel steps)
an endless loop or too many iterations: check each loop's exit condition; fix the program and send the corrected block again
```

The full list of error classes is `ERRS` in [`kernel/text_vm.py`](../kernel/text_vm.py). Some classes add a hint to the second line (`RETRY`).

## Why the errors are short

At first the machine wrote long error reports: the failing word, the stack, a hint. To write those, the kernel must copy text, and text copy is the hardest mechanism to compile into weights. So we tested first. On the 90-problem demo set, 4 samples each, stock Lightning with the card scored 71.9% with full reports and 72.8% with one-line errors. Recovery after an error was 58.7% and 57.5%. The short form cost nothing, so the spec kept it. We added the fixed retry line after reading transcripts: it turns an error into "fix it and send it again".

## What we dropped, and why

The rule is parity first. If the kernel cannot do a feature exactly, the spec drops it. Then the training data, which the spec writes, matches what the weights do.

| Dropped | Why |
|---|---|
| `.s`, `spaces`, `>r`, `r>`, `r@` | The kernel never had them. Fewer than 2.5% of machine rollouts used them, and those were mostly wrong. |
| `recurse` | A word calls itself by its own name. One spelling is enough. |
| Strings (`."`, `emit`) | The machine prints numbers only. Strings need a text-copy mechanism. |
| Re-send rollback | A block that re-defines a name used to roll back code memory. It changed visible output in 3.4% to 8.7% of rollouts, mostly failing ones, and the kernel never had it. Freeing top-level code replaced most of its benefit. |
| Name versions | A name has one binding. Redefining a name rebinds it everywhere. Real programs never used the old binding. |
| Long error reports | See above: no accuracy loss with one line. |
| Big-number types | "Just make the number bigger": 20 digits in one stack row, no limbs. |

## Python helpers in this folder

| File | Use |
|---|---|
| [`run.py`](run.py) | Run blocks through the spec from the command line |
| [`sp.py`](sp.py) | Standard Forth to kernel tokens, with the kernel's limits; loads the pinned Lightning tokenizer |
| [`ref.py`](ref.py) | A small independent Forth interpreter, used to check the spec |
| [`curriculum.py`](curriculum.py) | Seeded problem generators with reference programs and exact answers |
| [`check_programs.py`](check_programs.py) | Check that example programs use only supported words and reproduce in `ref.py` |
