# In the weights

On this page, no Python machine runs. The composite model (stock Lightning plus the compiled kernel) answers on its own. When the model closes a ` ```forth ` block, the kernel inside the same model writes the ` ```output ` block. Between the two, the kernel works through thousands of hidden tokens that no reader sees.

The run: the demo set (89 of its 90 problems), the round-1 adapter (`s2a_r1`), card v2, think 8000 / turn 16000, the Hugging Face composite at batch 1, one GPU per problem. After each run, a script checked every kernel-owned position against the spec ("exact" below).

| Result | Number |
|---|---|
| Correct | 74 of 89 (83.1%) |
| Turns exact at every kernel position | 82 of 89 |
| Kernel-owned positions | 2,748,289, of which 0.69% differed, all in 7 turns |
| Hidden kernel tokens / visible tokens | 2,735,992 / 565,391 |
| Median time per problem | 457 s (hidden steps about 114 tok/s, visible about 25 tok/s) |

The 7 inexact turns found two kernel gaps: code memory at its limit (5 turns) and over 32 variables (2 turns). Later kernel releases fixed both ([the kernel](../kernel/README.md#bugs-that-the-tests-found)). With the same adapter, card and budget, the external machine scored 76.7% on this set (4 samples per problem). One sample of 89 problems has a 95% interval of about ±8 points, so the difference is within sampling noise.

## The shortest turn

`DMbig.div_remainder`: "Calculate the remainder when 82225043186179281 is divided by 6469193." Answer: 1290731.

~~~text
We need to compute 82225043186179281 mod 6469193. Use Forth. Since numbers up to 20 digits. Both are 17 and 7 digits. We can use mod word. In Forth: 82225043186179281 6469193 mod . We'll use the machine.</think>```forth
82225043186179281 6469193 mod .
```
```output
1290731
```
The remainder is 1290731. 

\boxed{1290731}
~~~

The model wrote 169 visible tokens. The kernel wrote 297 hidden tokens and the output block. Every kernel position matched the spec. Time: 9.5 s.

## A loop of 136 steps

`L6.lcg`: "A random number generator starts at 43392 and repeatedly replaces x with (21066x + 36252) mod 161691. What is the value after 136 steps?" Answer: 12036.

~~~text
We need to compute x_136 starting from x_0 = 43392, with recurrence x_{n+1} = (21066 * x_n + 36252) mod 161691.

We need after 136 steps.
[…]
Let's write a block. We'll use variable x. Initialize x to 43392. Then loop 136 times: x = (21066 * x + 36252) mod 161691.

But careful: mod in Forth is the remainder after division. Division truncates toward zero. Since all numbers are positive, mod is fine. Multiplication: 21066 * x might be up to 21066 * 161690 ~ 3.4e9, which fits in 20 digits? 20 digits max 10^20-1. 3.4e9 is 10 digits, so fine. But the rule says: "* and mod: the top operand (pushed last) may have up to 10 digits, the other up to 20, and a product must fit in 20 digits." Here dividend might be up to 10 digits? Actually we have a*x + c. a up to 21066 (5 digits), x up to 161691 (6 digits), product up to 11 digits maybe? 21066*161691 = let's compute: 21066*161691 ≈ 3.4e9, 10 digits. So product fits in 20 digits. Then add c (5 digits). Then mod 161691 (6 digits). That's fine.

We'll just do:

variable x
43392 to x
136 0 do
   x @ 21066 * 36252 + 161691 mod x ! 
loop

But need to define 21066, 36252, 161691 as constants or just literals. We can use constants.

Let's do: 21066 constant a, 36252 constant c, 161691 constant m.

Then program:

variable x
43392 to x
136 0 do
   x @ a * c + m mod x !
loop

Then x @ . prints the result.
[…]
Let's test with a small block.</think>```forth
variable x
21066 constant a
36252 constant c
161691 constant m

43392 to x
136 0 do
   x @ a * c + m mod x !
loop

x @ .
```
```output
12036
```
The value after 136 steps is 12036. 

\boxed{12036}
~~~

The model wrote 806 visible tokens. Between the closing fence and the output, the kernel ran 48968 hidden tokens: 136 passes of a multiply, an add and a `mod` on 6-digit numbers, each one replayed from code memory. Every one matched the spec. Time: 466.6 s, most of it the hidden steps.

## An error, then a better program

`L6.compound`: "An account holds $82920. Each year it earns 1% interest, rounded down to a whole dollar, and the interest is added. How many dollars after 38 years?" Answer: 120999.

The first block leaves the stack wrong. The kernel stops at the failing word and writes the fixed error lines. The model thinks until the 8000-token budget closes its thinking. Then it sends a one-line loop: `101 100 */` multiplies by 101 and divides by 100, which gives the balance plus its rounded-down 1%.

~~~text
We need to compute the balance after 38 years, starting with $82920. Each year: interest = floor(balance * 1%). Then balance = balance + interest. Interest is 1% rounded down to a whole dollar. So each year: new_balance = balance + floor(balance * 0.01). Equivalently, new_balance = balance + floor(balance / 100). Since balance is integer dollars.
[…]
Let's write a Forth program:

```forth
82920 38 0 do
  dup 100 /  +  swap 1+ ?dup  ( maybe not needed)
loop
. 
```
```output
error: stack underflow
a word needed more values than the stack held: check what each word takes and leaves; fix the program and send the corrected block again
```
We need to be careful with the stack. Let's design a correct loop.
[…]
```forth
82920 38 0 do  101 100 */ loop  .
```
```output
120999
```
The final balance after 38 years is 120999 dollars.

\boxed{120999}
~~~

Correct. 12874 hidden tokens, every kernel position exact.

## A card sentence read literally

`DMbig.div_remainder`, a 19-digit dividend. The card (v2) said: "for a bigger number, n d mod gives the remainder of any 20-digit n directly". The model read `n d mod` as a word:

~~~text
We'll do:
```forth
9050210769528805261 34576324 n d mod . cr
```
```output
error: unknown word
a word is not defined: check its spelling, or define it with : or declare it with variable; fix the program and send the corrected block again
```
Ah, the machine might not have n d mod defined?
[…]
We'll do:
```forth
9050210769528805261 34576324 /mod . cr
```
```output
error: / dividend over 10 digits
fix the program and send the corrected block again
```
Ah! So /mod rejects dividend over 10 digits.
~~~

Then the model sent `9050210769528805261 34576324 mod . cr`, and the machine printed 27794145, the right answer. (It sent that block twice.) The error lines did their job. Card v4 still has this sentence, so it is a candidate for the next card A/B test.

## What this shows, and what it does not

The kernel gives the same tokens as the spec in almost every turn, so results measured with the external machine carry over to the weights. With the round-3 adapter in the vLLM composite (dev set, card v3, think 12000 / turn 24000), 366 of 407 problems were correct (89.9%), against 89.3% with the external machine. 395 of 407 turns were exact at every kernel position.

It is not fast. Each hidden step is a forward pass of the whole composite. The 136-step loop above took 430 s of hidden steps at batch 1 in the Hugging Face composite. In vLLM with 64 problems at once, the 10-problem demo10 workload took about 50 s per problem. A Python interpreter runs that loop in microseconds. See [the contrast](README.md#the-usual-way-and-this-way).
