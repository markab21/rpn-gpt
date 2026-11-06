# Before and after training

The same problem, the same card, the same budget: stock Lightning against a trained adapter. Every transcript is a real rollout from the frozen test, with the external machine (the spec) writing the output blocks. Long thinking is cut at the `[…]` marks. `</think>` marks the end of the model's thinking.

## The model has the right answer and does not trust it

Problem (`DMbig.div_remainder`, hard): "Calculate the remainder when 8191249343249617212 is divided by 3423811." Answer: 2599017.

Stock Lightning, card v2, think 12000 / turn 24000. It spends its first paragraphs on the card's wording. It sends the same block with a word that does not exist, three times. Then it sends the right block, and the machine prints the right answer. The model does not believe it. It starts a long division by hand and reaches the 24,000-token cap with no answer:

~~~text
We are asked to calculate the remainder when 8191249343249617212 is divided by 3423811.

This is a simple modular arithmetic problem. We need to compute 8191249343249617212 mod 3423811.

We can do this by repeated subtraction or using long division in our head, but it's large numbers. We can use properties or just do division step by step. Since I'm an AI, I can compute it directly, but the instruction says: "You are a mathematician with an exact Forth machine: write a forth block and the machine runs it when you close the block, then prints a output block for you to read. Use it only when the computation needs it; otherwise just answer."
[…]
```forth
8191249343249617212 3423811 n dmod .
```
```output
error: unknown word
a word is not defined: check its spelling, or define it with : or declare it with variable; fix the program and send the corrected block again
```
[…]
```forth
8191249343249617212 3423811 mod .
```
```output
2599017
```
[…]
3,423,811 * 40 = 136,952,440
149,822,890 - 136,952,440 = 12,870,450
3,423,811 * 3 = 10,271,433
12,870,450 - 10,271,433 = 2,599,017
3,423,811 * 0 = 0, remainder 2,599,017.
[…]
But wait, we also had the initial subtraction of 1,900,000 from the total? Let's trace back.
~~~

Result: no `\boxed` answer, 24,000 tokens. The other stock sample also ran to the cap.

The round-1 adapter (`s2a_r1`), same settings. One block, read, answered:

~~~text
We are asked: "Calculate the remainder when 8191249343249617212 is divided by 3423811." So we need to compute 8191249343249617212 mod 3423811.

We can use the Forth machine to compute this. Since numbers can have up to 20 digits, and divisor has up to maybe 7 digits. The dividend is 19 digits? Let's count: 8191249343249617212. That's 19 digits. Divisor: 3423811 is 7 digits. So we can compute mod using /mod or mod. Forth has mod. But we need to ensure the dividend fits in 20 digits; it does. We'll use mod.

We need to write a Forth block that does this. Let's set up the numbers. In Forth, we can just use numbers directly. But careful: the numbers might be large, but within 20 digits. The dividend: 8191249343249617212. Divisor: 3423811.

We can just write a block that does: 8191249343249617212 3423811 mod . or something. But we need to output the remainder. The problem says: "Calculate the remainder when 8191249343249617212 is divided by 3423811." So we need to output the remainder. We can compute mod and print it.

Let's write a Forth block:

```forth
8191249343249617212 3423811 mod . cr
```
```output
2599017 

```
But wait, we need to ensure that the machine processes it correctly. Let's test. The dividend is 8191249343249617212. That's 19 digits. The divisor 3423811. Mod should work. The result should be the remainder.

But we must be careful: the Forth machine uses integers. mod in standard Forth truncates toward zero. Since both numbers are positive, remainder will be positive and less than divisor.

Let's compute manually maybe to verify? We can do it step by step, but it's easier to just use the machine.

I'll write the block.</think>```forth
8191249343249617212 3423811 mod . cr
```
```output
2599017 

```
The remainder is 2599017. 

\boxed{2599017}
~~~

Result: correct, 663 tokens.

## The model computes by hand when a program is the better way

Problem (`L6.lcg`, hard): "Let x_0 = 27506 and x_(k+1) = (86756 * x_k + 99314) mod 127181. What is x_123?" Answer: 17405.

Stock Lightning writes no Forth at all. It thinks for about 19,600 characters, tries a closed form with modular powers by hand, and gives a wrong answer. Here the thinking is cut to its first lines and the final answer:

~~~text
We need to compute x_123 where x_0 = 27506, and x_{k+1} = (86756 * x_k + 99314) mod 127181. This is a linear congruential generator.

We need to compute x_123 mod 127181.
[…]
</think>

The recurrence is \(x_{k+1} = (86756 x_k + 99314) \bmod 127181\) with \(x_0 = 27506\). We compute \(x_{123}\) using closed-form formulas and modular exponentiation.

Since \(\gcd(86755, 127181) = 1\), the inverse of \(86755\) modulo \(127181\) is \(57267\). Using fast exponentiation for \(a = 86756\) modulo \(127181\):
\[
\begin{aligned}
a^1 &\equiv 86756 \\
a^2 &\equiv 31956 \\
a^4 &\equiv 49687 \\
a^8 &\equiv 87578 \\
a^{16} &\equiv 1517 \\
a^{32} &\equiv 12031 \\
a^{64} &\equiv 12983
\end{aligned}
\]
Then \(a^{123} = a^{64+32+16+8+2+1} \equiv 12983 \cdot 12031 \cdot 1517 \cdot 87578 \cdot 31956 \cdot 86756 \pmod{127181}\) reduces stepwise to \(74662 \cdot 86756 \bmod 127181 = 6477376472 \bmod 127181 = 364\) (after multiplying by the inverse and \(x_0\)). Finally,
\[
x_{123} \equiv 364 \cdot 57267 \cdot (86756^{123} - 1) \text{ wait — using } x_n = a^n x_0 + c(a^n-1)(a-1)^{-1} \bmod m
\]
after full evaluation gives \(x_{123} = 91418\).

\[
\boxed{91418}
\]
~~~

Result: 91418, wrong, 12,505 tokens. The second sample answered 104141, also wrong.

The round-1 adapter writes a four-line loop:

~~~text
We need to compute x_123 given recurrence: x_{k+1} = (86756 * x_k + 99314) mod 127181, with x_0 = 27506.

This is a linear congruential generator (LCG). We need x_123 mod 127181.
[…]
Let's use the Forth machine. The machine has standard arithmetic. We can do a loop: initialize x = 27506, then 123 times: x = (86756 * x + 99314) mod 127181. We need mod. The machine has mod word. Also * and +. We can do:

27506 constant x0? But variables: we can use a variable. Let's set up:

variable x
27506 to x

Then loop 123 times
[…]
Let's write the Forth block.

```forth
variable x
27506 to x
123 0 do x @ 86756 * 99314 + 127181 mod to x loop
x @ .
```
```output
17405
```
Wait, output says 17405. Let me verify. Actually, I need to ensure the loop runs correctly. The do loop: 123 0 do ... loop will iterate 123 times (from 0 to 122 inclusive). That's 123 iterations. Good.

But output shows 17405. Let's double-check with maybe a quick mental check? Not necessary, trust the machine. However, I'll re-run to be safe. Actually, we can also compute using matrix exponentiation but 123 is small. The machine result is 17405.

I should output final answer in \boxed{17405}.</think>```forth
variable x
27506 to x
123 0 do x @ 86756 * 99314 + 127181 mod to x loop
x @ .
```
```output
17405
```
\boxed{17405}
~~~

Result: correct, 888 tokens. The adapter ran the block once inside its thinking and once in its answer. Training later reduced such re-runs.

## The model fights the stack

Problem (`L6.euclid_trace`, small): "Euclid on 2426035688 and 4967564154: print the second number, each nonzero remainder in order, and finally the gcd."

The round-3 adapter (`s2a_r3`), card v4. It knows the algorithm, but its programs keep breaking: stack underflow, a structure that does not close, an endless loop. Both samples ran to the 24,000-token cap. Here are its start and one of its later blocks:

~~~text
We need to compute the Euclidean algorithm on 2426035688 and 4967564154. The instruction: "print the second number, each nonzero remainder in order, and finally the gcd." So we start with 4967564154, then the sequence of remainders until 0, and then the gcd (which is the last nonzero remainder). We need to output these numbers. The machine is a Forth-like system; we can write a program to compute the Euclidean algorithm and print the second number, each nonzero remainder in order, and finally the gcd. The output should be the numbers printed one per line? The instruction says "print the second number, each nonzero remainder in order, and finally the gcd." Likely we should output them in some format.
[…]
Plan with variables:

```forth
variable first
variable second

2426035688 to first
4967564154 to second

( Print the second number )
second @ . cr

( Loop: while second != 0 )
begin
  ( compute remainder = first mod second )
  first @ second @ mod
  ( stack has remainder )
  ( if remainder == 0, exit loop; gcd is current second, already printed? )
  ( But we need to check zero without messing stack. Maybe we can use the loop condition? )
  ( Actually, we can rely on the loop condition at the start, and inside we always print the remainder, but we must ensure we don't print 0. We can check if the remainder is 0 by using 0=, but we need to keep the remainder. Maybe we can use a different approach: we only enter the loop body if second != 0, and we compute rem. If rem == 0, we exit. How to check rem == 0 without losing rem? We can do: mod, then over 0= ? Not sure. )
```
```output
4967564154 

error: unterminated structure
fix the program and send the corrected block again
```
The output just shows the second number, then error.
~~~

The round-5 adapter (`s2a_r5`), same settings. Round 5 trained on demonstrations: a short plan, one compact program, the output, the answer. This is the habit:

~~~text
Euclid's algorithm on (2426035688, 4967564154): replace (a, b) by (b, a mod b) until b is 0. The trace starts with 4967564154, then every nonzero remainder, then the gcd (the last nonzero remainder, so it appears twice). Each step is a long division of big numbers: the machine prints the whole trace exactly. Word steps ( a b -- ): while b is nonzero, print b (dup .), then swap over mod turns a b into b (a mod b). When b is 0, drop it and print a, the gcd.</think>```forth
: steps ( a b -- ) begin dup while dup . swap over mod repeat drop . ;
2426035688 4967564154 steps
```
```output
4967564154 2426035688 115492778 687350 17978 4186 1234 484 266 218 48 26 22 4 2 2
```
The gcd is 2.

\boxed{4967564154 2426035688 115492778 687350 17978 4186 1234 484 266 218 48 26 22 4 2 2}
~~~

Result: correct, 378 tokens. The grader accepts the trace with or without the repeated gcd at the end.

## What changed, in numbers

Frozen test, 600 problems, 2 samples each. "Uses machine" is the share of rollouts with at least one block.

| Kind | Stock: correct | Stock: uses machine | `s2a_r1`: correct | `s2a_r1`: uses machine | `s2a_r5`: correct | `s2a_r5`: uses machine |
|---|---|---|---|---|---|---|
| hard | 74.5% | 55.2% | 83.2% | 77.1% | 86.1% | 71.1% |
| small | 80.0% | 24.3% | 87.9% | 33.6% | 91.4% | 35.7% |
| variant | 65.3% | 43.1% | 62.5% | 34.7% | 87.5% | 31.9% |
| easy | 87.6% | 23.7% | 93.4% | 8.4% | 95.5% | 7.1% |
| easy_big | 86.8% | 20.5% | 92.3% | 5.0% | 92.7% | 3.6% |
| all | 81.0% | | 87.4% | | 91.0% | |
| median tokens | 3384 | | 1147 | | 850 | |

The trained model uses the machine more on hard problems and less on easy ones. The cold start and the rounds set out to teach exactly that.

Settings: stock and `s2a_r1` ran with card v2 and the grader of 2026-11-05, so those two columns compare directly: +6.4 [+4.1, +8.8]. `s2a_r5` ran with card v4 and the grader of 2026-11-06. Its column compares directly only with `s2a_r3` under the same settings: +2.0 [+0.1, +4.0] ([training](../training/README.md#the-rounds)).
