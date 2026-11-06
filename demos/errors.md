# Errors, recovery and limits

An error stops the block, and the output shows two fixed lines: what went wrong and what to do ([the error list](../forth/README.md#errors)). The model reads them like any other output and sends a corrected block. These are real rollouts of the round-5 adapter on the frozen test (card v4, think 12000 / turn 24000).

## Which errors happen

Error lines in all 1200 frozen-test rollouts of `s2a_r5`:

| Error | Count |
|---|---|
| unknown word | 163 |
| stack underflow | 122 |
| code memory full | 76 |
| / dividend over 10 digits | 39 |
| to needs the name of a value or variable | 18 |
| * / MOD top operand over 10 digits | 18 |
| ; outside a definition | 16 |
| / or MOD by zero | 15 |
| strings and emit are not supported | 12 |
| over 128 prints in one block | 10 |
| a name collides with another name | 10 |
| the block ran too long | 8 |

## A missing declaration

`L6.compound`: "A population of 34565 grows by 5% per year (fractional individuals dropped each year). What is it after 50 years?" Answer: 396261.

~~~text
Let's write Forth:

```forth
34565 to population  (or we can just start with 34565 on stack and use variable)
50 0 do
    population 105 * 100 / to population
loop
population .
```
```output
error: to needs the name of a value or variable
fix the program and send the corrected block again
```
But we can just use variable population. We'll declare variable population, then assign initial value, then loop.

Better: 
```forth
variable pop
34565 to pop
50 0 do
    pop 105 * 100 / to pop
loop
pop .
```
```output
396261
```
That seems plausible. Let's verify: 50 iterations of 5% growth with dropping fractions. We'll trust the Forth output.

But let's double-check the arithmetic. Could also be rounding? The problem says "fractional individuals dropped each year", meaning we drop the fractional part, i.e., floor. Our Forth code does integer division truncating toward zero, which for positive numbers is floor. So correct.

Thus final answer after 50 years is 396261. We'll output \boxed{396261}.</think>\boxed{396261}
~~~

Correct, 807 tokens. The fix is one line: declare the variable. A bare variable name fetches its value, so `pop 105 *` works.

## A limit, read and obeyed

`DMbig.mul`: "What is 901918761221 times 8579391?" Answer: 7737913702750596411.

The top operand of `*` (pushed last) has at most 10 digits. The model pushes the 12-digit number last, reads the error, and swaps the order:

~~~text
We'll create a Forth block: 
```forth
8579391 901918761221 * . 
```
```output
error: * / MOD top operand over 10 digits
fix the program and send the corrected block again
```
Wait, the machine said error: * / MOD top operand over 10 digits. That means the top operand (which is 901918761221) has 12 digits > 10. So we need to swap the order: push the larger number first, then the smaller. So we should do: 901918761221 8579391 * . But then top is 8579391 (7 digits) which is fine. However, the other operand (901918761221) has 12 digits which is <=20. So that should work. Let's try: 901918761221 8579391 * . 

But wait: The description says: "top operand (pushed last) may have up to 10 digits, the other up to 20". So if we push 901918761221 first, then 8579391, the top is 8579391 (7 digits), the other is 901918761221 (12 digits). That's allowed. So we should do 901918761221 8579391 * . 

Let's test that.</think>```forth
901918761221 8579391 * .
```
```output
7737913702750596411
```
The product is 7737913702750596411. 

\boxed{7737913702750596411}
~~~

Correct, 1086 tokens.

## Code memory

Code memory holds 192 cells per turn, and a definition sent again takes new cells ([why](../forth/README.md#code-memory)). So the habit that fills memory is re-sending code that already worked. The share of machine rollouts that hit "code memory full" on the frozen test (think 12000 / turn 24000):

| Adapter | Code memory full | Note |
|---|---|---|
| `s2a_r1` (card v2) | 10.6% | correct in 36% of those turns, 83% otherwise |
| `s2a_r1` (card v3) | 12.6% | |
| `s2a_r3` (card v3) | 7.6% | rows with redundant re-runs ranked last |

The kernel change that frees top-level code shipped after these runs. Replayed on real turns, it avoided 45% of the fulls and lost no answers. On the code-heavy dev set, `s2a_r5` hit "code memory full" in 11 of 400 rollouts.

## The step cap and the stack cap

The machine stops a block after 196608 kernel steps, so an endless loop ends with an error instead of a hung model. A loop that pushes a value each pass ends at 1024 values. Both errors say what to check ([examples](../forth/README.md#errors)).
