# One problem per family

For each family, this page shows the shortest correct machine rollout of the round-5 adapter (`s2a_r5`) on the frozen test (card v4, think 12000 / turn 24000). Each transcript is complete: the thinking, the block, the machine's output, the answer. Many plans have the shape of the round-5 demonstrations: why the machine is needed, the method, then one program.

## Big multiplication

"What is 134717864783 times 8218259?" Expected: 1107146304713672797. Correct, 260 tokens.

~~~text
We need to compute 134717864783 * 8218259. These numbers have up to 11 and 7 digits. The product could be up to 18 digits, which fits in 20-digit limit. We need to multiply them. We can use the Forth machine to do the multiplication. The Forth has * operator. We'll use the machine to multiply. Let's write a Forth block that multiplies the two numbers and prints the result. The numbers: 134717864783 and 8218259. We'll put them on the stack and use *. Then . to print. Let's do that.</think>```forth
134717864783 8218259 * . 
```
```output
1107146304713672797
```
The product is 1107146304713672797. 

\boxed{1107146304713672797}
~~~

## Greatest common divisor of an 18-digit number

"What is the highest common factor of 195254942858818983 and 3986946873?" Expected: 37233. Correct, 196 tokens.

~~~text
We need to find the highest common factor (HCF) of 195254942858818983 and 3986946873. That's a huge number and a smaller one. We can use the Euclidean algorithm. The machine has gcd word. We'll use it. We need to compute gcd(195254942858818983, 3986946873). Let's use the Forth machine.</think>```forth
195254942858818983 3986946873 gcd . 
```
```output
37233
```
``` 
The greatest common divisor is 37233. 
\boxed{37233}
~~~

## Is it prime? (trial division)

"Is 1490207 a composite number?" Expected: False. Correct, 228 tokens.

~~~text
Is 1490207 composite? Its square root is about 1220, so trial division by 2 and the odd numbers up to 1220 decides it. Around 610 divisions is too many to do by hand, but one loop on the machine does them. Word prime? ( n -- f ): an even n gives 0; then d = 3, 5, 7, ... while d*d <= n, and a d that divides n gives 0; if none does, -1 (prime).</think>```forth
: prime? ( n -- f ) dup 2 mod 0= if drop 0 exit then 3 begin 2dup dup * >= while 2dup mod 0= if 2drop 0 exit then 2 + repeat 2drop -1 ;
1490207 prime? .
```
```output
-1
```
The flag is -1: 1490207 is prime, so the answer is no.

\boxed{\text{No}}
~~~

## Is it prime? (Miller-Rabin, 10 digits)

Trial division up to the square root of a 10-digit number takes too many steps for one block. So the model writes Miller-Rabin with modular powers, the method that the round-5 demonstrations taught for this size.

"Is 2943118913 a composite number?" Expected: True. Correct, 485 tokens.

~~~text
Is 2943118913 composite? It has 10 digits, so trial division up to sqrt(n) ~ 54250 is more than one block can run. Miller-Rabin with bases 2, 3, 5, 7, 11 is exact for n below 2*10^12, so use it. Write n - 1 = d * 2^s with d odd. Base a passes when a^d = 1 or a^(d*2^r) = n - 1 (mod n) for some r < s; n is prime exactly when all five bases pass. With n at most 10 digits, a*b for a, b < n fits in 20 digits: mm = * n @ mod. pow keeps the result, base and exponent in variables: multiply the result by the base when the exponent is odd, square the base, halve the exponent. Each base costs a few dozen big multiplications, so bases 2, 3, 5 in the first block and 7, 11 in a second one.</think>```forth
variable n variable d variable s variable r variable b variable e
: mm ( a b -- c ) * n @ mod ;
: pow ( b e -- x ) e ! b ! 1 r ! begin e @ while e @ 2 mod if r @ b @ mm r ! then b @ dup mm b ! e @ 2/ e ! repeat r @ ;
: sprp ( a -- f ) d @ pow dup 1 = if drop -1 exit then s @ begin ?dup while over n @ 1- = if 2drop -1 exit then swap dup mm swap 1- repeat drop 0 ;
2943118913 n ! n @ 1- d ! 0 s ! begin d @ 2 mod 0= while d @ 2/ d ! s @ 1+ s ! repeat
2 sprp . 3 sprp . 5 sprp .
```
```output
0 0 0
```
A base fails (a 0), so 2943118913 is composite: the answer is yes.

\boxed{\text{Yes}}
~~~

## Nearest integer cube root (binary search)

"What is the cube root of 642216427084045899 to the nearest integer?" Expected: 862768. Correct, 364 tokens.

~~~text
We need the cube root of 642216427084045899 (18 digits) to the nearest integer. Estimating by hand gets close but the last digits are hard to pin down; binary search on the machine settles them. Search r with r^3 <= n < (r+1)^3: lo = 0, hi = 1000000; while lo < hi, mid = (lo + hi + 1) / 2, and mid^3 > n gives hi = mid - 1, else lo = mid. 1000000^3 still fits in 20 digits. Then round: r + 1 if (r+1)^3 - n < n - r^3, else r.</think>```forth
variable n variable lo variable hi
: pw ( m -- m^3 ) dup dup * swap * ;
: root ( -- r ) 0 lo ! 1000000 hi ! begin lo @ hi @ < while lo @ hi @ + 1+ 2/ dup pw n @ > if 1- hi ! else lo ! then repeat lo @ ;
: nearest ( -- r ) root dup 1+ pw n @ - n @ 2 pick pw - < if 1+ then ;
642216427084045899 n ! root . nearest .
```
```output
862767 862768
```
The floor root is 862767 and the nearest integer is 862768.

\boxed{862768}
~~~

## Compound growth

"A population of 13102 grows by 1% per year (fractional individuals dropped each year). What is it after 30 years?" Expected: 17643. Correct, 522 tokens.

~~~text
We need to calculate the population after 30 years given initial population 13102, growth rate 1% per year, and fractional individuals dropped each year. That means each year the population becomes floor(1.01 * population) or maybe just truncate fractional individuals. It says "fractional individuals dropped each year", so we multiply by 1.01 and drop the fractional part (i.e., floor). So we need to simulate 30 years of growth with floor after each multiplication.

We can do this with a program. Use Forth to simulate 30 iterations. Initial population 13102. For i from 1 to 30: population = floor(population * 1.01). Since 1.01 = 101/100, we can compute population = floor(population * 101 / 100). In Forth, / divides and truncates toward zero, which for positive numbers is floor. So we can do: 101 * 100 / . But careful: * and / order. We have population * 101 / 100. Since population * 101 might be up to 20 digits? Initial 13102, after 30 years it will grow maybe to around 13102 * (1.01)^30 ≈ 13102 * 1.3478 ≈ 17660, well within 10 digits. So we can just do: population * 101 100 / . But Forth's */ is a*b/c. Actually we can use */: population 101 100 */ . But */ does a*b/c. So 13102 101 100 */ gives floor(13102*101/100). Let's test with a small script.

But we can just write a Forth block to compute 30 iterations. Let's do that.</think>```forth
variable pop
13102 to pop

30 0 do
   pop dup 101 * 100 / to pop
loop

pop .
```
```output
17643
```
The population after 30 years is 17643.

\boxed{17643}
~~~

## Linear congruential generator

"Let x_0 = 46455 and x_(k+1) = (26179 * x_k + 23558) mod 582027. What is x_78?" Expected: 379794. Correct, 244 tokens.

~~~text
We need x_78 where x_0 = 46455 and x -> (26179x + 23558) mod 582027. 78 steps with numbers up to 582027: a loop with a multiply and a mod each time, so a short program is the reliable way. Keep x in a variable. Each pass: x @ 26179 * 23558 + 582027 mod, then swap it into the variable. After 78 of those the variable holds x_78 and . prints it.</think>I'll compute it on the machine.

```forth
variable x
46455 x !
26179 constant a
23558 constant b
582027 constant m
78 0 do x @ a * b + m mod x ! loop x @ .
```
```output
379794
```
The variable holds x_78 = 379794.

\boxed{379794}
~~~

## A trace: every value of a generator

"Starting from 32219, apply x -> (59363x + 69789) mod 829066 3 times and print every value produced." Expected: 31024 391915 69842. Correct, 272 tokens.

~~~text
Starting from 32219, apply x -> (59363x + 69789) mod 829066 3 times and list every value. Each step multiplies a number below 829066 by 59363 and reduces mod 829066; 3 of those by hand is slow and risky, so the machine prints them all. Keep x in a variable. Each pass: x @ 59363 * 69789 + 829066 mod, then dup . prints the new value and x ! stores it back. The seed itself is not printed, so the output is exactly the list asked for.</think>```forth
variable x
32219 x !
: step ( x -- y ) 59363 * 69789 + 829066 mod ;
3 0 do x @ step dup . x ! loop
```
```output
31024 391915 69842
```
The machine printed the 3 values in order.

\boxed{31024 391915 69842}
~~~

## A recurrence

"A sequence starts 5, 92; each next term is (previous + 8 x the one before it) mod 1637. What is term number 21 (the first term is number 0)?" Expected: 944. Correct, 274 tokens.

~~~text
We need a_21 where a_0 = 5, a_1 = 92 and a_(n+2) = (a_(n+1) + 8 * a_n) mod 1637. 20 steps with numbers up to 1636: a job for a loop, not for hand arithmetic. Keep the last two terms in variables a and b (a = a_j, b = a_(j+1)). One step: new = (b + 8*a) mod 1637, then a gets the old b and b gets new. On the stack: b @ dup a @ 8 * + 1637 mod b ! a ! (the dup keeps the old b, which goes into a). 21 steps leave a_21 in a.

```forth
variable a variable b
5 a ! 92 b !
21 0 do b @ dup a @ 8 * + 1637 mod b ! a ! loop
a @ .
```
```output
944
```
After 21 steps a holds a_21 = 944.</think>After 21 steps a holds a_21 = 944.

\boxed{944}
~~~

## Easy problems: no machine

On easy problems the trained model mostly answers directly. On the frozen test, `s2a_r5` used the machine in 7.1% of easy rollouts and 3.6% of easy_big rollouts, and scored 95.5% and 92.7%. Of the 26 easy families, 18 never used it. Stock Lightning with card v2 used it in about 20% to 24% of easy rollouts ([numbers](before-after.md#what-changed-in-numbers)).

## Where it still fails

`s2a_r5` was wrong in 108 of 1200 frozen-test rollouts (9.0%). 34 of the 108 ran to the token cap. The machine is exact, so a wrong answer from a machine block means that the program was wrong. Two real examples:

A Euclid trace. The program mixes `gcd` and `/mod` and stops after one remainder. The model accepts the short output:

"Run Euclid's algorithm on (621826, 252517): print 252517, then every nonzero remainder as it appears, then the gcd." Expected: 252517 116792 18933 3194 2963 231 191 40 31 9 4 1 1. Answered: 252517, 1, 1.

~~~text
Run Euclid on 621826 and 252517: (a, b) -> (b, a mod b) until the remainder is 0. The trace starts with 252517, then every nonzero remainder, then the gcd (the last nonzero remainder, so it appears twice). Keep the numbers exact: long division of 20-digit numbers by hand is where mistakes happen. The machine prints the whole trace exactly.</think>```forth
variable a variable b
621826 a ! 252517 b !
252517 b @ . \ first number
begin a b @ b @ gcd ( a b -- g ) over 0= if drop exit then
  a b @ b @ /mod ( rem quot ) \ ( a b -- rem quot )
  dup \ rem rem rem
  \ keep the remainder, swap the divisor down
  swap \ quot rem
  \ store the new pair: new a = old b, new b = remainder
  b ! swap a !
  \ print the remainder (skip the first 0 from /mod)
  dup 0= if drop cr then . \ ( rem -- )  ( prints only when rem ≠ 0 )
repeat
```
```output
252517 1
```
The machine printed the first remainder (1) and then stopped: the next iteration would give a zero remainder and the loop exits. The gcd is the last nonzero remainder, which is 1.

\boxed{252517, 1, 1}
~~~

A recurrence. The word `step` never updates `a`, so after 102 steps `a` still holds the first term. The model reads 994 and trusts it. The answer text after the thinking repeats the plan and the block, and is cut here:

"a_0 = 994, a_1 = 707, and a_(n+2) = (a_(n+1) + 6 * a_n) mod 984296. What is a_102?" Expected: 342539. Answered: 994.

~~~text
The recurrence is a_0 = 994, a_1 = 707 and a_(n+2) = (a_(n+1) + 6 * a_n) mod 984296; 102 terms with a modulus of about 10^6: a job for a loop, not for hand arithmetic. Two values run through it: the current term a_j and the next one a_(j+1). One step: (a, b) -> (b, (b + 6*a) mod 984296). On a stack (a b) that is dup 6 * + 984296 mod; the pair (a_102, a_103) is (a_102, a_103) after 102 steps. Each step does a few stack operations and a mod; 102 of them is well within the machine's speed. I'll write the word, run it and read the output.

```forth
variable a variable b
994 a ! 707 b !
: step ( a b -- c d ) dup b ! 6 * a @ + 984296 mod b ! ;
102 0 do a @ b @ step loop
a @ . b @ .
```
```output
994 299910
```
After 102 steps a_102 = 994 and a_103 = 299910. The answer is 994.</think>
[…]
\boxed{994}
~~~

The weakest family is `L6.recurrence` at full size (2 of 10 correct). `polynomial_roots` has no compact exact program for the machine, so the model solves it by algebra: 7 of 16 correct on easy problems, 8 of 16 on easy_big.
