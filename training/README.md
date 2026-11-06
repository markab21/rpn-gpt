# Teaching the model to use the machine

The kernel is exact from the start, but stock Lightning does not know when or how to use it. This page tells how we taught it: the problem sets, the training rounds, what worked, and what did not. Every number comes from the project ledger.

## Contents

- [Problems and held-out sets](#problems-and-held-out-sets)
- [How we measure](#how-we-measure)
- [Before any training](#before-any-training)
- [The rounds](#the-rounds)
- [What worked](#what-worked)
- [What did not work](#what-did-not-work)
- [The recipe](#the-recipe)
- [Rules we learned](#rules-we-learned)
- [Files](#files)

## Problems and held-out sets

Every problem is generated with a seed and has an exact answer. Problems that need a program are run on the spec when they are made. If the reference program fails (for example, it goes past the machine's limits), the problem is drawn again.

| Kind | What it is | Families |
|---|---|---|
| hard | Problems that need many exact steps: long simulations, exact traces, counting, searches, and DeepMind math scaled past what a model can do in its head | 14: `L6.lcg`, `L6.lcg_trace`, `L6.recurrence`, `L6.euclid_trace`, `L6.compound`, `L6.crt_search`, `L6.count_digitsum`, `L6.count_quadres`, `DMbig.mul`, `DMbig.div_remainder`, `DMbig.gcd`, `DMbig.lcm`, `DMbig.is_prime`, `DMbig.nearest_integer_root` |
| small | The `L6` families at 5 to 40 steps | 8 |
| variant | Easier forms of the families that no model solved at first | 4 |
| easy | DeepMind `mathematics_dataset` at training difficulty | 26 |
| easy_big | DeepMind's extrapolation modules (bigger numbers, longer lists) | 15 |

Two examples from the frozen test:

- `L6.recurrence`: "A sequence starts 479, 195; each next term is (previous + 5 x the one before it) mod 40438. What is term number 139 (the first term is number 0)?" Answer: 22969.
- `DMbig.div_remainder`: "Calculate the remainder when 926084932167062084 is divided by 77260069." Answer: 64287736.

The easy kinds are there on purpose. On easy problems the machine does not help, and the model must learn not to use it.

Held-out sets are never trained on. Training sets are drawn with new seeds and checked to be disjoint from all of them. The held-out sets are in [`data/`](data/):

| Set | Size | Use |
|---|---|---|
| [`frozen_test.json`](data/frozen_test.json) | 600 (194 hard, 70 small, 36 variant, 190 easy, 110 easy_big) | Carved out first. Used only at milestones, never to pick a round. |
| [`dev_round.json`](data/dev_round.json) | 407, stratified like the frozen test | The decision for each round |
| [`dev_codefull.json`](data/dev_codefull.json) | 200 hard, code-heavy | Code-memory habits |
| [`demo_set.json`](data/demo_set.json) | 90 | Hard problems where stock Lightning gets 21% alone and 47% with a one-line machine prompt |
| [`dm_frontier.json`](data/dm_frontier.json) | 260 (10 per family) | Easy problems |
| [`math500.json`](data/math500.json) | 500 | MATH-500, no machine: the guard against a plain-math regression |
| [`demo10.json`](data/demo10.json) | 10 | Smoke tests in the weights |

The prompts are in [`data/cards/`](data/cards/). The card is the system prompt that tells the model about the machine. Its current version is [`reference_v4.txt`](data/cards/reference_v4.txt).

## How we measure

A rollout is one full model turn on one problem, with the machine in the loop. [`loop_sim.py`](loop_sim.py) runs rollouts against a vLLM server. When the model closes a block, the spec computes the output block and the loop appends it. This is the external machine. It gives the same tokens as the kernel in the weights, and it runs on a plain vLLM server.

The grader is exact: the last `\boxed{}` answer, normalized, must equal the expected answer (`loop_sim.same`). MATH-500 is graded with `math_verify` instead ([`math_grade.py`](math_grade.py)).

Every comparison is paired. Both arms run on the same problems, on the same engines, with the same card, the same thinking budget, and the same grader. [`paired_delta.py`](paired_delta.py) computes the mean difference per problem and a 95% bootstrap interval over problems. We write it as `+2.0 [+0.1, +4.0]`: the difference in points, then the interval.

Samples per problem (x2, x4) set the resolution. On the frozen test, x2 resolves about 2.5 points and x4 about 1.5 points. Smaller effects are noise.

A control must use the same code path. An adapter arm is compared with an all-zero adapter on the same LoRA engine ([`zero_adapter.py`](zero_adapter.py)), not with the base model name. This rule came from a mistake, described [below](#grpo).

## Before any training

Stock Lightning with the machine, by card (the system prompt):

| Set | No machine | One-line prompt | Full card |
|---|---|---|---|
| Easy (260 DeepMind problems, x2) | 88.7% | 78.5% | 75.6% |
| Hard (demo set, 90 problems, x4) | 28.3% | 42.8% | 73.1% |

On easy problems the machine hurts: the model computes by hand well, and the machine costs tokens and capping. On hard problems it more than doubles the score. So the target was not "use the machine". The target was "use it when the problem needs it".

## The rounds

An adapter is a LoRA trained on stock Lightning. "s2a" means stage 2a, expert iteration: the model's own correct rollouts become its next training rows.

| Adapter | Training data | Result | Verdict |
|---|---|---|---|
| `s1_cold` | 60 clean rows: 30 hard (machine), 30 easy (direct answers) | Hard demo set x4 50.3% to 65.0%. Easy x2 81.7% to 85.8%. | The cold start works |
| `s2a_r1` | 562 rows from its own correct rollouts | Frozen test x2 vs stock: **+6.4 [+4.1, +8.8]** (81.0% to 87.4%) | Promoted |
| `s2a_r2` | 1025 rows, same kind | Demo set x4: 67.8% vs 70.6% for `s2a_r1` | Plateau, not promoted |
| GRPO `r1b` | 20 steps of 8 x 8 rollouts | Frozen x4 vs a zero-adapter control: +0.1 [-1.2, +1.4] | Bought nothing, dropped |
| `s2a_r3` | 8214 clean on-policy rows, weak families weighted up, few re-runs | Dev x2 vs `s2a_r1`: +5.0 [+2.7, +7.5]. Frozen x2: +1.4 [-0.2, +3.1], hard +5.2 [+1.3, +9.0] | Promoted |
| `s2a_r4` | + 342 weak-cell rows at a 12k budget, a MATH guard | Dev x4 vs `s2a_r3`: +0.5 [-1.3, +2.2] | Tie, not promoted |
| `s2a_r5` | + 1177 program demonstrations (16% of rows) | Dev x4: +1.4 [-0.3, +3.0]. Code-heavy dev x2: **+11.0 [+5.5, +16.5]**. Frozen x2 vs `s2a_r3`: **+2.0 [+0.1, +4.0]** (89.0% to 91.0%) | Promoted, current best |
| `s2a_r6` | + 550 more demonstrations (stack forms, recovery) | Dev x4 vs `s2a_r5`: -0.1 [-1.7, +1.4] | Tie, not promoted |

Settings for the frozen-test lines: `s2a_r1` vs stock used card v2, think 12000 / turn 24000, grader of 2026-11-05 (commit `6e5947e`). `s2a_r3` vs `s2a_r1` used card v3, regraded with the euclid / roots fix. `s2a_r5` vs `s2a_r3` used card v4, grader of 2026-11-06. Each line compares two arms under one setting. Stock Lightning against `s2a_r5` under one setting is not measured yet.

The plain-math guard held at every promotion: on MATH-500 x2 with no machine, stock scored 97.2% and `s2a_r3` 97.7% (+0.5 [-0.4, +1.4]). `s2a_r5` vs stock was +0.0 [-1.1, +1.1]. Neither adapter wrote Forth on MATH-500.

## What worked

### A small cold start

Sixty clean rows and 8 LoRA steps took the hard demo set from 50.3% to 65.0% (x4) and the easy set from 81.7% to 85.8%. The rows teach the format and two behaviors: the machine on hard problems, a direct answer on easy ones.

The rows came from rollouts with the full card, rebuilt under a short prompt with no policy lines. A row was kept only if it was clean: no copied card sentence (8-word runs), one `</think>`, no identical re-run, no `...` placeholder block.

### Round 3: clean on-policy rows

Round 3 is the first large gain after round 1: dev +5.0 [+2.7, +7.5], hard frozen problems +5.2 [+1.3, +9.0]. The rows came from mass sampling: 21,060 fresh problems x4, then the informative ones x16. Four changes made the difference:

1. Rows are exact against the composite's token stream, built segment by segment with the kernel's own token ids.
2. A row with a redundant re-run (a block at least 80% like an earlier one that worked) ranks last. Re-runs fill code memory. Redundant re-runs fell from 1.75 to 1.20 per machine rollout.
3. The weak families got more rows.
4. Rows with a thought force-closed at the budget were dropped. They end mid-sentence and teach stray `</think>` tokens.

Median turn length fell from 1257 to 929 tokens. "Code memory full" fell from 12.6% to 7.6% of machine rollouts.

### Round 5: program demonstrations

After round 3, the weak cells were `recurrence`, `is_prime`, `nearest_integer_root`, `lcg_trace`, and `euclid_trace`. We read the failures. The model computed by hand, or fought the stack (`to` on a variable, a missing `dup`, `x` with no `@`), then ran out of thinking. More on-policy rows did not help (round 4), because the correct rollouts in those cells were mostly by-hand traces.

So [`r5_demos.py`](r5_demos.py) writes demonstrations: a short plan, one compact working program, the machine's exact output, and the answer read from it. Each problem is fresh. Its expected answer comes from the generator's own reference program, and the demo program is a different program, so agreement is a real check. Every row is graded, exact against the composite stream, and clean.

The gains were in the targets: recurrence hard 12% to 50%, `numbers.is_prime` 40% to 69%, `euclid_trace` near 50% to near 75% (dev x4). On the code-heavy set: +11.0 [+5.5, +16.5]. On the frozen test: +2.0 [+0.1, +4.0], with `euclid_trace` +28.6 and `is_prime` +15.9 to +21.4 points. Median tokens fell from 966 to 850.

A demonstration teaches a habit that the model's own rollouts do not yet show.

### Fixing the card and the budget

Two of the biggest gains were not training at all.

The card bug: after two rounds and a learning-rate sweep had "plateaued", we read the 106 wrong hard rollouts. The biggest bucket was capped big-number division. The card said that `* / mod` take operands of up to 10 digits, so the model split 17-digit numbers by hand. In fact `mod` takes a 20-digit dividend. With the corrected card (v2), these problems took one block.

The thinking budget: the model's thinking is closed at a budget of visible tokens. On the demo set x4, think 4000 gave 70.3%, 8000 gave 76.7%, 12000 gave 79.4%, 24000 gave 81.6%, and no limit gave 77.8%. The budget cost about 10 points, more than any training round. The default became think 12000, turn 24000.

## What did not work

### Context distillation

The plan was to roll out with the full card, then train on rows rebuilt under a short prompt, so the model learns the card's behavior without the card. Lightning quotes its system prompt ("the instructions say: ..."). Every card sentence appeared verbatim in 35% to 55% of machine rollouts. A "never quote" line did not help (20 of 20 still quoted). Rows rebuilt without the card teach the model to cite text that is not there. Filtering left only 10% to 15% of good rollouts, and editing the quotes out makes the rows off-policy. We replaced it with the small clean cold start.

### GRPO

GRPO (group relative policy optimization) is a form of reinforcement learning: sample a group of rollouts per prompt, reward the correct ones, push the policy toward them. We built it with NeMo RL, with the machine inside the rollouts ([`grpo_forth.py`](grpo_forth.py), [`configs/grpo_forth_r1.yaml`](configs/grpo_forth_r1.yaml)). It worked end to end: 2 train GPUs and 2 vLLM GPUs, LoRA refit, sync and async generation. The reward was correct minus a small length cost (1e-5 per token, because 1e-4 rivaled correctness).

After 20 steps, the adapter scored +2.0 [+0.6, +3.5] over the base model name on the same engines. That looked like a gain. But the base-name control came in low twice, at 83.3% and 83.1%, while the same weights on a plain engine scored 85.4%. So we built an exact control: an adapter with every `lora_B` set to zero, which runs the adapter code path with no change to the model. Against it, the GRPO adapter scored +0.1 [-1.2, +1.4], 71 problems better and 71 worse. The earlier gain was the engine path. A later check showed that `--enable-lora` does not change numerics. The low base-name readings were run-to-run variance.

At about 38 minutes per 64 x 8 step on 4 GPUs, a measurable RL gain needed far more hardware than we had. SFT rounds gave more per GPU-hour, so we dropped RL. The pipeline stays in this folder.

### More rows of the same kind

Round 2 (1025 rows, the same kind as round 1) tied round 1. 63% of the correct rollouts were filtered out. A recipe sweep (learning rate 1e-4 to 5e-4, rank 8 and 16, 1 and 2 epochs) showed that SFT on those rows was saturated: the best validation loss was 0.1817 to 0.1832 in every setting below 5e-4.

Round 4 added rollouts of the weak cells at a 12k budget. Only 342 of 1056 correct rollouts were accepted, and about 230 of the correct ones were right only after a forced close of the thinking. The rows taught by-hand traces more than programs. Dev x4: +0.5 [-1.3, +2.2].

Round 6 added a second batch of demonstrations of the same kind as round 5. Dev x4: -0.1 [-1.7, +1.4]. Demonstration rounds look saturated for these failures.

### Grader false negatives that looked like skill gaps

Some "weak families" were weak graders. Four grader fixes changed the picture:

| Fix | Example of an answer that was marked wrong | Effect |
|---|---|---|
| 1 | "Yes, 7432 is composite" for a yes / no question. Roots written as "x = a or x = b". | `polynomial_roots` 16.0% to 41.3% on 44k rollouts: a grader artifact, not a skill gap |
| 2 | "15500 micrograms" for a number with a unit | 8 false negatives on the frozen baseline |
| 3 | A Euclid trace with or without the leading number and the repeated gcd. Roots with "(double root)". | `euclid_trace` 8 to 21 of 42 correct on the frozen test |
| 4 | "51353 is composite" for "Is 51353 prime?" (No) | 6 and 3 flips on two dev arms, 0 flips the other way |

A paired delta is safe across a grader change only when both arms are regraded with the same grader. We regraded every arm. [`tests/test_grader.py`](../tests/test_grader.py) keeps these cases.

Also, `lcg_trace` and `euclid_trace` questions were ambiguous (with or without the seed, the first number). The grader accepts both readings.

## The recipe

[`configs/sft_text.yaml`](configs/sft_text.yaml), NeMo AutoModel:

| Setting | Value |
|---|---|
| Base | Lightning BF16 |
| LoRA | rank 16, alpha 32 |
| Optimizer | Adam, learning rate 1e-4 |
| Batch | global 16, local 1 |
| Sequence length | 16384 |
| Epochs | 1 (round 3 and later: validation loss rose in epoch 2) |
| Loss | only on what the model writes. The prompt and every kernel output block are masked. A program block whose output is an error is masked too. |

Each round trains a fresh LoRA on the previous rows plus the new rows. The previous adapter stays the fallback. The adapter is trained on BF16 Lightning and served on the NVFP4 base. Steps: [playbooks](../playbooks/README.md#train-a-lora-round).

## Rules we learned

- Read and bucket the failures before you train again. A five-minute read found a card bug that two rounds and a sweep did not fix.
- Small first: run 1 to 10 cases and read them by hand before a batch.
- Measure before you build. An A/B test on the existing pipeline takes minutes and decided the error format.
- Measure a no-machine baseline on every new problem set. Train the machine only where it wins.
- Never train on a held-out set. Never build rows from a prompt that the row does not contain.
- A spec change needs replay proof, and training data must be generated with the final spec.
- Teach a new habit with demonstrations, not with more on-policy rows.
- Compare adapters on the same code path, with a zero-adapter control.

## Files

| File | Use |
|---|---|
| [`loop_sim.py`](loop_sim.py) | Rollouts with the external machine (the spec) against a vLLM, mlx or llama.cpp server. Also the exact grader. |
| [`paired_delta.py`](paired_delta.py) | Paired A/B with a bootstrap interval |
| [`math_grade.py`](math_grade.py) | Regrade MATH-style answers with `math_verify` |
| [`mass_sets.py`](mass_sets.py) | The frozen test and the 22k-problem sampling pool |
| [`dev_round.py`](dev_round.py), [`stage1_sets.py`](stage1_sets.py), [`stage2a_sets.py`](stage2a_sets.py), [`r4_sets.py`](r4_sets.py), [`demo_gen.py`](demo_gen.py), [`dm_gen.py`](dm_gen.py), [`math_train_set.py`](math_train_set.py) | Problem sets, each disjoint from the held-out sets |
| [`mass_report.py`](mass_report.py), [`mass_select.py`](mass_select.py), [`loop_report.py`](loop_report.py) | Pass rates per family and kind, and selections from them |
| [`distill_rows.py`](distill_rows.py), [`r3_rows.py`](r3_rows.py) | Rows from rollouts: accept, clean, rebuild exact against the composite stream, mask |
| [`r5_demos.py`](r5_demos.py) | Program demonstrations |
| [`masked_chat.py`](masked_chat.py) | The AutoModel dataset class with the loss mask |
| [`zero_adapter.py`](zero_adapter.py), [`merge_lora.py`](merge_lora.py) | The zero-adapter control, and merging a LoRA into the base |
| [`logprob_gap.py`](logprob_gap.py), [`decode_gap.py`](decode_gap.py) | Engine checks: trainer vs sampler log-probs, two served names on one engine |
| [`grpo_forth.py`](grpo_forth.py), [`test_grpo_forth.py`](test_grpo_forth.py), [`sitecustomize.py`](sitecustomize.py), [`configs/grpo_forth_r*.yaml`](configs/) | The GRPO attempt (NeMo RL), kept for the record |
| [`rubric.py`](rubric.py) | Process scores from a judge model, used only to rank accepted rows |
| [`hf_put.py`](hf_put.py) | Upload rows and adapters to a Hugging Face repo in one commit |
| [`mlx_local.sh`](mlx_local.sh) | Lightning 8-bit on a Mac (MLX) for a few local rollouts |

The training rows, rollouts and adapters are in a private Hugging Face dataset and model repo. They are not part of this repository.
