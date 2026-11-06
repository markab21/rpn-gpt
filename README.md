# rpn-gpt

An exact Forth machine, compiled by hand into the weights of a language model, and a model trained to use it.

The model is NVIDIA Nemotron 3.5 Lightning (30B parameters, 3B active). Beside it, in the same network, sits a small stack machine (the kernel) whose weights we computed, not trained. The model writes ordinary Forth in a ` ```forth ` block in its answer. When it closes the block, the kernel writes the ` ```output ` block, token by token, inside the same forward passes. There is no tool call and no host program. Then we trained the model (LoRA SFT rounds) to use the machine when a problem needs it, and served the result in vLLM.

## Contents

- [The question](#the-question)
- [A 30-second example](#a-30-second-example)
- [Results](#results)
- [How it works](#how-it-works)
- [The story](#the-story)
- [What is next](#what-is-next)
- [Repository](#repository)
- [Credits and license](#credits-and-license)

## The question

Language models are bad at long exact arithmetic. A model that must multiply two 12-digit numbers, or run a recurrence for 100 steps, usually computes by hand in its thinking, slowly and with errors. The usual fix is a tool call: the model writes Python, and a host runs it.

We asked another question. Can the model's own weights hold an exact machine that the model drives by writing text? And if so, can we teach the model to use it well?

Forth fits this. It is a stack language with reverse Polish notation (RPN): `2 3 +` means 2 + 3. A word takes its inputs from the stack and leaves its outputs there. A stack machine needs no parser for expressions and no registers, so it can be built from the layers that the model already has.

## A 30-second example

A real turn from the in-weights evaluation ([full context](demos/in-the-weights.md#the-shortest-turn)). Problem: "Calculate the remainder when 82225043186179281 is divided by 6469193." The model's thinking ends with `</think>`, then it writes a block:

```forth
82225043186179281 6469193 mod .
```
```output
1290731
```

The output block came from the kernel inside the model. In between, the kernel ran 297 hidden tokens (special tokens that a reader never sees), and every one matched the spec. Then the model answered `\boxed{1290731}`, which is correct.

You can run the same program through the spec on a laptop:

```sh
uv sync
uv run python -m forth.run '82225043186179281 6469193 mod .'
uv run python kernel/show_stream.py '2 3 + .'     # the hidden tokens of one block
```

## Results

| Claim | Evidence |
|---|---|
| The kernel in the weights is exact | 618 test cases, 765,734 kernel-owned positions, 0 wrong, in prefill and in cached decode. One more 171k-token block, exact. [Details](kernel/README.md#parity-testing). |
| The in-weights machine gives the same results as the external one | Round-3 adapter, 407 dev problems: 89.9% in the weights (vLLM composite) vs 89.3% with the external machine. 395 of 407 turns exact at every kernel position.<sup>1</sup> |
| Training makes the model use the machine well | Stock Lightning 81.0% vs round-1 adapter 87.4% on the frozen test: +6.4 [+4.1, +8.8].<sup>2</sup> |
| Program demonstrations teach a new habit | Round-5 adapter vs round-3 adapter: +2.0 [+0.1, +4.0] on the frozen test, +11.0 [+5.5, +16.5] on the code-heavy dev set.<sup>3</sup> |
| Plain math does not get worse | MATH-500 with no machine: stock 97.2%, round-3 adapter 97.7% (+0.5 [-0.4, +1.4]). Round 5 vs stock: +0.0 [-1.1, +1.1].<sup>4</sup> |
| The composite runs batched in vLLM | 10 hard problems, 64 streams at once: 50 s per problem, against 457 s in the HF composite. [Details](kernel/README.md#serving-in-vllm). |

A delta is in points, paired by problem, with a 95% bootstrap interval over problems. Both arms of every comparison run on the same problems, engines, card (the system prompt that describes the machine), thinking budget and grader:

1. Dev set (407 problems), card v3, think 12000 / turn 24000. External machine: 2 samples per problem. In the weights: 1 sample.
2. Frozen test (600 problems, 2 samples each), card v2, think 12000 / turn 24000, grader of 2026-11-05.
3. Frozen test x2 and the 200-problem code-heavy dev set x2, card v4, grader of 2026-11-06, adapters on the same LoRA engines.
4. MATH-500 x2, no machine, the same engine path for both arms.

Stock Lightning against the round-5 adapter under one setting is not measured yet. The full ledger of rounds is in [training](training/README.md#the-rounds).

## How it works

- The kernel is a second, small NemotronH stack (27 layers: Mamba-2 mixers, MLPs and one idle attention layer) with every weight computed by a Python builder. Numbers live as rows of Mamba-2 state, 20 decimal digits each. bf16 stays exact because no feature goes above 256. [The kernel](kernel/README.md).
- The composite is one model: stock Lightning, byte-identical, and the kernel as a submodule. Where the kernel emits, its logits win. The base reads only the visible text, so a LoRA trained on plain-text rows runs on it unchanged.
- The spec, [`kernel/text_vm.py`](kernel/text_vm.py), is a Python machine that gives the kernel's exact token stream. Training rows, rollouts and the kernel all follow it. If the kernel cannot do a feature exactly, the feature leaves the spec. [The dialect](forth/README.md).
- Training is LoRA SFT with NeMo AutoModel: a 60-row cold start, then rounds of the model's own clean correct rollouts, then program demonstrations. [Training](training/README.md).
- Serving is a vLLM out-of-tree model: an NVFP4 or BF16 base with a LoRA, the kernel in bf16. [Playbooks](playbooks/README.md).

## The story

1. The question: can a model's weights hold an exact machine? A first version on Qwen3.5-0.8B was exact, but the small model could not drive it ([history](kernel/history.md)).
2. The machine: a stack machine built for Mamba-2 layers, read from plain Forth text, and placed beside stock Lightning as a composite ([kernel](kernel/README.md)).
3. Making it exact: a spec, parity tests at every position, and features dropped when the kernel could not match them. The tests found real bugs, for example a counter that drifted after 80,000 kernel tokens ([bugs](kernel/README.md#bugs-that-the-tests-found)).
4. Teaching the model: on easy problems the machine hurts (88.7% without it, 75.6% with the full card), on hard ones it helps (28.3% to 73.1%). So the target was when to use it, not only how ([training](training/README.md#before-any-training)).
5. Results: a cold start, clean on-policy rows and program demonstrations each gave a measured gain ([before and after](demos/before-after.md), [in the weights](demos/in-the-weights.md)).
6. What did not work: context distillation (the model quotes its prompt), GRPO (+0.1 against a correct control), more rows of the same kind, and grader false negatives that looked like skill gaps ([what did not work](training/README.md#what-did-not-work)).

## What is next

- Bolt (74B, 9B active) and Super (120B, 12B active), the bigger models of the same family. They share the hybrid Mamba-2 / attention / MoE design and the vocabulary size, so the kernel can go beside them after a check of its token ids.
- The weak families: recurrences at full size, and polynomial roots, which have no compact program for this machine.
- A cheaper kernel: one forward pass per kernel step is the main cost (a `/` on 6-digit numbers takes about 115 hidden tokens).

## Repository

| Folder | What it holds |
|---|---|
| [`kernel/`](kernel/) | How the machine lives in the layers, the decisions and bugs; the spec, builder, parity tests, the composite and the vLLM plugin |
| [`forth/`](forth/) | The dialect: words, limits, error lines, what we dropped; a command-line runner |
| [`training/`](training/) | Problem sets, held-out sets, the rounds with their numbers; rollout, row, grading and GRPO tools |
| [`playbooks/`](playbooks/) | How to do each step on a GPU machine: Ansible playbooks with hosts and paths as variables |
| [`demos/`](demos/) | Real transcripts: before and after, in the weights, errors, one per family |
| [`tests/`](tests/) | CPU tests: the spec self-check, the grader, and every Forth example in the docs |

The CPU parts run with `uv sync` and `uv run --group dev pytest -q`. The GPU parts were run on machines with 8 RTX PRO 6000 GPUs. They were not re-run for this repository.

The training rows, rollouts and adapters live in a private Hugging Face dataset and model repo. They are not published with this repository. The held-out evaluation sets are in [`training/data/`](training/data/).

## Credits and license

The idea of compiling a machine into a language model's weights comes from [qwenputer](https://github.com/divagr18/qwenputer) by Divyansh Agrawal. Our first, Qwen-based phase reused its MIT-licensed weight compiler and carry-lookahead construction. The Nemotron kernel here is a new construction, but the [LICENSE](LICENSE) keeps the upstream copyright notice.

Nemotron 3.5 Lightning is NVIDIA's model, under its own license. This repository holds no model weights.
