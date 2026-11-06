# rpn-gpt

An exact Forth machine, compiled by hand into the weights of a language model, and a model trained to use it.

The model is NVIDIA Nemotron 3.5 Lightning (30B parameters, 3B active). In the same network sits a small stack machine, the kernel. We computed its weights. We did not train them. The model writes ordinary Forth in a ` ```forth ` block. When the model closes the block, the kernel writes the ` ```output ` block, token by token, in the same forward passes. There is no tool call and no host program. Then we trained LoRA adapters so that the model uses the machine when a problem needs it, and served the result in vLLM.

A real turn (round-1 adapter, kernel in the weights, [full context](demos/in-the-weights.md#the-shortest-turn)). The problem: "Calculate the remainder when 82225043186179281 is divided by 6469193." After its thinking, the model writes:

```forth
82225043186179281 6469193 mod .
```
```output
1290731
```

The kernel inside the model wrote the output block. Before it, the kernel ran 297 hidden tokens. A hidden token is a reserved special token that decodes to an empty string, so a reader never sees it. Every hidden token matched the spec, our Python reference machine. The model then answered `\boxed{1290731}`, which is correct.

Try the spec on a laptop:

```sh
uv sync
uv run python -m forth.run '82225043186179281 6469193 mod .'
uv run python kernel/show_stream.py '2 3 + .'     # the hidden tokens of one block
```

To run the released model (the adapter on Hugging Face) on one GPU or on a Mac, follow [Deploy and test](playbooks/deploy.md). Agents can start from [AGENT.md](AGENT.md).

## Why

Language models are bad at long exact arithmetic. A model that must multiply two 12-digit numbers, or run a recurrence for 100 steps, usually computes by hand in its thinking, slowly and with errors. The usual fix is a tool call: the model writes Python, generation stops, and a host runs the code.

We asked a different question. Can the model's own weights hold an exact machine that the model drives by writing text? If so, can we teach the model to use it well?

Forth fits. It is a stack language in reverse Polish notation (RPN): `2 3 +` means 2 + 3. A stack machine needs no expression parser and no registers, so the layers that the model already has can build it.

We do not claim that this beats tool calling. Tool calling is more general and much faster. The [demos](demos/README.md) show what a machine inside the weights can do with no tool call at all.

## Results

| Claim | Evidence |
|---|---|
| All training together | Stock Lightning 79.1% vs our best adapter (round 7) 92.1% on the frozen test: **+13.0 [+10.2, +15.8]**. Hard problems: 69.1% to 89.7% (+20.6). Median answer length: 3494 to 775 tokens.<sup>5</sup> |
| The kernel in the weights is exact | 618 test cases, 765,734 kernel-owned positions, 0 wrong, in prefill and in cached decode. One more 171k-token block, exact. [Details](kernel/README.md#parity-testing). |
| The in-weights machine gives the same results as the external one | Round-3 adapter, 407 dev problems: 89.9% in the weights (vLLM composite) vs 89.3% with the external machine. 395 of 407 turns exact at every kernel position.<sup>1</sup> |
| Training makes the model use the machine well | Stock Lightning 81.0% vs round-1 adapter 87.4% on the frozen test: +6.4 [+4.1, +8.8].<sup>2</sup> |
| Program demonstrations teach a new habit | Round-5 adapter vs round-3 adapter: +2.0 [+0.1, +4.0] on the frozen test, +11.0 [+5.5, +16.5] on the code-heavy dev set.<sup>3</sup> |
| The model then learns from its own programs | Round-7 adapter vs round 5: frozen +0.9 [-1.0, +2.8], dev x4 +1.2 [-0.1, +2.6], code-heavy dev +5.0 [-0.2, +10.2]. Small, but positive on every set, with fewer cut-off answers (37 to 22).<sup>5</sup> |
| Plain math does not get worse | MATH-500 with no machine: stock 97.2%, round-3 adapter 97.7% (+0.5 [-0.4, +1.4]). Round 5 vs stock: +0.0 [-1.1, +1.1].<sup>4</sup> |
| The composite runs batched in vLLM | 10 hard problems (demo10), 64 streams at once: 50 s per problem. The HF composite at batch 1 took a median of 457 s per problem on the demo set. [Details](kernel/README.md#serving-in-vllm). |

The kernel's claim is exactness. The score gains come from training. Most training and evaluation ran the spec as an external machine: a Python loop that appends each output block. It gives the same tokens as the kernel (row 2), and it runs on a plain vLLM server.

A delta is in points, paired by problem, with a 95% bootstrap interval over problems. Both arms of every comparison use the same problems, engines, card (the system prompt that describes the machine), thinking budget and grader:

1. Dev set (407 problems), card v3, think 12000 / turn 24000. External machine: 2 samples per problem. In the weights: 1 sample.
2. Frozen test (600 problems, 2 samples each), card v2, think 12000 / turn 24000, grader of 2026-11-05.
3. Frozen test x2 and the 200-problem code-heavy dev set x2, card v4, grader of 2026-11-06, adapters on the same LoRA engines.
4. MATH-500 x2, no machine, the same engine path for both arms.
5. Frozen test x2, three arms run together on the same LoRA engines (stock as a zero adapter, round 5, round 7), card v4, think 12000 / turn 24000, grader of 2026-11-06. Dev and code-heavy dev for round 7 as in note 3.

The ledger of rounds is in [training](training/README.md#the-rounds).

## The story, page by page

1. [How we got here](kernel/history.md). A first machine in Qwen3.5-0.8B was exact, but the small model could not drive it.
2. [The kernel](kernel/README.md). A second, small NemotronH stack (27 layers) with every weight computed by a Python builder. Numbers live as rows of Mamba-2 state, 20 decimal digits each. The composite runs it beside stock Lightning, which stays byte-identical. Where the kernel emits, its logits win. Parity tests compare every kernel position with the spec. They found real bugs, for example [a counter that drifted](kernel/README.md#bugs-that-the-tests-found) after 80,000 kernel tokens.
3. [The Forth dialect](forth/README.md). The spec, [`kernel/text_vm.py`](kernel/text_vm.py), defines the exact token stream. If the kernel cannot do a feature exactly, the feature leaves the spec.
4. [Training](training/README.md). On easy problems the machine hurts (88.7% without it, 75.6% with the full card). On hard ones it helps (28.3% to 73.1%). So the goal was to teach when to use it, not only how. LoRA SFT gave the gains: a 60-row cold start, rounds of the model's own clean correct rollouts, then program demonstrations, then the model's own programs. Context distillation and GRPO did not help.
5. [Demos](demos/README.md). Real transcripts: stock against trained, the kernel in the weights, errors and recovery, one problem per family.

## What is next

- Bigger bases of the same family: Bolt (74B, 9B active) and Super (120B, 12B active). They share the hybrid Mamba-2 / attention / MoE design and the vocabulary size, so the kernel can go beside them after a check of its token ids.
- The weak families: recurrences at full size, and polynomial roots, which have no compact program for this machine.
- A cheaper kernel. One forward pass per kernel step is the main cost: a `/` on 6-digit numbers takes about 115 hidden tokens.

## Repository

| Folder | What it holds |
|---|---|
| [`kernel/`](kernel/) | The spec, the weight builder, parity tests, the composite and the vLLM plugin |
| [`forth/`](forth/) | The dialect and a command-line runner |
| [`training/`](training/) | Held-out sets, cards, and the rollout, row, grading and GRPO tools |
| [`playbooks/`](playbooks/) | Each GPU step as an Ansible playbook, and the [deployment guide](playbooks/deploy.md) |
| [`demos/`](demos/) | Real transcripts |
| [`tests/`](tests/) | CPU tests: the spec self-check, the grader, and every Forth example in the docs |

Run the CPU tests with `uv sync` and `uv run --group dev pytest -q`. We ran the GPU parts on machines with 8 RTX PRO 6000 GPUs. For this repository we ran the deployment paths again on a fresh machine, one GPU each: [Deploy and test](playbooks/deploy.md). We did not run the training steps again.

Model and data ([the rpn-gpt collection](https://huggingface.co/collections/markab/rpn-gpt-6aee2a080aedf1d6e4694814)): the round-7 model is [markab/rpn-gpt-lightning](https://huggingface.co/markab/rpn-gpt-lightning) (merged BF16; the page lists the LoRA, NVFP4, GGUF / Ollama, MLX and kernel-in-the-weights versions), and its training rows and evaluation sets are [markab/rpn-gpt-data](https://huggingface.co/datasets/markab/rpn-gpt-data). The rollouts and the other adapters stay private. The held-out evaluation sets are also in [`training/data/`](training/data/).

## Credits and license

The idea of compiling a machine into a language model's weights comes from [qwenputer](https://github.com/divagr18/qwenputer) by Divyansh Agrawal. Our first, Qwen-based phase reused its MIT-licensed weight compiler and carry-lookahead construction. The Nemotron kernel here is a new construction, but the [LICENSE](LICENSE) keeps the upstream copyright notice.

Nemotron 3.5 Lightning is NVIDIA's model, under its own license. This repository holds no model weights.

## Credit and citation

The MIT license requires that you keep the copyright and permission notice in copies. In addition, we ask you, politely, to credit rpn-gpt with a link to this repository when you use or change it. [CITATION.cff](CITATION.cff) has the citation data.

```bibtex
@misc{berry2026rpngpt,
  author       = {Mark Berry},
  title        = {rpn-gpt: An exact Forth machine compiled into the weights of a language model},
  year         = {2026},
  howpublished = {\url{https://github.com/markab21/rpn-gpt}}
}
```
