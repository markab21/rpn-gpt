<p align="center"><img src="assets/banner.jpg" alt="rpn-gpt: a stack machine inside the model. Exact computation. No tool calls." width="100%"></p>

# rpn-gpt

Models: [collection](https://huggingface.co/collections/markab/rpn-gpt-6aee2a080aedf1d6e4694814) · [kernel in the weights (`-Forth`)](https://huggingface.co/markab/rpn-gpt-lightning-Forth) · [merged BF16](https://huggingface.co/markab/rpn-gpt-lightning) · [LoRA](https://huggingface.co/markab/rpn-gpt-lightning-LoRA) · [GGUF / Ollama](https://huggingface.co/markab/rpn-gpt-lightning-GGUF) · [MLX 4-bit](https://huggingface.co/markab/rpn-gpt-lightning-MLX-4bit) · [data](https://huggingface.co/datasets/markab/rpn-gpt-data)

Language models make mistakes in long exact arithmetic. The usual fix is a tool call: the model writes code, generation stops, and a host program runs it. This project tries something else. Next to a language model, inside the same model, we put a small computer. The language model writes a program as plain text. The computer runs it and writes the result as the next tokens. Generation never stops, and no program runs outside the model.

Nobody trained the computer. A Python program calculated each of its weights. We call it the kernel. The language model is NVIDIA Nemotron 3.5 Lightning (30B parameters, 3B active), and we trained a LoRA adapter so that it uses the kernel well.

![A live request: the model's first program breaks a machine limit, the kernel inside the model answers with an error line, and the model fixes the program](demos/live-error.svg)

A live request to [`markab/rpn-gpt-lightning-Forth`](https://huggingface.co/markab/rpn-gpt-lightning-Forth) in vLLM on one RTX PRO 6000, in real time: 17 s from question to answer. The model's first program puts the 12-digit number where `*` allows at most 10 digits. The kernel, inside the same model, answers with an error line. The model swaps the operands, and the kernel writes the product. Nothing runs outside the model. [All four recordings](demos/README.md#live-recordings), one of them animated.

## Two networks, one token stream

```text
       model text   ```forth  2 3 + .  ```   ```output  5  ```
            │                                     │
            │ visible tokens only                 │ every token, hidden ones too
            ▼                                     ▼
 ┌─────────────────────────────┐    ┌─────────────────────────────┐
 │ Lightning + our LoRA        │    │ The kernel: 27 layers       │
 │ base weights unchanged      │    │ weights computed, not       │
 │ writes thinking, prose and  │    │ trained; keeps the stack    │
 │ Forth                       │    │ in its Mamba-2 state        │
 └──────────────┬──────────────┘    └──────────────┬──────────────┘
                └────────────► next token ◄────────┘
           Lightning's token, except where the kernel emits
```

- Lightning is the stock checkpoint, byte for byte. The kernel is a second, small network of standard layer types (19 Mamba-2 mixers, 7 MLPs, and one attention layer that does nothing). A Python builder, [`kernel/sp_kernel.py`](kernel/sp_kernel.py), computes every weight.
- One number on the stack is one row of Mamba-2 state, 20 decimal digits in 20 channels. Every decay is exactly 1, so nothing fades.
- The kernel's logits are 0 except where it has a step to do. There the composite multiplies them by 2^16, so any sampler picks the kernel's token.
- Most kernel tokens are hidden. A hidden token is one of the 982 reserved, untrained special tokens in Lightning's vocabulary. It decodes to an empty string. Each hidden token is one forward pass and one step of the machine, and the token the kernel picks names the next step.
- Lightning reads only the visible tokens. To Lightning, the output block simply appears after the closing fence. That is why an adapter trained on plain-text transcripts runs on the composite with no change.

## One block, token by token

Forth is a stack language in reverse Polish notation (RPN): `2 3 +` means 2 + 3. A stack machine needs no expression parser, so each word can run at its own token. This is the full token stream of `2 3 + .`, printed by the spec, [`kernel/text_vm.py`](kernel/text_vm.py), a Python machine that emits the kernel's exact tokens:

```
$ uv run python kernel/show_stream.py '2 3 + .'
'```' 'forth' '\n' '2' ' ' [LIT] [MD[2]] '3' ' +' [LIT] [MD[3]] ' .\n' [ADD] [CLR1] [MD[3]] [CLR0] [MD[2]]
[SPL] [MD[5]] [END] '```' [PR] [MD[5]] [PSP] {'\n'} {'```'} {'output'} {'\n'} {'5'} {'\n'} {'```\n'}
31 tokens: 9 model, 15 hidden, 7 visible kernel
```

Quoted tokens are the model's text. Bracketed tokens are hidden kernel steps. Tokens in braces are the visible output block, which the kernel also writes.

1. The model writes `2`. The next token starts with a space, so the word is complete. The kernel does not wait for the block to end: it writes `[LIT] [MD[2]]`, push the literal 2.
2. At ` .\n` the kernel writes `[ADD]`. The add happens inside that one token's forward pass.
3. `[CLR1] [MD[3]] [CLR0] [MD[2]]` erase the two inputs. `[SPL] [MD[5]]` writes the sum.
4. The model closes the block. `[PR] [MD[5]] [PSP]` is the hidden print. Then the kernel writes the visible output, and the model continues.

## A real program

`2 3 +` looks like a calculator. The model writes real programs. The question: "A random number generator starts at 43392 and repeatedly replaces x with (21066x + 36252) mod 161691. What is the value after 136 steps?" After its thinking, the model wrote this block, and the kernel wrote the output:

```forth
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

12036 is correct. The model wrote 806 visible tokens. Between the closing fence and the output, the kernel ran 48,968 hidden tokens: 136 passes of a multiply, an add and a `mod`, each replayed from code memory. Every hidden token matched the spec. This run used the round-1 adapter ([full transcript](demos/in-the-weights.md#a-loop-of-136-steps)). The released model (round 7) passed the same in-weights check in our [deploy test](playbooks/deploy.md#hf-transformers-reference-batch-1): 974 of 974 kernel tokens exact.

## When the machine helps, and when it hurts

The kernel was exact from the start. Using it well was the hard part. Before any training, we gave stock Lightning the machine through its system prompt:

| Problems | No machine | With the machine |
|---|---|---|
| Easy (260 DeepMind math problems) | 88.7% | 75.6% |
| Hard (90 multi-step problems) | 28.3% | 73.1% |

On easy problems the machine hurts: the model does them well by hand, and the machine costs tokens. On hard problems it more than doubles the score. So training had to teach when to use the machine, not only how. After one round, the model wrote a block in 77.1% of hard answers (stock: 55.2%) and in 8.4% of easy ones (stock: 23.7%). [Before and after](demos/before-after.md) shows real transcripts, for example stock Lightning getting the right output from the machine, not trusting it, and starting a long division by hand.

## Results

<!-- BENCH: arithmetic benchmark chart vs stock / stock+Python / Qwenputer -->

| Question | Result |
|---|---|
| Does training help? | On 600 held-out problems: stock Lightning 79.1%, trained model (round 7) 92.1%, **+13.0 points** [+10.2, +15.8]. On the 194 hard problems: 69.1% to 89.7%. Median answer length: 3494 to 775 tokens. |
| Is the kernel exact? | 618 test cases, 765,734 tokens written by the kernel, 0 wrong, in prefill and in cached decode. One more 171k-token block, also exact. [Parity tests](kernel/README.md#parity-testing). |
| Does the kernel score the same as the spec? | Round-3 adapter, 407 problems: 89.9% with the kernel in the weights, 89.3% with the spec. 395 of 407 turns exact at every kernel position. |
| Does plain math get worse? | MATH-500 with no machine: stock 97.2%, round-3 adapter 97.7%. Round 5 against stock: +0.0 points. |

The score gain comes from training. The kernel's claim is exactness. Most training and evaluation ran the spec outside the model, as a Python loop that appends each output block. It writes the same tokens as the kernel (row 3) and runs on an ordinary vLLM server.

How we measured: a problem counts only if its final answer equals the exact answer. Each comparison runs both models on the same problems, system prompt, thinking budget (12,000 tokens) and grader, and pairs results problem by problem. Intervals are 95% bootstrap over problems. [Training](training/README.md#the-rounds) has every round and its settings.

## What this buys, and what it costs

| | Tool call (Python, host) | rpn-gpt (Forth, in the model) |
|---|---|---|
| Who runs the code | A host program, in a sandbox | The model's own forward passes |
| Round trip | Generation stops; the context is sent again | None: the output is the next tokens |
| Language | Python, with libraries | A small Forth: integers of 20 digits, 32 variables, 192 code cells |
| Speed | Microseconds for a loop | One forward pass per step: the 136-step loop took 430 s at batch 1 |

What it buys: a block can go anywhere, also inside the thinking. Exactness is testable at every token. Training needs only plain text.

What it costs: speed above all. One `/` on two 6-digit numbers takes about 115 hidden tokens. In vLLM with 64 streams at once, 10 hard problems took about 50 s each, and 86% of the tokens were hidden. The kernel is exact only on vLLM 0.29.0 with fixed settings.

We do not claim this beats tool calling. Tool calling is more general and much faster. The open questions:

- Can a kernel step cost less than a full forward pass?
- Bolt (74B, 9B active) and Super (120B, 12B active) share Lightning's design and vocabulary size. The kernel can go beside them after a check of its token ids. Does a bigger model use it better?
- Recurrences at full size, and polynomial roots, which have no compact program for this machine, stay weak.

## Quickstart

On a laptop, with no GPU, run the spec (Python, not the model):

```sh
git clone https://github.com/markab21/rpn-gpt && cd rpn-gpt
uv sync
uv run python -m forth.run '82225043186179281 6469193 mod .'
```

On one NVIDIA GPU (Linux, driver 580 or newer, 80 GB or more), run the model with the kernel in the weights. We tested this on a 96 GB RTX PRO 6000 Blackwell. The first start downloads about 70 GB.

```sh
uv venv -p 3.12 venv-kernel
VIRTUAL_ENV=$PWD/venv-kernel uv pip install -r requirements/vllm-cu130.txt
FORTH_VISIBLE=1 VLLM_DISABLE_COMPILE_CACHE=1 PYTHONPATH=$PWD/kernel/composite \
  venv-kernel/bin/vllm serve markab/rpn-gpt-lightning-Forth --served-model-name forth --max-model-len 65536 --host 127.0.0.1 \
  --no-enable-flashinfer-autotune --no-enable-prefix-caching --no-async-scheduling -cc '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

In a second terminal, ask a question and watch the answer stream:

```sh
venv-kernel/bin/python demos/live.py "Calculate the remainder when 82225043186179281 is divided by 6469193." \
  --base-url http://localhost:8000 --model forth
```

Without a big GPU, use the 4-bit model in Ollama (24.5 GB). Here the spec runs the blocks on the CPU, outside the model:

```sh
ollama pull hf.co/markab/rpn-gpt-lightning-GGUF:Q4_K_M
uv run python rpn_chat.py "Calculate the remainder when 82225043186179281 is divided by 6469193." \
  --server ollama --base-url http://127.0.0.1:11434 --model hf.co/markab/rpn-gpt-lightning-GGUF:Q4_K_M
```

The answer must be `1290731`. Every tested path (BF16, NVFP4, GGUF, MLX on a Mac) and its measured result is in [Deploy and test](playbooks/deploy.md). Agents can start from [AGENT.md](AGENT.md).

## The story, page by page

1. [How we got here](kernel/history.md). A first machine in Qwen3.5-0.8B was exact, but the small model could not drive it.
2. [The kernel](kernel/README.md). How it stores numbers, runs programs and reads plain text, and the real bugs the parity tests found, for example [a counter that drifted](kernel/README.md#bugs-that-the-tests-found) after 80,000 kernel tokens.
3. [The Forth dialect](forth/README.md). The words, the limits and the error lines. If the kernel cannot do a feature exactly, the feature leaves the dialect.
4. [Training](training/README.md). A 60-row cold start, rounds of the model's own correct answers, program demonstrations, then the model's own programs. Reinforcement learning (GRPO) and context distillation did not help.
5. [Demos](demos/README.md). Real transcripts: stock against trained, the kernel in the weights, errors and recovery, one problem per family.

## Repository

| Folder | What it holds |
|---|---|
| [`kernel/`](kernel/) | The spec, the weight builder, parity tests, the composite and the vLLM plugin |
| [`forth/`](forth/) | The dialect and a command-line runner |
| [`training/`](training/) | Held-out sets, system prompts, and the rollout, row, grading and GRPO tools |
| [`playbooks/`](playbooks/) | Each GPU step as an Ansible playbook, and the [deployment guide](playbooks/deploy.md) |
| [`demos/`](demos/) | Real transcripts, live recordings and the live viewer |
| [`tests/`](tests/) | CPU tests: the spec self-check, the grader, and every Forth example in the docs |

Run the CPU tests with `uv sync` and `uv run --group dev pytest -q`. The training rows and evaluation sets are in [markab/rpn-gpt-data](https://huggingface.co/datasets/markab/rpn-gpt-data).

## Credits, license and citation

The idea of compiling a machine into a language model's weights comes from [qwenputer](https://github.com/divagr18/qwenputer) by Divyansh Agrawal. Our first, Qwen-based phase reused its MIT-licensed weight compiler and carry-lookahead construction. The Nemotron kernel is a new construction, and the [LICENSE](LICENSE) keeps the upstream notice.

Nemotron 3.5 Lightning is NVIDIA's model, under its own license. This repository holds no model weights.

MIT license. Please credit rpn-gpt with a link to this repository when you use or change it. [CITATION.cff](CITATION.cff) has the citation data.
