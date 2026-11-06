# The kernel

The kernel is a Forth machine built by hand as neural-network weights. Nothing in it is trained. It sits beside NVIDIA Nemotron 3.5 Lightning in one model, the composite. Both read the same token stream. When the model closes a ` ```forth ` block, the kernel takes over the next positions and writes the ` ```output ` block, one token per forward pass. Then the model continues.

This page shows how a stack machine fits in a language model's layers, how we made it exact, and how we test it. [How we got here](history.md) tells the earlier Qwen phase.

## A block from the inside

This is the real token stream of one block. [`show_stream.py`](show_stream.py) prints it from the spec.

```
$ uv run python kernel/show_stream.py '2 3 + .'
'```' 'forth' '\n' '2' ' ' [LIT] [MD[2]] '3' ' +' [LIT] [MD[3]] ' .\n' [ADD] [CLR1] [MD[3]] [CLR0] [MD[2]]
[SPL] [MD[5]] [END] '```' [PR] [MD[5]] [PSP] {'\n'} {'```'} {'output'} {'\n'} {'5'} {'\n'} {'```\n'}
31 tokens: 9 model, 15 hidden, 7 visible kernel
```

Quoted tokens are the model's text. Tokens in brackets are hidden kernel tokens. Tokens in braces are the visible output block, which the kernel also writes.

A hidden token is one of the 982 reserved, untrained special tokens in Lightning's tokenizer. It decodes to an empty string, so a reader never sees it. At each hidden position the kernel does one step of work, and the token it picks says which step comes next.

Read the stream from left to right:

1. The model writes `2`. The next token starts with a space, so the word is complete. The kernel writes `[LIT] [MD[2]]`: push the literal 2 (`MD[d]` is machine digit d).
2. The model writes ` +`. At ` .\n` the kernel writes `[ADD]`. The add happens inside that token's forward pass.
3. `[CLR1] [MD[3]] [CLR0] [MD[2]]` erase the two inputs from the stack. `[SPL] [MD[5]]` writes the sum to the top.
4. The model closes the block. `[PR] [MD[5]] [PSP]` is the hidden print of `5` and a space. Then the kernel writes the visible block.

Work costs tokens. For 6-digit numbers, `dup` and `drop` take about 16 hidden tokens, `+` about 31, `mod` 68, `*` 77, and `/` 115. A loop of 100 passes over `dup 100 / +` takes about 20k hidden tokens.

## Numbers in a Mamba-2 state

Lightning's layers are a hybrid of Mamba-2 mixers, attention, and mixture-of-experts (MoE) blocks. A Mamba-2 mixer keeps a state matrix that each token updates with a decay and an input. The kernel uses that state as memory.

The rules that make it exact:

- Every head has a decay of exactly 1 (`A_log = -60`), so nothing fades. The only wipe uses a step so large that the decay becomes exactly 0 in fp32.
- One stack entry is one row of state. The 20 digits of a number sit in 20 channels, most significant digit first, with a marker and a sign channel.
- A write adds the value. A clear adds the exact negative of the value, so the kernel re-emits the digits in clear mode. That is why the stream above shows `[CLR1] [MD[3]]`: it erases the 3.
- bf16 is exact for integers up to 256 only. So no feature in the residual stream goes above 256. Counters are kept as `lo + 128 * hi`, with carry and borrow tokens that the kernel emits.
- Addition and subtraction use carry lookahead in an MLP layer: the carries of all 20 digits in one pass.
- `*` and `mod` run one digit at a time (one `[MSTEP]` per digit of the top operand). `/` runs as `mod` and keeps the quotient beside the remainder.

Mamba-2 has no delta-rule overwrite, so a cell cannot be replaced in place. Also, a diagonal state cannot track the permutations that `swap` and `over` build up. So the kernel computes eagerly: each operation runs at its own token and writes its result to a row.

The kernel has 27 layers: 19 Mamba-2 mixers, 7 MLPs, and one attention layer that does nothing. (The cache code in `generate()` needs one attention layer.) Its residual stream is 6144 wide, and 5543 of those dimensions carry features. The layout is in [`results/parity_2026-11-06.json`](results/parity_2026-11-06.json).

## Running programs

A definition (`: sq dup * ;`) is stored when the model writes it. Its code goes into code cells in Mamba state, and its name goes into a dictionary. A call runs by replay: the kernel emits each code cell as a token, and it executes that cell when the token comes back in as input. Return addresses go on a return stack, which is also row-addressed.

```
$ uv run python kernel/show_stream.py ': sq dup * ; 12 sq .'
'```' 'forth' '\n' ':' ' sq' ' dup' [DEF] [NM] ' *' [DUP] ' ;' [MUL] ' ' [SEMI] '1' '2' ' sq' [LIT] [MD[1]] [MD[2]]
' .\n' [CALL] [NM] [A0] [DUP] [SPL] [MD[1]] [MD[2]] [END] [MUL] [SPX] [MD[1]] [MD[2]] [MSTEP] [CLR2] [MD[1]] [MD[2]]
[CLR1] [MD[1]] [MD[2]] [CLR0] [MD[1]] [MD[2]] [SPL] [MD[1]] [MD[4]] [MD[4]] [END] [EXIT] [DONE] '```' [PR] [MD[1]]
[MD[4]] [MD[4]] [PSP] {'\n'} {'```'} {'output'} {'\n'} {'1'} {'4'} {'4'} {'\n'} {'```\n'}
65 tokens: 14 model, 42 hidden, 9 visible kernel
```

While the model writes `: sq dup * ;`, the kernel only stores (`[DEF] [NM] [DUP] [MUL] [SEMI]`). At `sq` it calls (`[CALL] [NM] [A0]`) and replays `[DUP]` and `[MUL]` from code memory until `[EXIT] [DONE]`.

Control flow (`if`, loops, `leave`, `exit`) compiles in place, with per-level registers for up to 4 open structures. Top-level control code goes into a hidden definition that runs once.

## Reading plain text

The model writes ordinary Forth text, not special tokens. The kernel reads that text causally, with no lookahead:

- A fence opens only at ` ```forth ` followed by a newline. Prose such as "a ```forth block" opens nothing.
- A word ends at a token that starts with whitespace, or at the closing fence. The kernel emits the word's translation one token later.
- A name is keyed by two sums over its tokens, modulo primes, so names of any length work. The dictionary holds each name's kind, so `x @` and `sq` resolve correctly.
- Numbers, words, comments, and every printable character at token edges follow [`text_vm.py`](text_vm.py).

## The composite

The composite is one model with two parts: stock Lightning, untouched, and the kernel as its own submodule. [`composite/modeling_forth_lightning.py`](composite/modeling_forth_lightning.py) is the Hugging Face class (`trust_remote_code`).

- The kernel's logits are exactly 0 except where it emits. Where its top logit is over 0.5, it owns the next token. There the composite uses the kernel's logits times 2^16, so any sampler picks the kernel's token.
- Everywhere else, the logits are Lightning's own, bit for bit.
- The base checkpoint stays byte-identical. The composite stores only the kernel weights, a few small override rows, and a pointer to the base.
- The base reads only the visible stream: the model's text and the visible output block. Hidden tokens go to the kernel only, and the base's cache does not advance on them.

Why the visible-only view: hidden tokens made the base's next-token loss 2 to 5 times worse on the text that followed. With the visible-only view, the base sees exactly what its training rows show. So a LoRA trained on plain-text rows runs on the composite with no change.

Why a composite and not one merged checkpoint: we built a merged model first, with every Lightning weight padded to the kernel's width. It was about 150 GB and too slow to iterate on. The composite also lets the base be quantized to NVFP4 while the kernel stays exact bf16. LoRA adapters train on stock Lightning. The same kernel can go next to a bigger base.

## Serving in vLLM

[`composite/vllm_forth.py`](composite/vllm_forth.py) is an out-of-tree vLLM 0.29 model. It holds two stock vLLM NemotronH backbones under `base.` and `kernel.`, each with its own cache. [`composite/vllm_visible.py`](composite/vllm_visible.py) builds the base's visible-only view inside vLLM's model runner (`FORTH_VISIBLE=1`).

Measured on one GPU with an NVFP4 base and the round-1 LoRA:

| Workload | Stock Lightning | Composite |
|---|---|---|
| Plain text, 1 / 8 / 64 streams | 166 / 1249 / 5477 tok/s | 126 / 918 / 3611 tok/s |
| 10 Forth problems (demo10), 64 streams at once | | 50 s per problem (600 tok/s, 86% hidden tokens) |
| For scale: the HF composite at batch 1, 89-problem demo set | | median 457 s per problem |

The exactness rules for vLLM: the kernel's chunk size (64), bf16 GEMMs with cuBLAS reduced-precision reduction off, no prefix caching, no async scheduling, and `FULL_DECODE_ONLY` CUDA graphs. With those, all 264 kernel test cases were exact in batches of 64 sequences, on BF16 and NVFP4 bases.

## Parity testing

Parity means that the weights and the spec produce the same tokens. The spec is [`text_vm.py`](text_vm.py), a Python machine that emits the kernel's exact token stream. The kernel, the training rows, and the simulated rollouts all follow it.

| Test | What it checks | Where it runs |
|---|---|---|
| `text_vm.py` self-check | 100 programs, 16 front-end cases, 17 multi-block turns, 26 whole-block reports | CPU (`tests/test_spec.py`) |
| [`sp_test.py`](sp_test.py) | Every kernel-owned position of every case, compiled kernel vs spec, full prefill and cached decode, with a top-2 logit margin | GPU |
| [`replay.py`](replay.py) | Real saved model programs through the current spec: which outputs and answers a change moves | CPU |
| [`code_measure.py`](code_measure.py) | Code-memory variants on real turns, before any kernel build | CPU |
| [`composite/vllm_kernel_check.py`](composite/vllm_kernel_check.py) | The same positions through the vLLM plugin, batched | GPU |
| [`composite/e2e.py`](composite/e2e.py) | Real generations: the model writes, the kernel answers, every kernel position checked against the spec | GPU |

The last full kernel run (2026-11-06, [`results/parity_2026-11-06.json`](results/parity_2026-11-06.json)):

| Cases | Tokens | Kernel-owned positions | Wrong |
|---|---|---|---|
| 618 (312 from real model turns) | 938,230 | 765,734 | 0 in prefill, 0 in cached decode (596 cases) |

One more case is a single 171k-token block, exact token by token.

End to end, real generations match the spec in almost every turn. With the round-3 adapter in the vLLM composite, 395 of 407 dev turns were exact at every kernel position. The inexact turns found kernel gaps that later releases fixed ([in the weights](../demos/in-the-weights.md)).

A spec change must pass the self-check and then a replay of real rollouts. For example, freeing top-level code (below) changed 310 of 11,790 real block outputs, lost 0 answers, and gained 114.

## Bugs that the tests found

| Bug | How it showed | Fix |
|---|---|---|
| NOTEJ: a layer-5 counter of tokens since `output` grew by 1 per kernel token inside a long run. Its square enters every RMSNorm. | Near 79,900 uninterrupted kernel tokens, features read 0.2% low and a decode flipped. 5 of 204 in-weights dev turns. | Reset the counter at every hidden token (+1 dimension). An audit found no other counter that grows inside a block. |
| A hidden `gcd` or `lcm` defined after the closing fence stored a 0 as data | Wrong values in 1 of 3733 rollouts | A register for the end phase (+1 dimension) |
| The "code memory full" flag turned off again past cell 199 | In-weights output diverged from the spec after code memory filled | Threshold from the address one-hots (0 dimensions) |
| Token edges: training rows re-tokenized the text, the composite did not | All 199 machine rows of one round differed at output-block edges. The model ended its turn after every block. | Rows are built segment by segment, with the kernel's own token ids |
| vLLM used the base's chunk size (128) for the kernel | Inexact prefill | Use the smaller chunk size |
| Reduced-precision GEMM paths in the plugin (bf16 split-K, an fp32 upcast inside the compiler) | 31 inexact cases when batched | bf16 GEMMs with reduced-precision reduction off |
| HF's prefill clamps the Mamba step size | Base drift vs vLLM (0.157 vs 0.030 mean log-prob difference) | Clamp removed for the base only |
| A logit tie on the output header after a ` ```\n ` fence | A wrong token at one position | Tests assert top-2 margins, not only the top token |

## Decisions

Each decision came from a measurement, usually a small one. The dialect page lists the [short errors](../forth/README.md#why-the-errors-are-short) and the [dropped features](../forth/README.md#what-we-dropped-and-why).

- Plain text, not special tokens. First the model wrote Forth words as special tokens. With thinking on, that adapter wrote programs unrelated to its own reasoning (2 of 15 right). Stock Lightning, writing plain Forth with 3 examples, solved 130 of 200 problems in a suite that needs a machine, and 76% of one family where it solved 12% unaided. The model does not think in special tokens.
- Free top-level code. At the first "code memory full", top-level code that already ran filled 54% of memory, old copies of re-defined words 27%, and live words 19%. Freeing the first kind avoided 45% of the fulls. It costs 28 dimensions and no new layer.
- Check the 32-variable limit and "code memory full" at the name, before anything changes. Before this fix, the kernel dropped stores past 32 variables without an error.

## Tools

The kernel is a stock NemotronH configuration, so Hugging Face transformers, vLLM and NeMo AutoModel all load it with no custom code. We trained with NeMo AutoModel. Its forward pass matched vLLM BF16 at the noise floor (mean log-prob difference 0.031 per token). Hugging Face transformers drifted (0.195, growing with length), so we never used it for log-probs. vLLM 0.29 served the official NVFP4 checkpoint with a LoRA on its MoE at 184 tok/s, against 206 tok/s without the adapter.

## Files

| File | Use |
|---|---|
| [`text_vm.py`](text_vm.py) | The spec: plain Forth text to the kernel's exact token stream. `python kernel/text_vm.py` runs its self-check. |
| [`sp_vm.py`](sp_vm.py) | The kernel's token protocol and reference token machine |
| [`text_front.py`](text_front.py) | The plain-text front end, as a Python reference |
| [`sp_kernel.py`](sp_kernel.py) | The builder: computes every weight of the kernel as a stock NemotronH checkpoint |
| [`sp_test.py`](sp_test.py) | GPU parity test of a built kernel against the spec |
| [`mamba_kernel_test.py`](mamba_kernel_test.py) | The first probe: can a Mamba-2 mixer hold an exactly overwritable row? The builder reuses its constants. |
| [`stream_kcheck.py`](stream_kcheck.py) | Long streams, token by token with the cache |
| [`sp_vllm.py`](sp_vllm.py) | The kernel alone in vLLM |
| [`replay.py`](replay.py), [`code_measure.py`](code_measure.py) | Measure a spec change on real programs |
| [`show_stream.py`](show_stream.py) | Print one block's stream |
| [`composite/`](composite/) | The composite: HF class, builder, vLLM plugin, end-to-end evaluation |
| [`tests/kernel_turns_2d.json`](tests/kernel_turns_2d.json) | Multi-block test turns for `sp_test.py` |

To build and test on a GPU, follow the [playbooks](../playbooks/README.md).
