# Deploy and test rpn-gpt

This page shows how to run the released model on a fresh machine. Each section gives the commands we ran, the versions, the time each step took, and the result. The playbooks in [`ansible/`](ansible/) do the same steps for an 8-GPU research machine. This page needs no Ansible and one GPU.

The release has three parts:

- The adapter: [`markab/rpn-gpt-lightning-LoRA`](https://huggingface.co/markab/rpn-gpt-lightning-LoRA). It is the round-7 LoRA adapter (`s2a_r7`, rank 16, 23 MB) and its system prompt, the card `reference_v4.txt`. The same model, merged and packaged, is [`markab/rpn-gpt-lightning`](https://huggingface.co/markab/rpn-gpt-lightning) (BF16) with `-NVFP4`, `-GGUF`, `-MLX-4bit`, `-MLX-8bit` and `-Forth` versions: [Path 5](#path-5-packaged-formats).
- The base model: NVIDIA Nemotron 3.5 Lightning, as [BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) (61 GB) or [NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) (21 GB).
- This repository: the external machine ([`training/loop_sim.py`](../training/loop_sim.py)), the kernel builder, the composite, and the evaluation sets.

The model needs a machine that runs its Forth blocks. You can use one of two machines:

- The external machine. A Python loop sends the turn to a model server. When the model closes a ` ```forth ` block, the loop runs the block in the spec ([`kernel/text_vm.py`](../kernel/text_vm.py)) and appends the ` ```output ` block. Any server that serves the base model with a LoRA adapter works.
- The kernel in the weights (the composite). The composite is the base model plus the compiled kernel as a second network. The kernel writes the output blocks itself. It needs vLLM 0.29.0 and the plugin in [`kernel/composite/`](../kernel/composite/).

## Tested on

| Item | GPU machine | Mac |
|---|---|---|
| Date | 2026-11-06 | 2026-11-06 |
| Hardware | 1 x NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB (sm_120), per path | Apple M4 Max, 128 GB |
| Driver / OS | NVIDIA driver 615.71.09, Ubuntu 24.04.4 | macOS 27.0.1 |
| Base image | NGC PyTorch 26.08 container (CUDA 13.4). We did not use its Python packages. | none |
| Python | 3.12.3, in a new uv venv | 3.12, in a new uv venv |

## Versions

All packages came from wheels. Nothing needed a source build. The full pins are in [`requirements/`](../requirements/).

| Package | GPU path ([`vllm-cu130.txt`](../requirements/vllm-cu130.txt)) | Mac path ([`mlx-macos.txt`](../requirements/mlx-macos.txt)) |
|---|---|---|
| vLLM | 0.29.0 | not used |
| torch | 2.13.0+cu130 (CUDA 13.0 wheels, driver 580 or newer) | not used |
| transformers | 5.16.1 | 5.22.2 |
| peft | 0.21.2 | not used |
| flashinfer-python | 0.6.18 | not used |
| triton | 3.7.1 | not used |
| math-verify | 0.9.0 | not used |
| mlx / mlx-lm | not used | 0.32.4 / 0.32.2 |
| huggingface_hub | 1.33.0 | 2.4.0 |
| llama.cpp (Path 5) | `6c01da627` (2026-11-06), CUDA build for sm_120 | not used |
| Ollama (Path 5) | 0.40.15 (Linux tarball) | not used |

Keep vLLM at 0.29.0. The kernel is exact on vLLM 0.29.0. vLLM 0.32.0 emitted wrong kernel tokens in its compiled mode. You do not need `mamba-ssm` or `causal-conv1d`. vLLM has its own Mamba kernels. The HF composite runs without them, but slower.

Pinned base revisions:

| Checkpoint | Revision |
|---|---|
| `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` | `a9904d24bcc1d289a1950fa9d2b978c47cf903b9` |
| `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` | `bee7596271d1495f6992ae224aefde4410e816b8` |

## Results

| Path | GPUs | Set | Result | Time |
|---|---|---|---|---|
| BF16 + LoRA, external machine | 1 | frozen test x1 (600) | s2a_r7 91.7% (median 781 tokens). Stock (zero adapter) 79.5%. Paired +12.2 [+8.8, +15.7] | 25 min (s2a_r7), 47 min (stock) |
| BF16 + LoRA, external machine | 1 | demo10 x4 (40) | 37 of 40 | 5 min 42 s |
| NVFP4 + LoRA, external machine | 1 | demo10 x4 (40) | 34 of 40 | 4 min 53 s |
| NVFP4 + LoRA, external machine | 1 | frozen test x1 (600) | s2a_r7 87.5% (median 852 tokens): -4.2 [-7.0, -1.3] vs BF16, paired | 21 min |
| Composite in HF transformers | 1 | 1 problem | correct, 974 of 974 kernel tokens exact | 2 min 58 s, 28.6 tokens/s |
| Composite in vLLM (offline batch) | 1 | demo10 (10 hard problems) | 7 of 10 correct, 10 of 10 kernel-exact | 15 min (all at once) |
| Composite in vLLM (offline batch) | 1 | 16 dev problems (big arithmetic) | 15 of 16 correct, 16 of 16 kernel-exact | 4 min 53 s (all at once) |
| Composite in vLLM (OpenAI server) | 1 | 16 dev problems (big arithmetic) | 16 of 16 correct, every kernel output block equal to the spec | 2 min 32 s (all at once) |
| Mac, MLX NVFP4 + LoRA, external machine | Apple GPU | 3 problems from demo10 | 3 of 3 correct | 4 min 17 s |
| Merged BF16 (`markab/rpn-gpt-lightning`), external machine | 1 | dev_round[:100] x2 | 93.0%; BF16 + LoRA on the same problems 89.5%, +3.5 [-1.5, +8.5]. From the Hub (empty cache): `rpn_chat.py` correct in 20 s | about 20 min |
| Merged NVFP4 (`-NVFP4`), external machine | 1 | frozen test x1 (600) | 90.5% (median 806 tokens): -1.2 [-3.7, +1.3] vs BF16 + LoRA, +3.0 [+0.2, +5.8] vs NVFP4 + LoRA | about 30 min |
| GGUF Q4_K_M, llama-server, external machine | 3 | frozen test x1 (600) | 91.2% (median 749 tokens): -0.5 [-3.2, +2.2] vs BF16 + LoRA | about 2 h |
| GGUF Q8_0, llama-server, external machine | 3 | frozen test x1 (600) | 93.5% (median 669 tokens): +1.8 [-0.8, +4.5] vs BF16 + LoRA | about 2 h |
| Mac, MLX 4-bit (`-MLX-4bit`), `rpn_chat.py --server mlx` | Apple GPU | 3 problems from demo10 | 3 of 3 correct | 4 min, about 62 tokens/s |
| `-Forth` (composite on the merged base), HF transformers | 1 | 3 problems | 3 of 3 correct, 1431 of 1431 kernel tokens exact | 132 s |
| `-Forth`, vLLM OpenAI server, BF16 base | 1 | 3 problems x2 (local copy); 4 chats from the Hub | 6 of 6 correct; from the Hub 4 of 4 correct, 6 of 6 kernel output blocks equal to the spec | under 2 min after start |

The ledger values on the frozen test (x2, the same card and budgets) are 92.1% for s2a_r7 and 79.1% for stock. One sample per problem moves by about 2.5 points.

## Set up a GPU machine

You need Linux x86_64, an NVIDIA GPU with 80 GB or more for BF16 (about 30 GB for NVFP4), driver 580 or newer, about 100 GB of disk, and [uv](https://docs.astral.sh/uv/).

1. Get the code, the environment, and the checkpoints. Run each command from the repository root.

   ```sh
   git clone https://github.com/markab21/rpn-gpt && cd rpn-gpt
   uv venv -p 3.12 venv-kernel
   VIRTUAL_ENV=$PWD/venv-kernel uv pip install -r requirements/vllm-cu130.txt
   venv-kernel/bin/hf download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
     --revision a9904d24bcc1d289a1950fa9d2b978c47cf903b9 --local-dir lightning-bf16
   venv-kernel/bin/hf download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
     --revision bee7596271d1495f6992ae224aefde4410e816b8 --local-dir lightning-nvfp4
   venv-kernel/bin/hf download markab/rpn-gpt-lightning-LoRA --local-dir adapters/rpn-gpt-lightning
   export VLLM_API_KEY=$(openssl rand -hex 16)
   ```

2. Make sure that the environment imports:

   ```sh
   venv-kernel/bin/python -c "import vllm, torch, transformers, peft; print(vllm.__version__, torch.__version__, transformers.__version__, peft.__version__)"
   # 0.29.0 2.13.0+cu130 5.16.1 0.21.2
   ```

Measured: the install took 7 min 40 s on a cold cache (about 4 GB of wheels) and 1.5 s from a warm cache. The BF16 download took about 12 min and the NVFP4 download 6.5 min.

The NGC PyTorch image sets `PIP_CONSTRAINT=/etc/pip/constraint.txt` for its own torch build. We ran the install with `env -u PIP_CONSTRAINT` so that it does not apply to the new venv.

## Path 1: BF16 + LoRA, external machine

This is the main evaluation path. It gave every score in the [results](../README.md#results).

1. Start the server on one GPU. vLLM reads the API key from `VLLM_API_KEY`.

   ```sh
   CUDA_VISIBLE_DEVICES=0 venv-kernel/bin/vllm serve lightning-bf16 --served-model-name lightning \
     --host 127.0.0.1 --port 8200 --max-model-len 65536 --gpu-memory-utilization 0.92 --max-num-seqs 128 \
     --no-enable-flashinfer-autotune --enable-lora --max-lora-rank 16 \
     --lora-modules s2a_r7=$PWD/adapters/rpn-gpt-lightning
   ```

   The first start took 4 min (torch.compile and CUDA graph capture). The weights take 59.8 GiB. A restart with the compile cache took 1.5 min.

2. Run one problem first. The adapter writes a short program and reads the output.

   ```sh
   python3 -c "import json; json.dump(json.load(open('training/data/demo10.json'))[6:7], open('one.json', 'w'))"
   LOOP_MODEL=s2a_r7 LIGHTNING_URL=http://127.0.0.1:8200/v1/completions LOOP_TOKENIZER=$PWD/lightning-bf16 \
     venv-kernel/bin/python training/loop_sim.py one.json training/data/cards/reference_v4.txt \
     --think --think-budget 12000 --max-tokens 24000 --samples 1 --out runs/one.json
   ```

   Expected output (28 s):

   ```text
   OK   2 block(s) | Calculate the remainder when 82225043186179281 is divided by 6469193. | want 1290731 got 1290731
        [1] 82225043186179281 6469193 mod . cr  =>  1290731
   correct 1 / 1
   ```

   The model can send a first block that fails, read the error, and send a fixed block. That is normal.

3. Run the frozen test (600 problems, one sample each):

   ```sh
   LOOP_MODEL=s2a_r7 LIGHTNING_URL=http://127.0.0.1:8200/v1/completions LOOP_TOKENIZER=$PWD/lightning-bf16 \
     venv-kernel/bin/python training/loop_sim.py training/data/frozen_test.json training/data/cards/reference_v4.txt \
     --think --think-budget 12000 --max-tokens 24000 --samples 1 --workers 100 --out runs/frozen_r7.json
   ```

   Measured: 550 of 600 correct (91.7%) in 25 min 29 s. Per kind: hard 170/194, easy 179/190, easy_big 101/110, small 65/70, variant 35/36. The run appends each rollout to `runs/frozen_r7.json.partial.jsonl`. If it stops, the same command resumes it.

To measure stock Lightning on the same engine path, serve an adapter with every `lora_B` set to zero ([`training/zero_adapter.py`](../training/zero_adapter.py)). Do not compare against the base model name. The engine path changes the score by about 2 points.

```sh
venv-kernel/bin/python training/zero_adapter.py adapters/rpn-gpt-lightning adapters/zero
# serve as above with --lora-modules stock=$PWD/adapters/zero, then LOOP_MODEL=stock
```

Measured stock (zero adapter) on the frozen test x1: 477 of 600 correct (79.5%), median 3,735 tokens. It thinks longer than the adapter, so the run took 47 min (28 min on one GPU, then 19 min on two). Paired against s2a_r7: +12.2 points, 95% interval [+8.8, +15.7]. s2a_r7 is better on 98 problems and worse on 25. The ledger value with 2 samples per problem is +13.0 [+10.2, +15.8].

On demo10 x4 with BF16: 37 of 40 correct in 5 min 42 s.

## Path 2: NVFP4 + LoRA, external machine

The NVFP4 base takes 18.7 GiB instead of 59.8 GiB. The adapter was trained on BF16 and loads on the NVFP4 base without a change.

```sh
CUDA_VISIBLE_DEVICES=0 venv-kernel/bin/vllm serve lightning-nvfp4 --served-model-name lightning \
  --host 127.0.0.1 --port 8202 --max-model-len 65536 --gpu-memory-utilization 0.92 --max-num-seqs 128 \
  --no-enable-flashinfer-autotune --enable-lora --max-lora-rank 16 \
  --lora-modules s2a_r7=$PWD/adapters/rpn-gpt-lightning \
  --kv-cache-dtype bfloat16 --enable-prefix-caching --mamba-cache-mode align --async-scheduling
```

Then run `loop_sim.py` as in path 1, with port 8202.

What runs on an RTX PRO 6000 (sm_120): vLLM picks `MarlinNvFp4LinearKernel` and the `MARLIN` NVFP4 MoE backend. These are FP4 weights with bf16 activations (W4A16). vLLM logs "Your GPU does not have native support for FP4 computation". That is expected for this checkpoint on this GPU. Keep the KV cache in bf16 (`--kv-cache-dtype bfloat16`). The FP8 KV cache of the checkpoint makes the cache blocks of the hybrid model twice as large, and fewer prompts hit the prefix cache.

Measured on the frozen test x1: 525 of 600 correct (87.5%) in 20 min 48 s, median 852 tokens. Paired against the BF16 run of path 1 (the same problems, card and budgets), the NVFP4 base costs 4.2 points, 95% interval [-7.0, -1.3]. Most of the loss is on the hard (-6.2) and variant (-16.7, n=36) problems. Use the BF16 base for evaluation. Use NVFP4 when memory or speed matters more.

On demo10 x4 (10 hard problems, 4 samples each): 34 of 40 correct (85%) in 4 min 53 s.

## Path 3: the kernel in the weights (the composite)

The composite is not on the Hub. You build it from the BF16 base. The build computes the kernel weights (3.7B parameters, 7.4 GB in bf16). The base is referenced by path and never copied.

```sh
SP_LIGHTNING=$PWD/lightning-bf16 venv-kernel/bin/python kernel/composite/build.py $PWD/forth-lightning $PWD/lightning-bf16
```

Measured: 64 s on the CPU of the GPU machine (55 s on a Mac CPU, 8 GB of RAM at peak). It writes `forth-lightning/` with `build_info.json`: 453 override ids, 3,698,247,232 kernel parameters.

### HF transformers (reference, batch 1)

```sh
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=kernel/composite:kernel E2E_SET=training/data/demo10.json \
  E2E_SYS=training/data/cards/reference_v4.txt E2E_THINK=12000 E2E_TURN=24000 \
  venv-kernel/bin/python kernel/composite/e2e_demo.py $PWD/forth-lightning runs/hf_composite.json 6 \
  $PWD/adapters/rpn-gpt-lightning
```

The script checks every kernel-owned position against the spec. Measured on problem 6 of demo10: correct (`1290731`), `"exact": true`, 974 of 974 kernel tokens equal to the spec. The turn had 3,875 visible and 859 hidden tokens. It took 2 min 58 s in total, 28.6 tokens/s. Without `mamba-ssm`, HF uses the torch Mamba scan, which is slower but exact.

### vLLM, offline batch

[`e2e_vllm.py`](../kernel/composite/e2e_vllm.py) runs all problems at once in one vLLM engine and checks every kernel position. It uses the NVFP4 base. Make the NVFP4 variant of the composite first, as [`serve_forth.sh`](../kernel/composite/serve_forth.sh) does on its first start, or start the server once.

```sh
FORTH_VISIBLE=1 CUDA_VISIBLE_DEVICES=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_DISABLE_COMPILE_CACHE=1 \
PYTHONPATH=kernel/composite:kernel PROBLEMS=training/data/demo10.json SYSTEM=training/data/cards/reference_v4.txt \
THINK=12000 TURN=24000 VLLM_KW='{"moe_backend": "marlin", "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"}}' \
  venv-kernel/bin/python kernel/composite/e2e_vllm.py forth-lightning-nvfp4 runs/inw_demo10.json all \
  $PWD/adapters/rpn-gpt-lightning
```

Measured on demo10: 7 of 10 correct, and all 10 turns exact at every kernel position (`"exact": 10`). The 10 turns held 279,832 tokens, hidden kernel tokens included (one lcg turn: 74,663 hidden tokens). Wall time 896 s after a 1 min start, 312 tokens/s in total for 10 streams. The 3 misses: one turn hit the 24,000-token cap, one answered without the machine, and one got a wrong answer after the forced end of its thinking. The kernel was exact in all three.

Measured on the same 16 dev problems as the server test below: 15 of 16 correct, 16 of 16 exact, 83,762 tokens in 293 s after the start. For an example turn from this run, see [`example_r7_in_weights.md`](https://huggingface.co/markab/rpn-gpt-lightning/blob/main/example_r7_in_weights.md) on the Hub.

### vLLM, OpenAI server

```sh
CUDA_VISIBLE_DEVICES=0 WORK=$PWD bash kernel/composite/serve_forth.sh $PWD/adapters/rpn-gpt-lightning
```

The server listens on `http://localhost:8100/v1`. Send chat requests with `"model": "forth"` and the card as the system message. The reply holds the Forth blocks and the output blocks that the kernel wrote. The hidden kernel tokens decode to empty strings.

The server was ready 3.5 min after the start. Measured on 16 dev problems (2 each from 8 big-arithmetic families), sent at once: 16 of 16 correct in 2 min 32 s. 7 turns used the machine. In each of them, every output block that the kernel wrote was equal to the output of the spec for the Forth blocks of that turn. The server reply has no hidden tokens, so this check works per block, not per token. For the per-token check, use `e2e_vllm.py`.

The server has no thinking budget. Send `max_tokens` (we used 24000). The flags in `serve_forth.sh` are required: prefix caching off, async scheduling off, and `FULL_DECODE_ONLY` CUDA graphs. Prompts must have 2 or more tokens.

## Path 4: a Mac with MLX NVFP4

NVIDIA's NVFP4 checkpoint does not load in mlx-lm. MLX 0.32 has its own `nvfp4` quantization mode (4-bit E2M1 values, one FP8 scale per 16 values). The script quantizes the BF16 checkpoint to it, converts the PEFT adapter to an mlx-lm adapter ([`training/peft_to_mlx.py`](../training/peft_to_mlx.py)), and serves both. The machine is the external one: the spec runs on the CPU.

You need Apple silicon with 48 GB or more and about 80 GB of disk (the BF16 download, once).

```sh
git clone https://github.com/markab21/rpn-gpt && cd rpn-gpt && uv sync
uvx hf download markab/rpn-gpt-lightning-LoRA --local-dir adapters/rpn-gpt-lightning
caffeinate -i bash training/mlx_local.sh $PWD/adapters/rpn-gpt-lightning     # leave it running
```

In a second terminal, from the repository root:

```sh
caffeinate -i env LOOP_SERVER=mlx VLLM_API_KEY=local LOOP_MODEL=default_model \
  LOOP_TOKENIZER=$HOME/.cache/rpn-gpt-mlx/lightning-nvfp4-mlx LIGHTNING_URL=http://localhost:8090/v1/completions \
  uv run python training/loop_sim.py training/data/demo10.json training/data/cards/reference_v4.txt \
  --think --think-budget 12000 --max-tokens 24000 --samples 1 --workers 1 --out runs/mac.json
```

Measured with the BF16 checkpoint already in the HF cache:

- Venv, quantization and adapter conversion: 22 s. The quantization alone took 28 s in a separate run, with 19 GB of memory at peak. The NVFP4 model is 17 GB at 4.5 bits per weight.
- The converted adapter matches the PEFT delta `(alpha / r) B A x` to a relative error of about 1e-7 on the tested modules.
- Three problems from demo10 (6, 3 and 9), one at a time: 3 of 3 correct in 4 min 17 s, 13,104 tokens, about 50 tokens/s. The server process used 14 GB (RSS) after the run.

A real Mac turn (problem 6), after the thinking:

````text
```forth
82225043186179281 6469193 mod . cr
```
```output
1290731 

```
The remainder is 1290731. \boxed{1290731}
````

Keep Mac runs to a few problems. A long-thinking problem runs for minutes.

## Path 5: packaged formats

These are the Hugging Face repositories with the adapter already merged. No LoRA flags are needed. Every format uses the same external machine, [`rpn_chat.py`](../rpn_chat.py) for one question or `training/loop_sim.py` for a set. The only exception is `-Forth`, which has the kernel in the weights.

Tested on the same RTX PRO 6000 machine, one GPU per server, on 2026-11-06. The same card, think 12000 / turn 24000, temperature 1.0 and top-p 0.95 were used for every run. Versions: vLLM 0.29.0 and transformers 5.16.1 (as above), llama.cpp `6c01da627` (2026-11-06, built with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120`), Ollama 0.40.15 (Linux tarball), mlx 0.32.4 / mlx-lm 0.32.2 (conversion on Linux).

| Repository | What it is | Result | Time |
|---|---|---|---|
| `markab/rpn-gpt-lightning` | BF16, adapter merged ([`training/merge_lora.py`](../training/merge_lora.py)), 62 GB | Teacher-forced top-1 agreement with BF16 + LoRA: 99.5% of 2025 tokens (stock: 89.8%). dev_round[:100] x2: 93.0% vs BF16 + LoRA 89.5%, +3.5 [-1.5, +8.5] | merge 13 s (CPU) |
| `markab/rpn-gpt-lightning-NVFP4` | NVIDIA's NVFP4 checkpoint, with the 93 adapter-target linears as merged BF16 ([`training/nvfp4_merge.py`](../training/nvfp4_merge.py)), 22 GB | frozen x1 90.5% (median 806 tokens): -1.2 [-3.7, +1.3] vs BF16 + LoRA, +3.0 [+0.2, +5.8] vs NVFP4 + LoRA | 600 problems in about 30 min |
| `markab/rpn-gpt-lightning-GGUF` | Q4_K_M 24.5 GB, Q5_K_M 26.0 GB, Q8_0 33.6 GB | frozen x1: Q4_K_M 91.2%, Q8_0 93.5%. Q5_K_M not measured | about 2 h each (llama-server, 64 slots, about 9 tokens/s per slot) |
| Ollama (the same GGUF) | `ollama pull hf.co/markab/rpn-gpt-lightning-GGUF:Q4_K_M` | `rpn_chat.py --server ollama`: 3 of 3 correct, output blocks exact. Plain chat stops after the program (`params` stop) | 7 to 17 s a question |
| `markab/rpn-gpt-lightning-MLX-4bit`, `-MLX-8bit` | `mlx_lm.convert -q --q-bits 4` / `8`, group 64, 17 GB / 32 GB | M4 Max 128 GB, 4-bit: 3 of 3 demo10 problems correct through `rpn_chat.py --server mlx` (mlx-lm 0.32.2) | 240 s, about 62 tokens/s. 8-bit: converted, not run |
| `markab/rpn-gpt-lightning-Forth` | the composite: kernel (7.4 GB) + modeling code, `base_model` = the merged repo | HF transformers: 3 of 3 correct, 1431 of 1431 kernel tokens exact. vLLM OpenAI server with the BF16 base: 6 of 6 correct with no client | 132 s (HF, 3 problems); server start about 3 min |

### One question, any server

```sh
uv run python rpn_chat.py "Calculate the remainder when 82225043186179281 is divided by 6469193." \
  --base-url http://localhost:8000 --model rpn                       # vLLM (a trailing /v1 is fine)
#   --server llamacpp --base-url http://localhost:8080               # llama-server
#   --server ollama --base-url http://localhost:11434 --model hf.co/markab/rpn-gpt-lightning-GGUF:Q4_K_M
#   --server mlx --base-url http://localhost:8090 --model default_model
```

It prints the turn, with each ` ```output ` block that the spec wrote, then `--- N Forth block(s) run, T tokens, answer: A`. On vLLM and llama-server (Q4_K_M), the mod question above came back correct in about 5 s. On Ollama (Q4_K_M), a two-part question (a product, then the product mod 1000003) ran 2 blocks, and both outputs were exact. Set `VLLM_API_KEY` if the server has a key.

### vLLM: merged BF16 and NVFP4

```sh
vllm serve markab/rpn-gpt-lightning --served-model-name rpn --max-model-len 65536
vllm serve markab/rpn-gpt-lightning-NVFP4 --served-model-name rpn --max-model-len 65536 --kv-cache-dtype bfloat16
```

The NVFP4 checkpoint loads as `modelopt_mixed`, with the MoE weights as W4A16 through Marlin on sm_120. NVIDIA's own scales did not come out of a plain re-quantization (my recipe matched only the BF16 tensors). NVFP4 rounding error on the adapter targets is about 9%, so those linears stay BF16 (+1.3 GB). The merged NVFP4 does better than NVIDIA NVFP4 + LoRA at runtime (+3.0 points, paired).

### llama.cpp and Ollama

```sh
llama-server -m rpn-gpt-lightning-Q4_K_M.gguf -ngl 99 -c 131072 -np 4 --port 8080
OLLAMA_HOST=127.0.0.1:11434 ollama serve &
ollama pull hf.co/markab/rpn-gpt-lightning-GGUF:Q4_K_M   # or: ollama create rpn -f Modelfile (FROM ./rpn-gpt-lightning-Q4_K_M.gguf)
```

Ollama 0.40.15 detects the architecture (`nemotron_h_moe`). After `ollama create` from the file, it uses its built-in `nemotron-3-nano` renderer. After `ollama pull hf.co/...`, it uses the chat template inside the GGUF (`template source gguf_chat_template`). Both keep thinking on, so the repository has no `template` file. Its `system` file is the card. Its `params` sets temperature 1.0, top-p 0.95, top-k 0, min-p 0, context 32768, at most 16000 new tokens, and stop ` ```output `. With these, `/api/chat` returns the thinking in its own field and ends the reply with the program. `rpn_chat.py --server ollama` uses `/api/generate` in raw mode with its own prompt. It streams the reply and cuts it at each fence, because Ollama does not say which stop string fired. Tested on the published repository: `ollama pull hf.co/markab/rpn-gpt-lightning-GGUF:Q4_K_M` into an empty `OLLAMA_MODELS` applied the card and every `params` value. The box downloaded the 24.5 GB blob at about 1 MB/s, so we placed our local file in the store first: it has the same SHA-256, and Ollama verified it. Then `rpn_chat.py` went 3 of 3 correct (7 to 17 s each), and a plain `/api/chat` call ended after the program with the thinking in its own field.

### The kernel in the weights from the Hub

```sh
FORTH_VISIBLE=1 VLLM_DISABLE_COMPILE_CACHE=1 PYTHONPATH=$PWD/kernel/composite \
  vllm serve markab/rpn-gpt-lightning-Forth --served-model-name forth --max-model-len 65536 --host 127.0.0.1 \
  --no-enable-flashinfer-autotune --no-enable-prefix-caching --no-async-scheduling -cc '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

This is the composite of Path 3 with the merged model as its base, so it needs no adapter and no NVFP4 copy. `vllm_forth.py` downloads `config.base_model` when it is a Hub id. Tested from the Hub with an empty `HF_HOME`: the server downloaded `markab/rpn-gpt-lightning-Forth` and its base `markab/rpn-gpt-lightning`, then started in about 3 minutes. 4 of 4 chat answers were correct, and 3 of them used the machine. All 6 output blocks that the kernel wrote were equal to the spec (one error message, then the fixed program).

Without `--host`, vLLM listens on every interface on port 8000, and with no `VLLM_API_KEY` anyone who can reach the port can use the server. A second test on another machine of the same type, from an empty cache: the download took about 12 minutes (69 GB), and the server was ready 7 minutes after the start (314 s of it compiling, because `VLLM_DISABLE_COMPILE_CACHE=1` compiles at every start).

To watch one request, run [`demos/live.py`](../demos/live.py) on the same machine. It streams the turn and marks the blocks that the kernel wrote:

```sh
venv-kernel/bin/python demos/live.py "Calculate the remainder when 82225043186179281 is divided by 6469193." --gpu 0
```

The time of a turn changes from sample to sample, because the server has no thinking budget. On this question, the turn took between 5 s and 95 s in our runs (one GPU, one request at a time, about 122 tokens/s, hidden kernel tokens included). A loop costs about 346 hidden tokens per pass for a 6-digit multiply and mod: 15 passes took 5,194 hidden tokens, about 42 s of the 45 s turn.

### Plain chat, no machine

Use the card as the system prompt, `stop: ["```output"]` and a `max_tokens` limit. Without the loop's thinking budget, a hard question can make the model compute by hand for a long time. "What is 2^64 mod 1000007?" ran into a 16000-token limit in 3 of 4 chat samples. Through `loop_sim.py` (budget 12000), the same question went 4 of 4 correct.

## Problems we hit, and the fixes

| Problem | Fix |
|---|---|
| `loop_sim.py` skipped every problem: `apply_chat_template requires jinja2` | `jinja2` is now a dependency in `pyproject.toml`. Run `uv sync` again. |
| `e2e_demo.py` imported `training/eval_sp.py`, which is not in this repository | `attach_lora` is now in `e2e_demo.py`. |
| The composite server wrote garbage after the first hidden kernel run (for example " stabilizing", then the end of the turn) | vLLM 0.29 serves with Model Runner V2 unless a request needs custom logits processors. The visible-stream patch works only on the V1 runner. `vllm_forth.py` now sets `VLLM_USE_V2_MODEL_RUNNER=0` when `FORTH_VISIBLE=1`. |
| One shard of the BF16 download stopped at 2.4 GB for minutes | Stop `hf download` and run it again. It resumes and checks each file. |
| vLLM warns "Your GPU does not have native support for FP4 computation" | Expected on sm_120 with this checkpoint: the MoE weights are W4A16. |
| NVIDIA's NVFP4 checkpoint does not load in mlx-lm | Quantize the BF16 checkpoint with `mlx_lm.convert --q-mode nvfp4` (done by `training/mlx_local.sh`). |
| `kernel/composite/e2e.py` read `training/data/demo_pool.json`, which is not in this repository | It now picks its big-number problems from `demo_set.json`. |
| `vllm_forth.py` read `config.base_model` as a local folder, so a composite that names a Hub repository did not load | It now downloads a Hub id with `snapshot_download`. |
| `mlx_lm.server` stops generating after an exact prompt-cache hit | `training/mlx_local.sh` patches one line of `mlx_lm/models/cache.py`. A repeated prompt is then prefilled again. |

## Not tested

- GPUs other than the RTX PRO 6000 (sm_120). On Hopper or newer data-center GPUs, vLLM can pick native FP4 kernels for the NVFP4 base. We did not measure them.
- `e2e_vllm.py` and `serve_forth.sh` with a BF16 base. We served the `-Forth` composite (merged BF16 base) with the same flags in Path 5.
- SGLang. The GGUF Q5_K_M file. The MLX 8-bit model.
- A Mac with less than 128 GB.
