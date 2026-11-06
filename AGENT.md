# AGENT.md: run and test rpn-gpt

You are an agent. This file tells you how to install, run and test this model. Follow the steps in order. Every command and number here comes from a tested run on 2026-11-06. The full guide is [`playbooks/deploy.md`](https://github.com/markab21/rpn-gpt/blob/main/playbooks/deploy.md) in the code repository.

## What the model is

- `markab/rpn-gpt-lightning` is a LoRA adapter (rank 16, 23 MB) for NVIDIA Nemotron 3.5 Lightning (30B parameters, 3B active).
- The adapter teaches the model to write Forth in a ` ```forth ` block when a problem needs exact computation. A machine runs the block and writes a ` ```output ` block. The model reads the output and answers in `\boxed{}`.
- The machine is not in this repository. You use one of two machines:
  - The external machine: [`training/loop_sim.py`](https://github.com/markab21/rpn-gpt/blob/main/training/loop_sim.py) in [github.com/markab21/rpn-gpt](https://github.com/markab21/rpn-gpt). It calls a model server and runs each block in a Python spec.
  - The kernel in the weights (the composite): a second network that you build from the base weights. It writes the output blocks itself, inside vLLM.
- The model needs its system prompt, the card [`reference_v4.txt`](training/data/cards/reference_v4.txt). Use temperature 1.0 and top-p 0.95. Do not use greedy decoding.
- Evaluation data: [`markab/rpn-gpt-data`](https://huggingface.co/datasets/markab/rpn-gpt-data). The same sets are in `training/data/` of the code repository.

## Choose the hardware

| Hardware | Path | Expected score, frozen test x1 |
|---|---|---|
| 1 NVIDIA GPU, 80 GB or more | BF16 base + LoRA in vLLM (use this to measure) | 89 to 95% (we measured 91.7%) |
| 1 NVIDIA GPU, 40 GB or more | NVFP4 base + LoRA in vLLM | 85 to 91% (we measured 87.5%) |
| Mac, Apple silicon, 48 GB or more | MLX NVFP4 + LoRA | not measured at scale (3 of 3 on a smoke test) |

We tested on an RTX PRO 6000 Blackwell (96 GB, sm_120), driver 615, Ubuntu 24.04, and on an M4 Max with 128 GB. Smaller GPUs and smaller Macs are untested.

## Set up an NVIDIA GPU machine

You need Linux x86_64, NVIDIA driver 580 or newer, Python 3.12, 100 GB of free disk, and [uv](https://docs.astral.sh/uv/). All packages are wheels. Do not build anything from source.

```sh
git clone https://github.com/markab21/rpn-gpt && cd rpn-gpt
uv venv -p 3.12 venv-kernel
VIRTUAL_ENV=$PWD/venv-kernel uv pip install -r requirements/vllm-cu130.txt   # vllm 0.29.0, torch 2.13.0+cu130, transformers 5.16.1, peft 0.21.2
venv-kernel/bin/hf download markab/rpn-gpt-lightning --local-dir adapters/rpn-gpt-lightning
venv-kernel/bin/hf download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
  --revision a9904d24bcc1d289a1950fa9d2b978c47cf903b9 --local-dir lightning-bf16
export VLLM_API_KEY=$(openssl rand -hex 16)
```

For the NVFP4 base, download `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` at revision `bee7596271d1495f6992ae224aefde4410e816b8` into `lightning-nvfp4`.

Rules:

- Keep vLLM at 0.29.0. The kernel in the weights is exact on 0.29.0 only.
- If the machine is an NGC PyTorch container, run the install with `env -u PIP_CONSTRAINT`. Do not use the Python packages of the image.

## Start the server

Run this in the background on one GPU. Wait until `curl -s -H "Authorization: Bearer $VLLM_API_KEY" http://127.0.0.1:8200/v1/models` lists `s2a_r7`. The first start takes about 4 minutes.

```sh
CUDA_VISIBLE_DEVICES=0 venv-kernel/bin/vllm serve lightning-bf16 --served-model-name lightning \
  --host 127.0.0.1 --port 8200 --max-model-len 65536 --gpu-memory-utilization 0.92 --max-num-seqs 128 \
  --no-enable-flashinfer-autotune --enable-lora --max-lora-rank 16 \
  --lora-modules s2a_r7=$PWD/adapters/rpn-gpt-lightning
```

For NVFP4, serve `lightning-nvfp4` instead and add `--kv-cache-dtype bfloat16 --enable-prefix-caching --mamba-cache-mode align --async-scheduling`.

## Smoke test (5 minutes)

Run one problem through the external machine:

```sh
python3 -c "import json; json.dump(json.load(open('training/data/demo10.json'))[6:7], open('one.json', 'w'))"
LOOP_MODEL=s2a_r7 LIGHTNING_URL=http://127.0.0.1:8200/v1/completions LOOP_TOKENIZER=$PWD/lightning-bf16 \
  venv-kernel/bin/python training/loop_sim.py one.json training/data/cards/reference_v4.txt \
  --think --think-budget 12000 --max-tokens 24000 --samples 1 --out runs/one.json
```

Expected output (about 30 seconds):

```text
OK   1 block(s) | Calculate the remainder when 82225043186179281 is divided by 6469193. | want 1290731 got 1290731
     [0] 82225043186179281 6469193 mod . cr  =>  1290731
correct 1 / 1
```

The model can send a block that fails first, read the error, and send a fixed block. Then you see 2 blocks. That is normal. Sampling is random, so one miss is possible. If it misses, run the command again once.

## Quick eval

1. Run demo10 (10 hard problems, 4 samples each, about 5 minutes on one GPU):

   ```sh
   LOOP_MODEL=s2a_r7 LIGHTNING_URL=http://127.0.0.1:8200/v1/completions LOOP_TOKENIZER=$PWD/lightning-bf16 \
     venv-kernel/bin/python training/loop_sim.py training/data/demo10.json training/data/cards/reference_v4.txt \
     --think --think-budget 12000 --max-tokens 24000 --samples 4 --workers 40 --out runs/demo10.json
   ```

   Expected: 33 to 40 of 40. We measured 37 of 40 on BF16 and 34 of 40 on NVFP4.

2. Run the frozen test (600 problems, about 25 minutes on one GPU): use the same command with `training/data/frozen_test.json`, `--samples 1` and `--workers 100`. Expected: 89 to 95% on BF16, 85 to 91% on NVFP4.

3. Report the `correct N / M` line and the file `runs/*.json`.

If a run stops, run the same command again. It resumes from `runs/*.json.partial.jsonl`.

To compare with stock Lightning, serve a zero adapter on the same engine path. Do not use the base model name. Make it with `venv-kernel/bin/python training/zero_adapter.py adapters/rpn-gpt-lightning adapters/zero`. Then serve it as `--lora-modules stock=$PWD/adapters/zero` and run with `LOOP_MODEL=stock`. Expected stock: 76 to 83% on the frozen test (we measured 79.5%, and +12.2 points for the adapter, paired).

## Run the kernel in the weights

Do this on a GPU machine after the setup above. You also need the NVFP4 base in `lightning-nvfp4`.

1. Build the composite. It takes about 1 minute on the CPU and writes 7.4 GB into `forth-lightning/`.

   ```sh
   SP_LIGHTNING=$PWD/lightning-bf16 venv-kernel/bin/python kernel/composite/build.py $PWD/forth-lightning $PWD/lightning-bf16
   ```

2. Start the composite server on a free GPU. It makes `forth-lightning-nvfp4/` on its first start, and it is ready after about 3.5 minutes.

   ```sh
   CUDA_VISIBLE_DEVICES=1 WORK=$PWD bash kernel/composite/serve_forth.sh $PWD/adapters/rpn-gpt-lightning
   ```

3. Send a chat request to `http://localhost:8100/v1/chat/completions` with `"model": "forth"`, the card as the system message, `"max_tokens": 24000`, `"temperature": 1.0` and `"top_p": 0.95`. There is no external loop. The reply holds the ` ```forth ` block, the ` ```output ` block that the kernel wrote, and the boxed answer.

4. For a batch with a check of every kernel position against the spec, stop the server and run `kernel/composite/e2e_vllm.py` as shown in `playbooks/deploy.md`. Expected on demo10: 7 of 10 correct, 10 of 10 exact.

## Run on a Mac

Use Apple silicon with 48 GB or more and 80 GB of free disk. The script downloads the BF16 base (60 GB, once), quantizes it to MLX NVFP4 (about 30 seconds, 17 GB), converts the adapter, and serves it on port 8090.

```sh
git clone https://github.com/markab21/rpn-gpt && cd rpn-gpt && uv sync
uvx hf download markab/rpn-gpt-lightning --local-dir adapters/rpn-gpt-lightning
caffeinate -i bash training/mlx_local.sh $PWD/adapters/rpn-gpt-lightning
```

In a second terminal, run 1 to 3 problems only:

```sh
python3 -c "import json; json.dump(json.load(open('training/data/demo10.json'))[6:7], open('one.json', 'w'))"
caffeinate -i env LOOP_SERVER=mlx VLLM_API_KEY=local LOOP_MODEL=default_model \
  LOOP_TOKENIZER=$HOME/.cache/rpn-gpt-mlx/lightning-nvfp4-mlx LIGHTNING_URL=http://localhost:8090/v1/completions \
  uv run python training/loop_sim.py one.json training/data/cards/reference_v4.txt \
  --think --think-budget 12000 --max-tokens 24000 --samples 1 --workers 1 --out runs/mac.json
```

We measured 3 of 3 correct on problems 6, 3 and 9 of demo10 in 4 minutes 17 seconds, about 50 tokens/s.

## Common failures

| Symptom | Fix |
|---|---|
| `loop_sim.py` prints `SKIPPED (ImportError('apply_chat_template requires jinja2 ...'))` | Run `uv sync` again in an up-to-date clone, or `uv pip install jinja2` in the venv that runs `loop_sim.py`. |
| `hf download` stops on one file for minutes | Stop it and run the same command again. It resumes. |
| vLLM warns "Your GPU does not have native support for FP4 computation" | This is expected for the NVFP4 base on sm_120 GPUs. The MoE weights run as W4A16 (Marlin). |
| The composite server answers with garbage after the first Forth word (for example " stabilizing", then the end of the turn) | Your `kernel/composite/vllm_forth.py` is old. Pull the repository. The fix forces the V1 model runner of vLLM. |
| The composite server rejects a prompt of 1 token | Expected. Send a real chat prompt. |
| `vllm serve` fails with "Weight files referenced in index but missing" | The download is not complete. Run `hf download` again. |
| The score is far below the range | Make sure that you use the card `reference_v4.txt`, `--think --think-budget 12000 --max-tokens 24000`, temperature 1.0 and top-p 0.95, and `LOOP_MODEL` set to the adapter name, not the base name. |
