#!/bin/bash
# OpenAI-compatible server for the composite (Lightning NVFP4 base + Forth kernel, plugin vllm_forth.py) with a LoRA on
# the base, vLLM 0.29 (venv-kernel), http://localhost:8100/v1, auth: Bearer $VLLM_API_KEY (vllm reads it from env).
#   CUDA_VISIBLE_DEVICES=1 [NEMO=/srv/nemo] bash kernel/composite/serve_forth.sh ADAPTER_DIR [extra vllm serve args]
# NEMO: the dir holding kernel/composite/ and forth-lightning/ (venv-kernel and the bases stay in /srv/nemo).
# Loop-style evals in the weights (batched, offline, kernel positions checked): kernel/composite/e2e_vllm.py (one command in
# its docstring). Requests: "model": "forth" = base + adapter, "model": "/srv/nemo/forth-lightning-nvfp4" = no adapter.
# The base reads only the visible stream (FORTH_VISIBLE=1, vllm_visible.py): it needs prefix caching off, async
# scheduling off and FULL_DECODE_ONLY cudagraphs (vllm_forth.py asserts it); bf16 KV; the SSM cache stays fp32.
# shortcut: the flags are the ones kernel/composite/vllm_suite.sh / vllm_speed.sh were verified with; the server itself
# has not been run with them yet.
set -e
ADAPTER=${1:?adapter dir (adapter_config.json + adapter_model.safetensors)}
shift
NEMO=${NEMO:-/srv/nemo}
cd $NEMO
CKPT=$NEMO/forth-lightning-nvfp4
if [ ! -f $CKPT/config.json ]; then  # forth-lightning with its base pointed at the NVFP4 checkpoint
  mkdir -p $CKPT
  for f in model.safetensors tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json; do
    ln -sf ../forth-lightning/$f $CKPT/
  done
  $NEMO/venv-kernel/bin/python -c "import json; c = json.load(open('forth-lightning/config.json'));
c['base_model'] = 'nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4'; json.dump(c, open('$CKPT/config.json', 'w'), indent=1)"
fi
# VLLM_DISABLE_COMPILE_CACHE: the two separately compiled backbones share one compile-cache dir ('backbone'); a cached
# artifact of one gets loaded for the other. shortcut: no cache (~1 min more compile per start), fix the cache prefix
FORTH_VISIBLE=1 VLLM_DISABLE_COMPILE_CACHE=1 PYTHONPATH=$NEMO/kernel/composite exec $NEMO/venv-kernel/bin/vllm serve $CKPT --host localhost --port 8100 \
  --max-model-len 262144 --max-num-seqs 256 --no-enable-flashinfer-autotune --moe-backend marlin \
  --no-enable-prefix-caching --no-async-scheduling -cc '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --enable-lora --max-lora-rank 64 --lora-modules forth=$ADAPTER "$@"
