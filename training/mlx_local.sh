#!/bin/bash
# Local loop tests on a Mac (Apple silicon, >= 64 GB): Lightning 8-bit (mlx-community, quantized from bf16; the official
# NVFP4 needs a runtime stock mlx-lm lacks) served by mlx_lm.server on localhost:8090, for
# training/loop_sim.py with the kernel simulated by kernel/text_vm.py (CPU). Builds the venv once (uv), downloads the
# pinned checkpoint (~34 GB, HF cache), patches mlx_lm's prompt cache (an exact cache hit leaves an empty prompt and
# kills the batched server's generation thread; with it off, a repeated prompt is just prefilled again), then serves.
#   training/mlx_local.sh [VENV_DIR]          (foreground; stop with ctrl-c)
# Then, from the repo root:
#   LOOP_SERVER=mlx VLLM_API_KEY=local LOOP_MODEL=default_model LOOP_TOKENIZER=$(cat VENV_DIR/model_dir) \
#   LIGHTNING_URL=http://localhost:8090/v1/completions PYTHONPATH=.. VENV_DIR/bin/python ../training/loop_sim.py ...
set -euo pipefail
VENV=${1:-$HOME/.cache/rpn-gpt-mlx}
REPO=mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-8bit
REV=881a85f9e8
[ -x "$VENV/bin/mlx_lm.server" ] || { uv venv -q -p 3.12 "$VENV"; VIRTUAL_ENV=$VENV uv pip install -q transformers tokenizers jinja2 mlx-lm huggingface_hub; }
CACHE=$("$VENV/bin/python" -c "import mlx_lm.models.cache as c; print(c.__file__)")
grep -q "if False and result.exact is not None:" "$CACHE" || {
  grep -q "        if result.exact is not None:" "$CACHE" || { echo "mlx_lm changed: re-check the prompt-cache patch" >&2; exit 1; }
  sed -i '' 's/        if result.exact is not None:/        if False and result.exact is not None:/' "$CACHE"; }
MODEL_DIR=$("$VENV/bin/hf" download "$REPO" --revision "$REV" --quiet)
echo "$MODEL_DIR" > "$VENV/model_dir"
exec "$VENV/bin/mlx_lm.server" --model "$MODEL_DIR" --host localhost --port 8090 --decode-concurrency 2 --prompt-cache-size 4  # (a long session's cache exhausts Metal's buffer limit)
