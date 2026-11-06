#!/bin/bash
# Local loop tests on a Mac (Apple silicon, >= 48 GB): Lightning quantized to MLX NVFP4 (mlx's own nvfp4 mode, 4.5 bits
# per weight, ~17 GB) + the rpn-gpt adapter as an mlx-lm LoRA, served by mlx_lm.server on localhost:8090, for
# training/loop_sim.py with the machine simulated by kernel/text_vm.py (CPU). NVIDIA's NVFP4 checkpoint does not load in
# mlx-lm, so this quantizes the BF16 checkpoint itself (~30 s; the BF16 download is ~60 GB, once).
# Builds the venv once (uv, wheels only), patches mlx_lm's prompt cache (an exact cache hit leaves an empty prompt and
# kills the batched server's generation thread; with it off, a repeated prompt is just prefilled again), then serves.
#   training/mlx_local.sh PEFT_ADAPTER_DIR [VENV_DIR]      (foreground; stop with ctrl-c)
# Then, from the repo root:
#   LOOP_SERVER=mlx VLLM_API_KEY=local LOOP_MODEL=default_model LOOP_TOKENIZER=VENV_DIR/lightning-nvfp4-mlx \
#   LIGHTNING_URL=http://localhost:8090/v1/completions uv run python training/loop_sim.py ...
set -euo pipefail
ADAPTER=${1:?PEFT adapter dir (adapter_config.json + adapter_model.safetensors)}
VENV=${2:-$HOME/.cache/rpn-gpt-mlx}
REPO=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
REV=a9904d24bcc1d289a1950fa9d2b978c47cf903b9
HERE=$(cd "$(dirname "$0")" && pwd)
[ -x "$VENV/bin/mlx_lm.server" ] || { uv venv -q -p 3.12 "$VENV"
  VIRTUAL_ENV=$VENV uv pip install -q --only-binary :all: mlx==0.32.4 mlx-lm==0.32.2 transformers==5.22.2 huggingface_hub==2.4.0; }
CACHE=$("$VENV/bin/python" -c "import mlx_lm.models.cache as c; print(c.__file__)")
grep -q "if False and result.exact is not None:" "$CACHE" || {
  grep -q "        if result.exact is not None:" "$CACHE" || { echo "mlx_lm changed: re-check the prompt-cache patch" >&2; exit 1; }
  sed -i '' 's/        if result.exact is not None:/        if False and result.exact is not None:/' "$CACHE"; }
MODEL=$VENV/lightning-nvfp4-mlx
[ -f "$MODEL/config.json" ] || "$VENV/bin/mlx_lm.convert" -q --q-mode nvfp4 --mlx-path "$MODEL" \
  --hf-path "$("$VENV/bin/hf" download "$REPO" --revision "$REV" --quiet)"
LORA=$VENV/$(basename "$ADAPTER")-mlx
[ "$LORA/adapters.safetensors" -nt "$ADAPTER/adapter_model.safetensors" ] || "$VENV/bin/python" "$HERE/peft_to_mlx.py" "$ADAPTER" "$LORA"
exec "$VENV/bin/mlx_lm.server" --model "$MODEL" --adapter-path "$LORA" --host localhost --port 8090 \
  --decode-concurrency 2 --prompt-cache-size 4  # (a long session's cache exhausts Metal's buffer limit)
