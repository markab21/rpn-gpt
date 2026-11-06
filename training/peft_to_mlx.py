"""Convert a PEFT LoRA adapter for Lightning (the s2a_r7 release) into an mlx-lm adapter directory, for
mlx_lm.server --adapter-path / mlx_lm.generate --adapter-path on a Mac. Run it with the python of the mlx venv:
  python training/peft_to_mlx.py PEFT_DIR MLX_ADAPTER_DIR
PEFT computes y + (alpha / r) * B(A x); mlx-lm computes y + scale * (x @ lora_a) @ lora_b. So lora_a = A^T, lora_b = B^T,
scale = alpha / r. The release targets only dense linears (Mamba in_proj, shared experts, attention q/k/v/o)."""

import json
import re
import sys
from pathlib import Path

import mlx.core as mx

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
cfg = json.loads((src / "adapter_config.json").read_text())
assert cfg["peft_type"] == "LORA" and not cfg.get("use_dora") and cfg.get("bias", "none") == "none", cfg
out, keys = {}, set()
for k, v in mx.load(str(src / "adapter_model.safetensors")).items():
    m = re.fullmatch(r"base_model\.model\.model\.layers\.(\d+)\.(.+)\.lora_([AB])\.weight", k)
    assert m, f"unexpected key {k}"
    layer, mod, ab = m.groups()
    keys.add(mod)
    out[f"backbone.layers.{layer}.{mod}.lora_{ab.lower()}"] = v.T
n_layers = 1 + max(int(re.search(r"layers\.(\d+)\.", k).group(1)) for k in out)
dst.mkdir(parents=True, exist_ok=True)
mx.save_safetensors(str(dst / "adapters.safetensors"), out)
(dst / "adapter_config.json").write_text(json.dumps({
    "fine_tune_type": "lora", "num_layers": n_layers,
    "lora_parameters": {"rank": cfg["r"], "scale": cfg["lora_alpha"] / cfg["r"], "dropout": 0.0, "keys": sorted(keys)},
}, indent=1))
print(f"{len(out)} tensors, {n_layers} layers, modules {sorted(keys)}, scale {cfg['lora_alpha'] / cfg['r']}")
