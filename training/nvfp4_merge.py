"""Merged NVFP4: NVIDIA's NVFP4 Lightning checkpoint, unchanged, except the LoRA-target linears, which hold the merged
BF16 weights and are marked unquantized (removed from quantized_layers / config_groups, added to the ignore lists).
Requantizing them would round most of the small LoRA delta away (NVFP4 error on these weights is ~9%), so they stay BF16
(+~1.3 GB).
  python training/nvfp4_merge.py BF16_MERGED NVFP4_BASE ADAPTER_DIR OUT   (CPU, ~1 min; BF16_MERGED from merge_lora.py)
"""
import json
import os
import shutil
import sys

from safetensors import safe_open
from safetensors.torch import save_file

merged, q4, adapter, out = sys.argv[1:5]
mods = ["backbone." + m[len("model."):] for m in json.load(open(os.path.join(adapter, "adapter_config.json")))["target_modules"]]
q4idx = json.load(open(os.path.join(q4, "model.safetensors.index.json")))["weight_map"]
midx = json.load(open(os.path.join(merged, "model.safetensors.index.json")))["weight_map"]
drop = {k for k in q4idx for m in mods if k.startswith(m + ".")}  # weight + its scales
new = {}
for m in mods:
    with safe_open(os.path.join(merged, midx[m + ".weight"]), "pt") as f:
        new[m + ".weight"] = f.get_tensor(m + ".weight")
os.makedirs(out, exist_ok=True)
touched = {q4idx[k] for k in drop}
for fn in os.listdir(q4):
    if os.path.isfile(os.path.join(q4, fn)) and fn not in touched and fn not in ("config.json", "hf_quant_config.json",
                                                                                 "model.safetensors.index.json"):
        shutil.copy2(os.path.join(q4, fn), os.path.join(out, fn))
for sh in sorted(touched):
    with safe_open(os.path.join(q4, sh), "pt") as f:
        T, meta = {k: f.get_tensor(k) for k in f.keys() if k not in drop}, f.metadata()
    T.update({k: v for k, v in new.items() if q4idx[k] == sh})
    save_file(T, os.path.join(out, sh), metadata=meta)
idx = json.load(open(os.path.join(q4, "model.safetensors.index.json")))
idx["weight_map"] = {k: v for k, v in q4idx.items() if k not in drop} | {k: q4idx[k] for k in new}
json.dump(idx, open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=2)
cfg = json.load(open(os.path.join(q4, "config.json")))
hq = json.load(open(os.path.join(q4, "hf_quant_config.json")))
for q, ign in ((cfg["quantization_config"], "ignore"), (hq["quantization"], "exclude_modules")):
    q["quantized_layers"] = {k: v for k, v in q["quantized_layers"].items() if k not in mods}
    q[ign] = sorted(set(q[ign]) | set(mods))
for g in cfg["quantization_config"]["config_groups"].values():
    g["targets"] = [t for t in g["targets"] if t not in mods]
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
json.dump(hq, open(os.path.join(out, "hf_quant_config.json"), "w"), indent=4)
print(f"{len(new)} LoRA-target linears now merged BF16 ({len(drop)} quantized tensors dropped) in {len(touched)} shards -> {out}")
