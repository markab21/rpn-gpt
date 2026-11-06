"""Merge a PEFT LoRA (AutoModel export) into a Lightning HF checkpoint on CPU: W += (alpha / r) * B @ A in fp32, saved
bf16. Every other file is hard-linked from the base; only shards holding a merged weight are rewritten.
  python training/merge_lora.py BASE_DIR ADAPTER_DIR OUT_DIR
Adapter keys 'base_model.model.model.layers.N.X.lora_A.weight' map to base 'backbone.layers.N.X.weight'.
"""
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

base, adapter, out = sys.argv[1:4]
cfg = json.load(open(os.path.join(adapter, "adapter_config.json")))
scale = cfg["lora_alpha"] / cfg["r"]
assert not cfg.get("use_dora") and not cfg.get("use_rslora"), cfg
A = {}
with safe_open(os.path.join(adapter, "adapter_model.safetensors"), "pt") as f:
    for k in f.keys():
        A[k] = f.get_tensor(k)
delta = {}
for k in A:
    if ".lora_A." in k:
        name = k.removeprefix("base_model.model.").replace(".lora_A.weight", ".weight").replace("model.", "backbone.", 1)
        delta[name] = (A[k.replace("lora_A", "lora_B")].float(), A[k].float())
idx = json.load(open(os.path.join(base, "model.safetensors.index.json")))["weight_map"]
missing = [n for n in delta if n not in idx]
assert not missing, missing[:5]
os.makedirs(out, exist_ok=True)
shards = {idx[n] for n in delta}
for fn in os.listdir(base):
    if os.path.isfile(os.path.join(base, fn)) and fn not in shards and not os.path.exists(os.path.join(out, fn)):
        os.link(os.path.join(base, fn), os.path.join(out, fn))
done = 0
for sh in sorted(shards):
    T = {}
    with safe_open(os.path.join(base, sh), "pt") as f:
        meta = f.metadata()
        for k in f.keys():
            T[k] = f.get_tensor(k)
    for n in [n for n in delta if idx[n] == sh]:
        B, Am = delta[n]
        assert T[n].shape == (B.shape[0], Am.shape[1]), (n, T[n].shape, B.shape, Am.shape)
        T[n] = (T[n].float() + scale * (B @ Am)).to(T[n].dtype)
        done += 1
    save_file(T, os.path.join(out, sh + ".tmp"), metadata=meta)
    os.replace(os.path.join(out, sh + ".tmp"), os.path.join(out, sh))
assert done == len(delta)
json.dump({"base": os.path.abspath(base), "adapter": os.path.abspath(adapter), "scale": scale, "merged": done},
          open(os.path.join(out, "merged_from.json"), "w"), indent=1)
print(f"merged {done} LoRA weights (scale {scale}) into {len(shards)} shards -> {out}")
