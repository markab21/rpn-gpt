"""Copy a PEFT LoRA adapter with every lora_B zeroed: the adapter code path with exactly no change to the model.
  python training/zero_adapter.py SRC_DIR DST_DIR   (serve DST next to the real adapter: the engine-matched control)"""
import os, shutil, sys
import torch
from safetensors.torch import load_file, save_file
s, d = sys.argv[1:3]
os.makedirs(d, exist_ok=True)
shutil.copy(os.path.join(s, "adapter_config.json"), d)
T = load_file(os.path.join(s, "adapter_model.safetensors"))
Z = {k: torch.zeros_like(v) if "lora_B" in k else v for k, v in T.items()}
save_file(Z, os.path.join(d, "adapter_model.safetensors"))
B = [v for k, v in Z.items() if "lora_B" in k]
assert B and all(float(v.abs().max()) == 0 for v in B)
print(len(Z), "tensors,", len(B), "lora_B zeroed")
