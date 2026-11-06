"""NVFP4 sampler vs BF16 trainer: the policy gap GRPO would see. For N rollouts (loop_sim output), every token the model
wrote (not the prompt, not the kernel's ```output blocks) is scored twice over the same id sequence:
  1) --vllm URL: the NVFP4 server's prompt_logprobs (raw distribution, temperature 1) -> OUT.json
  2) --hf:       BF16 Lightning + the LoRA in HF transformers (one GPU), compared to 1)
Reports mean |dlogp| per token, the fraction with |dlogp| > 1, and the importance ratios exp(logp_bf16 - logp_nvfp4)
(per token, and per sequence = exp of the summed difference).
  python training/logprob_gap.py ROLLOUTS.json OUT.json --n 1 --vllm http://localhost:8241/v1/completions --model s2ar1
  CUDA_VISIBLE_DEVICES=1 python training/logprob_gap.py ROLLOUTS.json OUT.json --hf --adapter DIR [--engine hf-fused|automodel]
  (--engine automodel: venv-automodel, the trainer's NemotronV3 with mamba_ssm kernels; scores land in OUT_ENGINE[_lora].json)"""
import argparse, json, math, os, re, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
F = "`" * 3
OUT_SPAN = re.compile(F + r"output\n.*?\n" + F + r"\n", re.S)


def sequence(r, tok, system):
    """(ids, mask): the chat prompt + the turn, tokenized whole (as loop_sim's requests); mask = 1 on model-written tokens."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": r["q"]}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    enc = tok(prompt + r["text"], add_special_tokens=False, return_offsets_mapping=True)
    spans = [(len(prompt) + m.start(), len(prompt) + m.end()) for m in OUT_SPAN.finditer(r["text"])]
    mask = [int(a >= len(prompt) and not any(s <= a < e for s, e in spans)) for a, _ in enc.offset_mapping]
    return enc.input_ids, mask


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("rollouts")
    a.add_argument("out")
    a.add_argument("--n", type=int, default=10)
    a.add_argument("--vllm")
    a.add_argument("--model", default="s2ar1")
    a.add_argument("--hf", action="store_true")
    a.add_argument("--adapter")
    a.add_argument("--engine", choices=["hf", "hf-fused", "automodel"], default="hf")
    a.add_argument("--base", default="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16")
    a.add_argument("--prompt", default=str(ROOT / "training/data/cards/reference.txt"))
    args = a.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base)
    system = open(args.prompt).read()
    if args.vllm:
        R = [r for r in json.load(open(args.rollouts)) if r["blocks"]][:args.n]  # machine-using turns: kernel spans in them
        res = []
        for r in R:
            ids, mask = sequence(r, tok, system)
            body = {"model": args.model, "prompt": ids, "max_tokens": 1, "temperature": 1.0, "prompt_logprobs": 0}
            req = urllib.request.Request(args.vllm, json.dumps(body).encode(), {"Content-Type": "application/json",
                                         "Authorization": f"Bearer {os.environ['VLLM_API_KEY']}"})
            pl = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["prompt_logprobs"]
            lp = [None] + [d[str(t)]["logprob"] for d, t in zip(pl[1:], ids[1:])]  # position k: log p(ids[k] | ids[:k])
            res.append({"q": r["q"], "ok": r["ok"], "ids": ids, "mask": mask, "nvfp4": lp})
            print(f"{sum(mask)} model tokens of {len(ids)} | {r['q'][:60]}", flush=True)
        json.dump(res, open(args.out, "w"))
        return
    import torch
    if args.engine == "automodel":  # the trainer's own model (configs/sft_text.yaml), LoRA merged into the weights
        m = automodel(args)
    else:
        from transformers import AutoModelForCausalLM
        if args.engine == "hf-fused":  # HF's torch fallback -> mamba_ssm / causal_conv1d kernels (as AutoModel and vLLM)
            import transformers.models.nemotron_h.modeling_nemotron_h as H
            from causal_conv1d import causal_conv1d_fn
            from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
            H.mamba2_chunk_scan = mamba_chunk_scan_combined
            H.causal_conv1d_fn = lambda x, weight, bias=None, activation=None, **kw: causal_conv1d_fn(x, weight, bias, activation=activation)
        m = AutoModelForCausalLM.from_pretrained(args.base, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda").eval()
        if args.adapter:
            from peft import PeftModel
            m = PeftModel.from_pretrained(m, args.adapter).eval()
    res = json.load(open(args.out))[:args.n]
    alld, ratios = [], []
    for x in res:
        ids = torch.tensor([x["ids"]], device="cuda")
        with torch.no_grad():
            o = m(ids)
            logits = (o if torch.is_tensor(o) else o.logits)[0, :-1].float()
        lp = torch.log_softmax(logits, -1).gather(1, ids[0, 1:, None])[:, 0].tolist()
        x["bf16"] = [None] + lp
        d = [x["bf16"][k] - x["nvfp4"][k] for k in range(1, len(x["ids"])) if x["mask"][k]]
        alld += d
        ratios.append(math.exp(sum(d)))
        if args.adapter and args.engine != "automodel" and x is res[0]:  # the adapter must matter: the same sequence with it switched off
            with torch.no_grad(), m.disable_adapter():
                lp0 = torch.log_softmax(m(ids).logits[0, :-1].float(), -1).gather(1, ids[0, 1:, None])[:, 0].tolist()
            d0 = [lp0[k - 1] - x["nvfp4"][k] for k in range(1, len(x["ids"])) if x["mask"][k]]
            print(f"  (adapter off: mean|d| {sum(map(abs, d0)) / len(d0):.4f}, sum d {sum(d0):+.2f})", flush=True)
        print(f"{len(d)} tokens: mean|d| {sum(map(abs, d)) / len(d):.4f}, |d|>1 {sum(abs(v) > 1 for v in d)}, "
              f"sum d {sum(d):+.2f}, seq ratio {ratios[-1]:.3g} | {x['q'][:50]}", flush=True)
    n = len(alld)
    print(f"ALL {len(res)} seqs, {n} tokens: mean |dlogp| {sum(map(abs, alld)) / n:.4f}, frac |d|>1 "
          f"{sum(abs(v) > 1 for v in alld) / n:.4%}, mean token ratio {sum(math.exp(v) for v in alld) / n:.4f}, "
          f"mean seq ratio {sum(ratios) / len(ratios):.4g} (median {sorted(ratios)[len(ratios) // 2]:.4g})")
    json.dump(res, open(args.out.replace(".json", f"_{args.engine}{'_lora' if args.adapter else ''}.json"), "w"))


def automodel(args):
    """NeMoAutoModelForCausalLM as configs/sft_text.yaml builds it (one GPU, no FSDP / EP), eval mode; the LoRA
    (scale alpha / r, dropout 0) merged into the base weights in fp32."""
    import torch
    import torch.distributed as dist
    from nemo_automodel import NeMoAutoModelForCausalLM
    from nemo_automodel.components.models.common.utils import BackendConfig
    if not dist.is_initialized():
        dist.init_process_group("nccl", init_method="tcp://localhost:29533", world_size=1, rank=0)
    m = NeMoAutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 num_nextn_predict_layers=0,
                                                 backend=BackendConfig(experts="torch_mm", dispatcher="torch")).cuda().eval()
    if args.adapter:
        from safetensors.torch import load_file
        cfg = json.load(open(f"{args.adapter}/adapter_config.json"))
        sd, mods = load_file(f"{args.adapter}/adapter_model.safetensors"), dict(m.named_modules())
        names = {k[len("base_model.model."):].rsplit(".lora_", 1)[0] for k in sd}
        assert names <= set(mods), sorted(names - set(mods))[:5]
        with torch.no_grad():
            for nm in sorted(names):
                A, B = (sd[f"base_model.model.{nm}.lora_{x}.weight"].float().cuda() for x in "AB")
                w = mods[nm].weight
                w.copy_((w.float() + cfg["lora_alpha"] / cfg["r"] * B @ A).to(w.dtype))
        print(f"merged {len(names)} LoRA modules", flush=True)
    return m


if __name__ == "__main__":
    main()
