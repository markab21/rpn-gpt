"""The base reads only the visible stream: for real text-mode streams (kernel/text_vm.stream), (a) every kernel-owned
position's argmax is text_vm's token, (b) at every model position the composite's logits equal a plain base run on the
visible-only sequence (hidden = kernel-emitted ids < HIDDEN_BELOW) at the matching position: non-word columns
(word_head replaces the word columns), argmax + max abs diff. Composite prefill and token-by-token cached decode, each
against base prefill and base cached decode on the visible sequence.
  CUDA_VISIBLE_DEVICES=1 python kernel/composite/test_skip.py CKPT     (EXPERTS=eager: base MoE eager, as test_composite)
"""

import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import text_vm as TV  # noqa: E402
from modeling_forth_lightning import HIDDEN_BELOW, ForthLightning  # noqa: E402

TEXT = ["Let me compute it.\n```forth\n3 4 + dup .\n```\nThat is 7. Doubling:\n```forth\n2 * .\n```\nDone, 14.",
        "Prose with 12 34 + . digits outside.\n```forth\n1 2 + .\n```\n```forth\n" + " ".join(["9"] * 60) + " "
        + " ".join(["+"] * 59) + " .\n```",
        "First a big one:\n```forth\n12345678901234567890 .\n```\nNow the stack.\n```forth\n1 2 swap . . 3 4 OVER . . .\n"
        "```\nAnd a sum:\n```forth\n" + " ".join(f"{10007 * i} ." for i in range(1, 21)) + "\n```\nAll done."]


def streams(tok):
    out = {f"text{i}": tok(s, add_special_tokens=False).input_ids for i, s in enumerate(TEXT)}
    chat = [{"role": "user", "content": "What is 3 + 4, doubled?"}, {"role": "assistant", "content": TEXT[0]}]
    out["chat0"] = tok(tok.apply_chat_template(chat, tokenize=False), add_special_tokens=False).input_ids
    return {k: TV.stream(v, tok)[:2] for k, v in out.items()}


def hidden_set(ids, owned, tok):
    """The predicate, checked against the tokenizer: kernel-emitted ids < HIDDEN_BELOW decode to '', the kernel's other
    tokens do not, and they are exactly what the kernel shows (text_vm.visible's decode; its single-block assert aside)."""
    hid = [o for o in owned if ids[o] < HIDDEN_BELOW]
    assert all(tok.decode([ids[o]], skip_special_tokens=True) == "" for o in hid)
    shown = [ids[o] for o in owned if ids[o] >= HIDDEN_BELOW]
    assert all(tok.decode([t], skip_special_tokens=True) != "" for t in shown)
    assert tok.decode(shown) == tok.decode([ids[o] for o in owned], skip_special_tokens=True)
    assert tok.decode(shown).count("```output\n") == tok.decode(ids).count("```forth")  # one block per fence
    return set(hid)


@torch.no_grad()
def run(m, f, ids):
    """Logits at every position (float, on CPU per row to keep memory flat): prefill or token by token (cached)."""
    x = torch.tensor([ids], device="cuda")
    if f == "prefill":
        return m(x).logits[0].float().cpu()
    out, lg = None, []
    for t in range(len(ids)):
        out = m(x[:, t:t + 1], past_key_values=out and out.past_key_values, use_cache=True)
        lg.append(out.logits[0, -1].float().cpu())
    return torch.stack(lg)


def check(m, tok, ids, owned, keep):
    hid = hidden_set(ids, owned, tok)
    vis = [t for p, t in enumerate(ids) if p not in hid]
    assert not set(vis) & set(m.config.override_ids), "reference would need embed_override"
    own = set(owned)
    kern = [o - 1 for o in owned]
    model = [p for p in range(len(ids)) if p + 1 not in own]  # positions where the base's logits pass through
    j = torch.tensor([p not in hid for p in range(len(ids))]).cumsum(0) - 1  # position -> last visible index
    r = {"tokens": len(ids), "kernel_tokens": len(owned), "hidden": len(hid), "visible": len(vis),
         "model_positions": len(model)}
    ref = {f: run(m.base, f, vis) for f in ("prefill", "cached")}
    r["base_prefill_vs_cached_max_diff"] = float((ref["prefill"] - ref["cached"]).abs().max())
    tgt = torch.tensor([ids[p + 1] for p in kern])
    for f in ("prefill", "cached"):
        lg = run(m, f, ids)
        r[f"{f}_kernel_wrong"] = int((lg[kern].argmax(-1) != tgt).sum())
        c = lg[model][:, keep]
        for rf, b in ref.items():
            b = b[j[model]][:, keep]
            r[f"{f}_vs_base_{rf}"] = {"argmax_equal": int((c.argmax(-1) == b.argmax(-1)).sum()),
                                      "max_abs_diff": float((c - b).abs().max()),
                                      "bit_identical": bool(torch.equal(c, b))}
    return r


@torch.no_grad()
def batch(m, rows):
    """Two right-padded rows as one batch (prefill and cached) == each row alone, at its real positions."""
    n = max(map(len, rows))
    x = torch.tensor([r + [0] * (n - len(r)) for r in rows], device="cuda")
    out, cached = None, []
    for t in range(n):
        out = m(x[:, t:t + 1], past_key_values=out and out.past_key_values, use_cache=True)
        cached.append(out.logits[:, -1].float().cpu())
    got = {"prefill": m(x).logits.float().cpu(), "cached": torch.stack(cached, 1)}
    R = {}
    for f, lg in got.items():
        one = [run(m, f, r) for r in rows]
        R[f"{f}_max_abs_diff_vs_batch1"] = max(float((lg[b, :len(r)] - one[b]).abs().max()) for b, r in enumerate(rows))
    return R


if __name__ == "__main__":
    ckpt = sys.argv[1]
    ex = {"base_kwargs": {"experts_implementation": os.environ["EXPERTS"]}} if os.environ.get("EXPERTS") else {}
    m = ForthLightning.from_pretrained(ckpt, dtype=torch.bfloat16, device_map="cuda", **ex).eval()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(m.config.base_model)
    keep = torch.ones(m.config.vocab_size, dtype=torch.bool)
    keep[m.config.word_ids] = False
    R = {}
    for name, (ids, owned) in streams(tok).items():
        R[name] = check(m, tok, ids, owned, keep)
        print(name, json.dumps(R[name]), flush=True)
    R["batch2"] = batch(m, [streams(tok)[k][0] for k in ("text0", "chat0")])
    print("batch2", R["batch2"], flush=True)
    json.dump(R, open(os.path.join(ckpt, f"skip_{os.environ.get('EXPERTS', 'default')}.json"), "w"), indent=1)
