"""Build the Lightning + Forth kernel composite checkpoint: config + kernel weights + base-side override rows + modeling
code. The base is referenced by path, never copied.   python kernel/composite/build.py OUT [BASE] [--keep=OLD_COMPOSITE]
(GPU box, CPU is enough). --kernel=DIR: a saved kernel (sp_kernel.save) instead of building one. --keep: override / word-head rows of ids OLD_COMPOSITE already has are copied from it (fitted
rows survive a kernel rebuild); only new ids get the init below. OLD_COMPOSITE may also be an adapter dir from
training/push.py (composite_config.json + composite_tables.safetensors): the rows that adapter was trained with.
Override init (merge.py's recipe): a word's embedding row = mean of its text spelling's subtoken embeddings, its
lm_head row = 1/2 x the first subtoken's row (below the real spelling: plain text never switches to a word); the
end-of-call token reads as its spelling "\n"; other kernel-emitted tokens start as the base's own rows.
"""

import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import sp_kernel as K  # noqa: E402
import sp_vm as V  # noqa: E402
from modeling_forth_lightning import ForthLightning, ForthLightningConfig  # noqa: E402

WORDS = {t: " " + w for w, t in V.WORDS.items()} | {V.LIT: " lit", V.DEF: " :", V.SEMI: " ;", V.VAR: " VARIABLE",
                                                    V.CALL: " call", V.FETCHM: " @", V.STOREM: " !"}
HANDBACK = {V.DONE: "\n"}  # shortcut: provisional end-of-call spelling, follows sp_vm.DONE


def base_rows(base, names):
    idx = json.loads((Path(base) / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for n in names:
        with safe_open(Path(base) / idx[n], "pt") as f:
            out[n] = f.get_tensor(n)
    return out


@torch.no_grad()
def build(out, base=K.LIGHTNING, keep=None, kernel=None):
    kern = AutoModelForCausalLM.from_pretrained(kernel, dtype=torch.bfloat16) if kernel else K.build()
    emitted = [t for t in range(18, 1000) if kern.lm_head.weight[t].any()]  # reserved ids the kernel emits
    override_ids = sorted(set(WORDS) | set(HANDBACK) | set(emitted))
    word_ids = sorted(WORDS)
    bcfg = json.loads((Path(base) / "config.json").read_text())
    cfg = ForthLightningConfig(base_model=str(base), kernel=kern.config.to_dict(), override_ids=override_ids,
                               word_ids=word_ids, base_hidden_size=bcfg["hidden_size"], vocab_size=bcfg["vocab_size"],
                               eos_token_id=[2, 11], pad_token_id=0, bos_token_id=1)
    m = ForthLightning(cfg).to(torch.bfloat16)
    m.kernel.load_state_dict(kern.state_dict())
    rows = base_rows(base, ["backbone.embeddings.weight", "lm_head.weight"])
    e, lm = rows["backbone.embeddings.weight"].float(), rows["lm_head.weight"].float()
    tok = AutoTokenizer.from_pretrained(base)
    spell = WORDS | HANDBACK
    for i, t in enumerate(override_ids):
        sub = tok(spell[t], add_special_tokens=False).input_ids if t in spell else [t]
        m.embed_override.weight[i] = e[sub].mean(0)
    for i, t in enumerate(word_ids):
        m.word_head.weight[i] = lm[tok(WORDS[t], add_special_tokens=False).input_ids[0]] / 2
    kept = {}
    if keep:
        adapter = (Path(keep) / "composite_config.json").exists()  # an adapter dir from training/push.py
        old = json.loads((Path(keep) / ("composite_config.json" if adapter else "config.json")).read_text())
        with safe_open(Path(keep) / ("composite_tables.safetensors" if adapter else "model.safetensors"), "pt") as f:
            for key, ids, new in (("embed_override.weight", old["override_ids"], override_ids),
                                  ("word_head.weight", old["word_ids"], word_ids)):
                w = f.get_tensor(key)
                both = [t for t in ids if t in new]
                getattr(m, key.split(".")[0]).weight[[new.index(t) for t in both]] = w[[ids.index(t) for t in both]]
                kept[key] = {"kept": len(both), "new": sorted(set(new) - set(ids))}
    m.save_pretrained(out)
    tok.save_pretrained(out)
    shutil.copy(Path(base) / "generation_config.json", Path(out) / "generation_config.json")
    n = lambda mod: sum(p.numel() for p in mod.parameters())
    info = {"override_ids": len(override_ids), "word_ids": len(word_ids), "emitted": len(emitted),
            "params_override": n(m.embed_override) + n(m.word_head), "params_kernel": n(m.kernel),
            "bytes_on_disk": sum(p.stat().st_size for p in Path(out).glob("*.safetensors")), "kept_from": keep} | kept
    print(json.dumps(info))
    json.dump(info, open(Path(out) / "build_info.json", "w"))


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    opt = lambda k: next((x[len(k):] for x in sys.argv[1:] if x.startswith(k)), None)
    build(*a, keep=opt("--keep="), kernel=opt("--kernel="))
