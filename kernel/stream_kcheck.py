"""Long streams: the HF kernel token by token (cached decode) on text_vm's reference stream of saved rows, checked on the
fly (no logits kept): kernel-owned positions argmax = text_vm's token with margin >= 0.5, model positions max logit
< 0.5. cwd kernel/.   python stream_kcheck.py KERNEL_DIR ROWS.json [ROW_NUMBERS|all] [CUT: stop 200 after the
generated stream's first mismatch]"""
import json
import sys

import torch
from transformers import AutoModelForCausalLM

import text_vm as T
from forth import sp

tok = sp.tokenizer()
rows = json.load(open(sys.argv[2]))["rows"]
pick = range(len(rows)) if len(sys.argv) < 4 or sys.argv[3] == "all" else map(int, sys.argv[3].split(","))
cut = len(sys.argv) > 4
m = AutoModelForCausalLM.from_pretrained(sys.argv[1], dtype=torch.bfloat16).cuda().eval()
for k in pick:
    r = rows[k]
    gen, claimed = r["ids"], set(r["claimed"])
    ids, owned, _ = T.stream([t for i, t in enumerate(gen) if i not in claimed], tok)
    own = set(owned)
    gm = next((i for i in range(min(len(ids), len(gen))) if ids[i] != gen[i] or ((i in own) != (i in claimed))), None)
    n = min(len(ids), gm + 200) if cut and gm is not None else len(ids)
    x, past, wrong, loud, low, mn = torch.tensor([ids[:n]], device="cuda"), None, [], [], [], 99.0
    with torch.no_grad():
        for p in range(n):
            out = m(x[:, p:p + 1], past_key_values=past, use_cache=True)
            past, lg = out.past_key_values, out.logits[0, -1].float()
            t2 = lg.topk(2)
            if p + 1 in own:
                mg = float(t2.values[0] - t2.values[1])
                mn = min(mn, mg)
                if int(t2.indices[0]) != ids[p + 1]:
                    wrong.append((p, ids[p + 1], int(t2.indices[0])))
                elif mg < 0.5:
                    low.append((p, mg))
            elif float(t2.values[0]) >= 0.5:
                loud.append((p, int(t2.indices[0])))
    print(k, "idx", r["idx"], "seed", r.get("seed"), "tokens", n, "of", len(ids), "generated-stream first mismatch", gm,
          "| wrong", len(wrong), wrong[:4], "loud", len(loud), loud[:4], "low margin", len(low), low[:4],
          "min margin", round(mn, 3), flush=True)
