"""Merge e2e_demo shard rows by idx (first file wins), re-check every kernel stream against kernel/text_vm.py, tag each
mismatched row by feature (its first differing kernel run), print the totals (correct, machine-backed, exactness, time).
  cd nemotron; PYTHONPATH=composite python kernel/composite/e2e_report.py SYSTEM.txt OUT_MERGED.json SHARD.json ..."""
import collections
import json
import re
import statistics as st
import sys

import e2e  # kernel/composite/ (its text_vm is the repo's kernel/text_vm.py)
from transformers import AutoTokenizer

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "training"))
from loop_sim import norm  # noqa: E402

tok = AutoTokenizer.from_pretrained("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", revision="a9904d24bcc1d289a1950fa9d2b978c47cf903b9")
sysm = open(sys.argv[1]).read()
rows = {}
for f in sys.argv[3:]:
    for r in json.load(open(f))["rows"]:
        rows.setdefault(r["idx"], r)
R = [rows[i] for i in sorted(rows)]
F = "`" * 3
DOT_S = re.compile(r"(?:^|\s)\.s(?:\s|$)")


def tag(b):
    if "<SPECIAL_810>" in b["want"] + b["got"]:
        return "code-past-full"
    if "<SPECIAL_809>" in b["want"] + b["got"]:
        return "vars-past-full"
    if "only defined" in b["want"] + b["got"]:
        return "named-note"
    if DOT_S.search(b["context"].rsplit("```forth", 1)[-1]):
        return ".s"
    return "other"


tags = collections.Counter()
for r in R:
    msgs = [{"role": "system", "content": sysm}, {"role": "user", "content": r["q"]}]
    prompt = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True),
                 add_special_tokens=False).input_ids
    c = e2e.check(tok, prompt, r["ids"], set(r["claimed"]))
    r.update(c)
    r["tag"] = tag(c["bad_runs"][0]) if c["bad_runs"] else ("other" if not c["exact"] else None)
    r["tags_all"] = sorted({tag(b) for b in c["bad_runs"]})  # (bad_runs: the first 12 differing runs)
    tags[r["tag"]] += r["tag"] is not None
json.dump({"rows": R}, open(sys.argv[2], "w"))


def backed(r):
    outs = [" ".join(re.sub(r"\nerror:.*", "", o, flags=re.S).split()) for o in r["blocks_kernel"]]
    return bool(r["ok"]) and "</think>" in r["text"] and r["text"].count(F + "output\n") == len(outs) \
        and any(norm(r["got"]) in o for o in outs)


n, ok, mb = len(R), sum(r["ok"] for r in R), sum(map(backed, R))
print(f"n={n} correct {ok} ({100 * ok / n:.1f}%) machine-backed {mb} ({100 * mb / n:.1f}%) used machine {sum(bool(r['blocks_kernel']) for r in R)}")
kp = sum(r["kernel_tokens"] for r in R)
rr = [sum(r["runs_real"][i] + r["runs_prose"][i] for r in R) for i in (0, 1)]
print(f"exact rows {sum(r['exact'] for r in R)}/{n}; kernel runs {rr[0]}, differing {rr[1]}; kernel tokens {kp}; mismatch rows by feature {dict(tags)}")
ex = [r for r in R if r["tag"] is None]
print(f"correct among exact rows {sum(r['ok'] for r in ex)}/{len(ex)}")
for k in ("visible_tokens", "hidden_tokens", "s"):
    v = [r[k] for r in R]
    print(f"{k}: median {st.median(v):.0f} mean {st.mean(v):.0f} max {max(v):.0f}")
print("turn_cut", sum(r["turn_cut"] for r in R), "safety_cut", sum(r["safety_cut"] for r in R))
fam = collections.defaultdict(lambda: [0, 0])
for r in R:
    fam[r["family"]][0] += r["ok"]
    fam[r["family"]][1] += 1
print("families", dict(sorted(fam.items())))
for r in R:
    if r["tag"]:
        b = r["bad_runs"][0] if r["bad_runs"] else {}
        print(f"MISMATCH idx {r['idx']} {r['family']} ok={r['ok']} tag={r['tag']} all={r['tags_all']} bad runs {r['runs_real'][1] + r['runs_prose'][1]}"
              f" | at: {b.get('context', '')[-70:]!r} | want {b.get('want', '')[:90]!r} | got {b.get('got', '')[:90]!r}")
