"""Replay saved loop transcripts (training/loop_sim.py results: real model programs) through the CURRENT text_vm, block by
block with state carried as in the turn: what a kernel / dialect change does to real programs, without a model.
Reports error kinds, unknown words, outputs that changed from the saved run, and blocks that print the expected answer.
  python kernel/replay.py RESULTS.json [MORE.json ...]     (e.g. the HF dataset's runs/20261103/r1_*_eval.json)
"""

import collections
import json
import re
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import text_vm as T  # noqa: E402
from forth import sp  # noqa: E402


def main():
    tok, kinds, words, c = sp.tokenizer(), collections.Counter(), collections.Counter(), collections.Counter()
    for f in sys.argv[1:]:
        for r in json.load(open(f)):
            blocks = [b for b in r["blocks"] if b["program"]]
            if not blocks:
                continue
            outs = T.blocks([b["program"] for b in blocks], tok)
            for b, o in zip(blocks, outs):
                c["blocks"] += 1
                c["changed"] += o != b["output"]
                c["print the answer"] += " ".join(o.replace(",", " ").split()) == " ".join(str(r["a"]).split())
                for line in o.splitlines():
                    if line.startswith("error"):
                        kinds[re.sub(r"\d+", "N", line)[:80]] += 1
                        if m := re.match(r"error: unknown word (\S+)", line):
                            words[m.group(1)] += 1
    print(f"{c['blocks']} blocks: {c['changed']} outputs changed from the saved run, {c['print the answer']} print the "
          f"expected answer, {sum(kinds.values())} errors")
    for k, n in kinds.most_common(15):
        print(f"  {n:5}  {k}")
    print("unknown words:", " ".join(f"{w}:{n}" for w, n in words.most_common(30)))


if __name__ == "__main__":
    main()
