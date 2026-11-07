"""Build the benchmark problem sets (bench/sets/*.json) with Qwenputer's own loaders.

    python bench/make_sets.py DATA_DIR
DATA_DIR holds the two raw sources:
  dm/mathematics_dataset-v1.0/extrapolate/arithmetic__*.txt
      https://storage.googleapis.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz
  nupa/test.json
      https://huggingface.co/datasets/HaotongYang/NUPA_text (test.json, 688 MB)

load_dm and load_nupa are copied from qwenputer chipformer/q35/arith_bench.py (MIT, commit cb1c969).
The DeepMind sets are exactly what qwenputer's RISC-V eval (chipformer/rv/evalarith.py --n 40, seed 0)
samples. NUPA: their loader at n=144 per task gives 4 tasks x 18 lengths x 8 = 576 problems, the
count of their CHIP-8 NUPA row; we take a seeded sample of 40 from those 576.
"""

import json
import random
import re
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

SEED, N = 0, 40
SUFFIX = "\n\nGive the final answer as a single number in \\boxed{} at the end."  # qwenputer evals.SUFFIX
DM = {"mul_big": "arithmetic__mul_big", "div_big": "arithmetic__div_big",
      "add_or_sub_big": "arithmetic__add_or_sub_big",
      "mul_div_multiple_longer": "arithmetic__mul_div_multiple_longer"}
NUPA_TASKS = {"nupa_add": ("add_Integer_Integer_Integer", "+"), "nupa_sub": ("sub_Integer_Integer_Integer", "-"),
              "nupa_mulh": ("multiply_hard_Integer_Integer_Integer", "*"),
              "nupa_mule": ("multiply_easy_Integer_Integer_Integer", "*")}


def load_dm(data_dir, mod, n=N, seed=SEED):
    lines = (data_dir / "dm" / "mathematics_dataset-v1.0" / "extrapolate" / f"{mod}.txt").read_text().splitlines()
    pairs = [(lines[i].strip(), lines[i + 1].strip()) for i in range(0, len(lines) - 1, 2)]
    idx = sorted(random.Random(seed).sample(range(len(pairs)), min(n, len(pairs))))
    return [(i, *pairs[i]) for i in idx]


def load_nupa(full, data, n=144, seed=SEED):
    key, sym = NUPA_TASKS[data]
    per, out = max(1, n // 18), []
    for L in range(3, 21):
        items = full[key][str(L)]
        order = list(range(len(items)))
        random.Random(f"{seed}-{data}-{L}").shuffle(order)
        for j in sorted(order[:per]):
            task = items[j].split(" . ", 1)[1]  # drop "Directly return the answer ... ."
            m = re.fullmatch(r"(.*?:\s*)(\d+) (\S) (\d+) = (\d+)", task)
            assert m and m[3] == sym, items[j]
            out.append((f"{data}-L{L}-{j}", f"{m[1]}{m[2]} {m[3]} {m[4]}", m[5], L))
    return out


def main(data_dir):
    data_dir, out = Path(data_dir), Path(__file__).parent / "sets"
    out.mkdir(exist_ok=True)
    src_dm = "DeepMind mathematics_dataset v1.0, extrapolate split, test file"
    for name, mod in DM.items():
        rows = [{"id": f"dmx_{name}-{i}", "set": name, "question": q, "gold": a, "q": q + SUFFIX, "a": a,
                 "family": f"dm.{name}"} for i, q, a in load_dm(data_dir, mod)]
        meta = {"source": f"{src_dm} {mod}.txt", "sampler": "qwenputer arith_bench.load_dm", "n": N, "seed": SEED}
        json.dump({"meta": meta, "problems": rows}, open(out / f"{name}.json", "w"), indent=1)
    full = json.loads((data_dir / "nupa" / "test.json").read_text())
    pool = [r for d in NUPA_TASKS for r in load_nupa(full, d)]
    assert len(pool) == 576
    pick = sorted(random.Random(SEED).sample(range(len(pool)), N))
    rows = [{"id": i, "set": "nupa", "question": q, "gold": a, "digits": L, "q": q + SUFFIX, "a": a,
             "family": "nupa"} for i, q, a, L in (pool[k] for k in pick)]
    meta = {"source": "NUPA_text test.json (HaotongYang/NUPA_text), integer add / sub / multiply_hard / multiply_easy",
            "sampler": "qwenputer arith_bench.load_nupa at n=144 per task (576), then random.Random(0).sample 40",
            "n": N, "seed": SEED}
    json.dump({"meta": meta, "problems": rows}, open(out / "nupa.json", "w"), indent=1)
    for f in sorted(out.glob("*.json")):
        P = json.load(open(f))["problems"]
        dec = sum(Fraction(Decimal(p["gold"])) .denominator != 1 if "/" not in p["gold"] else True for p in P)
        print(f.name, len(P), "non-integer golds", dec, "|", P[0]["question"], "=", P[0]["gold"])


if __name__ == "__main__":
    main(sys.argv[1])
