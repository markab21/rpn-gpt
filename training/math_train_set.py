"""The round-4 math guard's problems: N from the MATH *train* split (EleutherAI/hendrycks_math, 7 subjects; MATH-500 is
drawn from the test split, and no problem text here appears in training/data/math500.json), stratified by subject, answer =
the solution's last \\boxed{}. s2a_r3 answers them without the machine; its correct direct answers become ~5% of the rows.
  python training/math_train_set.py [N]   -> training/data/math_train600.json   (needs pyarrow + huggingface_hub)"""
import json, random, sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from loop_sim import last_boxed  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 600
SUBJ = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
held = {" ".join(p["q"].split()) for p in json.load(open(ROOT / "training/data/math500.json"))}
R, out = random.Random("math-train-2026-11-05"), []
for s in SUBJ:
    t = pq.read_table(hf_hub_download("EleutherAI/hendrycks_math", f"{s}/train-00000-of-00001.parquet", repo_type="dataset"))
    rows = [r for r in t.to_pylist() if last_boxed(r["solution"]) and " ".join(r["problem"].split()) not in held]
    for r in R.sample(rows, N // len(SUBJ)):
        out.append({"q": r["problem"], "a": last_boxed(r["solution"]), "family": f"math_train.{s}", "kind": "math",
                    "level": r.get("level")})
assert not {" ".join(p["q"].split()) for p in out} & held
json.dump(out, open(ROOT / f"training/data/math_train{N}.json", "w"), indent=1)
print(len(out), "problems")
