"""DeepMind mathematics_dataset problems (mechanical math: arithmetic, linear equations, gcd/lcm/remainders, primes,
rounding, unit conversion, polynomial evaluation, probability) with exact answers, as eval / train JSONL for the
curriculum loaders (level "DM", family = module, prompt, output, answer). Symbolic-answer modules (factor,
differentiate, expand) are left out: an integer kernel cannot help with them.
  eval:  extrapolation modules (bigger / longer than its training split), --n per module
  interp: interpolation test modules (held-out problems at training difficulty; covers every module), --n per module
  train: training modules at the hard end of the entropy range, --n per module
Needs the generator's own env (sympy 1.5): uv venv ENV --python 3.11; uv pip install --python ENV/bin/python \
  mathematics_dataset "sympy==1.5.1" "numpy<2"
  ENV/bin/python training/dm_gen.py eval --n 40 --out data/rendered/sp_kernel/DM_eval.jsonl
  ENV/bin/python training/dm_gen.py train --n 300 --out data/rendered/sp_kernel/DM_train.jsonl
"""

import argparse
import collections
import json
import random
import re

import numpy as np
from mathematics_dataset.modules import modules

SKIP = re.compile(r"factor|differentiate|expand|collect|simplify|compose|coefficient|base_conversion|in_base|sequence_nth")


def kind(answer):
    a = str(answer).strip()
    if re.fullmatch(r"-?\d+", a):
        return "integer"
    if re.fullmatch(r"-?\d+/\d+", a):
        return "fraction"
    if re.fullmatch(r"-?\d+\.\d+", a):
        return "decimal"
    if a in ("True", "False"):
        return "bool"
    if re.fullmatch(r"-?[\d./]+(, -?[\d./]+)+", a):
        return "list"
    return None  # symbolic / multiple-choice letters: skipped


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("split", choices=["eval", "interp", "train"])
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    random.seed(a.seed + ["eval", "interp", "train"].index(a.split))
    np.random.seed(a.seed + ["eval", "interp", "train"].index(a.split))
    mods = (modules.test_extra() if a.split == "eval" else modules.test() if a.split == "interp" else
            modules.train(entropy_fn=lambda r: (r[0] + 0.7 * (r[1] - r[0]), r[1])))
    rows, kinds, seen = [], collections.Counter(), set()
    for area, ms in sorted(mods.items()):
        for name, gen in sorted(ms.items()):
            fam = f"{area}.{name}"
            if SKIP.search(name):
                continue
            got, tries, errs = 0, 0, 0
            while got < a.n and tries < 20 * a.n and errs < 20:
                tries += 1
                try:
                    ex = gen()
                except Exception as e:  # noqa: BLE001 - some old modules break on newer numpy types: skip them
                    errs += 1
                    err = e
                    continue
                q, ans = str(ex.question).strip(), str(ex.answer).strip()
                k = kind(ans)
                if k is None or q in seen:
                    continue
                seen.add(q)
                rows.append({"level": "DM", "family": fam, "kind": k, "prompt": q, "output": ans, "answer": ans})
                kinds[k] += 1
                got += 1
            print(f"{fam}: {got}" + (f" (skipped after errors: {str(err)[:60]})" if errs >= 20 else ""), flush=True)
    with open(a.out, "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} rows -> {a.out}; answer kinds {dict(kinds)}")


if __name__ == "__main__":
    main()
