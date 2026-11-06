"""Start an AutoModel LoRA stage from the previous stage's adapter: copy PREV (a checkpoint dir, e.g.
CK/sft_cur_L0/epoch_1_step_83) to NEW/epoch_0_step_0 with model, optimizer and rng state as they are, the step
scheduler at step 0 and the dataloader at the start of an epoch (AutoModel's restore needs every part; the new stage
has its own dataset). Same GPU count / parallel layout as PREV (per-DP-rank dataloader files, sharded optimizer).
  venv-automodel/bin/python training/stage_init.py PREV NEW
  then train with --checkpoint.checkpoint_dir NEW --checkpoint.restore_from epoch_0_step_0
"""

import shutil
import sys
from pathlib import Path

import torch


def main(prev, new):
    prev, dst = Path(prev), Path(new) / "epoch_0_step_0"
    assert not dst.exists(), f"{dst} exists"
    dst.mkdir(parents=True)
    for part in ("model", "optim", "rng"):
        shutil.copytree(prev / part, dst / part)
    torch.save({"step": 0, "epoch": 0}, dst / "step_scheduler.pt")
    (dst / "dataloader").mkdir()
    for f in sorted((prev / "dataloader").glob("*.pt")):
        st = torch.load(f, weights_only=False)
        st.update({"_sampler_iter_state": {"samples_yielded": 0, "sampler_state": {"yielded": 0}},
                   "_sampler_iter_yielded": 0, "_num_yielded": 0, "_iterator_finished": False})
        torch.save(st, dst / "dataloader" / f.name)
    print("stage init", prev, "->", dst)


if __name__ == "__main__":
    main(*sys.argv[1:3])
