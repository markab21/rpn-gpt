"""Run Forth blocks through the spec (kernel/text_vm.py) and print each output block, as the model would see it.
  uv run python -m forth.run '17 5 mod .' ': sq dup * ; 12 sq .'      (one argument per block; state carries)
  uv run python -m forth.run -f program.fs                             (one block from a file)
The first run downloads the pinned Lightning tokenizer from the Hugging Face Hub (a few MB)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kernel"))
import text_vm  # noqa: E402

from forth import sp  # noqa: E402


def run(blocks):
    """Blocks of one turn -> the text of each output block."""
    return text_vm.blocks(blocks, sp.tokenizer())


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["-f"]:
        args = [open(args[1]).read()]
    for out in run(args):
        print("```output\n" + out + "\n```")
