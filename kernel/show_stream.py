"""Print the full token stream of one block as the composite sees it: model tokens in quotes, the kernel's hidden
tokens in [brackets] (named after kernel/sp_vm.py; MD[d] = machine digit d), its visible tokens in {braces}.
  uv run python kernel/show_stream.py '2 3 + .'"""

import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import sp_vm as V  # noqa: E402
import text_vm as T  # noqa: E402

from forth import sp  # noqa: E402

NAMES = {}
for k, v in vars(V).items():
    if k.isupper() and isinstance(v, int):
        NAMES.setdefault(v, k)
    elif k.isupper() and isinstance(v, (list, tuple)):
        for j, x in enumerate(v):
            if isinstance(x, int):
                NAMES.setdefault(x, f"{k}[{j}]")


def show(program):
    tok = sp.tokenizer()
    full, owned, _ = T.stream(T.text_ids(program, tok), tok)
    owned = set(owned)
    parts = []
    for i, t in enumerate(full):
        if i not in owned:
            parts.append(repr(tok.decode([t])))
        elif t < 1000:  # the kernel's ids below 1000 are reserved special tokens: hidden, they decode to ''
            parts.append(f"[{NAMES.get(t, t)}]")
        else:
            parts.append("{" + repr(tok.decode([t])) + "}")
    hidden = sum(1 for i in owned if full[i] < 1000)
    return " ".join(parts) + f"\n{len(full)} tokens: {len(full) - len(owned)} model, {hidden} hidden, {len(owned) - hidden} visible kernel"


if __name__ == "__main__":
    print(show(" ".join(sys.argv[1:])))
