"""Every ```forth block in the docs, with the ```output block right after it, runs through the spec (kernel/text_vm.py)
and prints exactly that output. Blocks under one heading share state, as the blocks of one model turn do. A section
whose text holds <!-- no-check --> is skipped (real outputs from an older spec, labelled there), and so is a pair whose
output was trimmed ("[…]")."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "kernel"), str(ROOT)]
import text_vm  # noqa: E402

from forth import sp  # noqa: E402

DOCS = [p for p in [ROOT / "README.md", *sorted(ROOT.glob("*/*.md"))] if p.exists()]
PAIR = re.compile(r"```forth\n(.*?)\n```\n```output\n(.*?)\n```", re.S)


def sections():
    for doc in DOCS:
        for k, sec in enumerate(re.split(r"\n(?=#{1,4} )", doc.read_text())):
            pairs = [(p, o) for p, o in PAIR.findall(sec) if "[…]" not in o]
            if pairs and "<!-- no-check -->" not in sec:
                yield f"{doc.relative_to(ROOT)} section {k}", pairs


def test_doc_examples_run_in_the_spec():
    tok, n = sp.tokenizer(), 0
    for where, pairs in sections():
        got = text_vm.blocks([p for p, _ in pairs], tok)
        for (p, want), g in zip(pairs, got):
            assert g == want, (where, p, want, g)
            n += 1
    assert n >= 20, n
