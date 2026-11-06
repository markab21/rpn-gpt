"""The spec's own self-check (kernel/text_vm.py main) runs green in this layout."""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_text_vm_self_check():
    out = subprocess.run([sys.executable, str(ROOT / "kernel/text_vm.py")], capture_output=True, text=True, check=True).stdout
    counts = re.findall(r"(\d+)/(\d+) as expected", out)
    assert len(counts) == 4, out
    assert all(a == b for a, b in counts), out
