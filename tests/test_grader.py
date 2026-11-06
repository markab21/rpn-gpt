"""The exact grader (training/loop_sim.same) on the false negatives the ledger records (grader fixes 1-4)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from loop_sim import same  # noqa: E402

CASES = [  # (answer, expected, problem fields, should match)
    ("Yes, 7432 is composite", "True", {"q": "Is 7432 composite?"}, True),
    ("51353 is composite", "False", {"q": "Is 51353 prime?"}, True),  # fix 4: the stated property, not the first word
    ("51353 is not prime", "False", {"q": "Is 51353 prime?"}, True),
    ("x = 4 or x = -3", "-3 4", {"q": "Solve ...", "family": "DM.polynomial_roots"}, True),
    ("-5 (double root)", "-5", {"q": "Solve ...", "family": "DM.polynomial_roots"}, True),
    ("15500 micrograms", "15500", {"q": "How many micrograms are in 15.5 grams?"}, True),
    ("1,234", "1234", {"q": ""}, True),
    ("\\frac{5}{33}", "5/33", {"q": ""}, True),
    ("1235", "1234", {"q": ""}, False),
    ("No", "True", {"q": "Is 7 prime?"}, False),
]


def test_grader_cases():
    for got, want, p, ok in CASES:
        assert same(got, want, p) is ok, (got, want)
