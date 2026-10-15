"""Every number in README.md and report/report.md is traced to a git-tracked results / report file (rule 3)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_writeup_numbers_have_sources():
    r = subprocess.run([sys.executable, "scripts/check_numbers.py"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
