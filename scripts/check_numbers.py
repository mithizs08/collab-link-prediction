"""Check that every number in the write-up comes from a committed results / report file (rule 3).

  python scripts/check_numbers.py [files ...]        # default: report/report.md README.md

Two checks against report/numbers.yaml, a provenance manifest that maps every number token to its source(s):
  1. every manifest source reproduces its token at the precision shown (row filters must select exactly one row,
     or several rows with `agg: mean`; percentages via `pct: true`; signs must agree);
  2. every checked number in the text has a manifest entry.
Sources must be files tracked by git (uncommitted local files, e.g. per-pair SHAP samples, do not count).

Checked tokens: every decimal number, every percentage, and every integer >= 100 that is not a year (1900-2099).
Not checked: integers below 100 (seeds, counts of features / fields, section numbers) and years; --verbose lists them.
Skipped text: the YAML header, fenced code blocks, inline code, link / image targets, {attributes} and References.
Exit code 1 on any failure.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

NUM = re.compile(r"(?<![\w.])([+\-−]?)(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(%?)(?![\w])")
MANIFEST = Path("report/numbers.yaml")


def clean(text: str) -> list[tuple[int, str]]:
    """Lines to check, with code, link targets, attributes, YAML header and references removed."""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        text = "\n" * text[:end + 4].count("\n") + text[end + 4:]
    out, fence = [], False
    for i, line in enumerate(text.splitlines(), 1):
        if line.strip().startswith("```"):
            fence = not fence
            continue
        if re.match(r"#+\s*References", line):
            break
        if fence:
            continue
        line = re.sub(r"`[^`]*`", " ", line)
        line = re.sub(r"\]\([^)]*\)", "]", line)
        line = re.sub(r"\{[^}]*\}", " ", line)
        line = re.sub(r"<[^>]*>", " ", line)
        out.append((i, line))
    return out


def tokens(path: Path, verbose=False):
    found, skipped = {}, []
    for ln, line in clean(path.read_text()):
        for m in NUM.finditer(line):
            sign, whole, frac, pct = m.groups()
            v = float(whole.replace(",", ""))
            if not frac and not pct and (v < 100 or 1900 <= v <= 2099):
                skipped.append(f"{path}:{ln}: {m.group(0)}")
                continue
            found.setdefault(m.group(0).replace("−", "-"), []).append(f"{path}:{ln}")
    if verbose:
        print(f"{path}: {len(skipped)} small integers / years not checked")
        for s in skipped:
            print("   ", s)
    return found


def parse(token: str) -> tuple[float, int]:
    sign, whole, frac, pct = NUM.fullmatch(token).groups()
    v = float(whole.replace(",", "") + (frac or ""))
    dec = len(frac) - 1 if frac else 0
    if pct:
        v, dec = v / 100.0, dec + 2
    return (-v if sign in ("-", "−") else v), dec


def config_value(path: str):
    node = yaml.safe_load(Path("configs/default.yaml").read_text())
    for k in path.split("."):
        node = node[k]
    return float(node)


def source_value(src: dict, tracked: set) -> float:
    if "constant" in src:
        return None
    if "config" in src:
        return config_value(src["config"])
    f = src["f"]
    if f not in tracked:
        raise ValueError(f"{f} is not tracked by git")
    if f.endswith(".json"):
        return float(json.loads(Path(f).read_text())[src["key"]])
    d = pd.read_csv(f)
    for k, v in src.get("w", {}).items():
        d = d[d[k].astype(str) == str(v)]
    col = src.get("col", "value")
    if src.get("agg") == "mean":
        if not len(d):
            raise ValueError("no rows")
        x = float(d[col].mean())
    else:
        if len(d) != 1:
            raise ValueError(f"filters select {len(d)} rows")
        x = float(d[col].iloc[0])
    return x


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", default=["report/report.md", "README.md"])
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    tracked = set(subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout.split())
    man = {k: v for k, v in yaml.safe_load(MANIFEST.read_text()).items() if not k.startswith("_")}
    errors = []
    # 1. manifest entries reproduce their token
    for tok, srcs in man.items():
        target, dec = parse(tok)
        for src in srcs:
            try:
                x = source_value(src, tracked)
            except Exception as e:  # noqa: BLE001
                errors.append(f"manifest {tok}: {src}: {e}")
                continue
            if x is None:
                continue
            if abs(x - target) > 0.5 * 10 ** (-dec) + 1e-9:
                errors.append(f"manifest {tok}: source gives {x:.6g}: {src}")
    # 2. every number in the text is in the manifest
    n = 0
    for f in args.files:
        for tok, where in tokens(Path(f), args.verbose).items():
            n += len(where)
            if tok not in man:
                errors.append(f"no manifest entry for {tok} ({', '.join(where)})")
    if errors:
        print(f"{len(errors)} problem(s):")
        print("\n".join(errors))
        sys.exit(1)
    print(f"OK: {n} number occurrences in {', '.join(args.files)} traced to {len(man)} manifest tokens; "
          "every manifest source reproduces its token from a git-tracked file")


if __name__ == "__main__":
    main()
