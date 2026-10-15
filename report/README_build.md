# Building the PDF

`report.md` is pandoc markdown (YAML header sets A4, 10 pt, 2.2 cm margins; figures are linked from `../figures/final/`).
pandoc 3.8 is available in the project environment, but no PDF engine was installed when the report was written.
Install one, then build from this folder so the relative figure paths resolve:

```bash
brew install --cask basictex        # or: conda install -c conda-forge tectonic
cd report && pandoc report.md -o report.pdf --pdf-engine=xelatex      # or --pdf-engine=tectonic
```

Before building, check every number: `python scripts/check_numbers.py` (from the repo root), and check every
reference marked [VERIFY].
