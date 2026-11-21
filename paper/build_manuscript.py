"""Compile the manuscript and supplement into a new folder without changing sources."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess

root = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--out", type=Path)
a = p.parse_args()
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
output = a.out.resolve() if a.out else root.parent / "outputs" / ("paper_" + stamp)
output.mkdir(parents=True, exist_ok=False)
inputs = ("main.tex", "references.bib", "ESM_1.tex")
for name in inputs:
    shutil.copyfile(root / "manuscript" / name, output / name)
latex = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
bibtex = shutil.which("bibtex") or shutil.which("bibtex8")
if bibtex is None:
    raise SystemExit("Neither bibtex nor bibtex8 is available; install a TeX bibliography engine.")
steps = [latex, [bibtex, "main"], latex, latex,
         ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "ESM_1.tex"]]
for i, command in enumerate(steps, start=1):
    result = subprocess.run(command, cwd=output, text=True, capture_output=True)
    (output / f"build_step_{i}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        print("\n".join((result.stdout + result.stderr).splitlines()[-23:]))
        raise SystemExit(f"Build step {i} failed. See {output}")
print(f"Manuscript: {output / 'main.pdf'}")
print(f"Supplement: {output / 'ESM_1.pdf'}")
