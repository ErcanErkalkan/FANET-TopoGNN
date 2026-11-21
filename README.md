# FANET-TopoGNN — current research snapshot

The **default branch `main`** contains one current research snapshot. Project code, experiments and scientific claims remain independent of any journal. The manuscript's presentation settings are maintained solely in `paper/manuscript/FORMAT_REQUIREMENTS.md`.

## Main finding

45 trajectory groups (180 paired radio/graph runs, 60 simulated minutes). Frozen TopoGNN threshold 0.15 produced 183 matches among 1,114 fragmentation onsets and 88 unmatched alarms (1.47/min overall); six of 15 block-by-mobility cells exceeded two/min. Silent 30-node runs missed all 82 onsets. History-4 remains a retrospective exploratory comparison, not model superiority evidence. These are development-only simulated connectivity outcomes, not real packet delivery or prospective independent validation.

## One complete artifact

The full experiment snapshot (including recorded CSV data and one current draft) is split into small binary **transport chunks** under `snapshot/` to keep GitHub writes manageable. The parts are **one ZIP**, not alternative release versions. Reconstruct and verify it:

```bash
python assemble_snapshot.py
```

Expected SHA-256: `4b224751ebda13025dcac322af4ab009c6da67eb4e1e5196c739bad1ec3ff3ea`. Extract `FANET-TopoGNN_Current.zip` and then run:

```bash
python run.py check
python run.py verify
python run.py test
python run.py build-paper
```

The repo's browsable `core/`, `frozen/`, `audit/`, `data_derived/` and `paper/` directories are corresponding text-source excerpts; full binary evidence is in the reconstructed ZIP. Use the extracted ZIP for exact manifest checks. `run.py build-paper` uses pdflatex and BibTeX. An XeLaTeX compilation path allows exact Times New Roman when that font is installed; a substitute font is not exact compliance.

## Scientific boundaries

The main evidence is a 2-D received-power threshold graph and does not establish real PHY/MAC behavior, routing performance, PDR, effective intervention, or field readiness. No independent blind test was opened. Model selection and post-hoc baseline limitations are described in the manuscript.

Project status: **RESEARCH / AUTHOR REVIEW — HOLD**. Human author approval and verified data-sharing permissions are required before external submission.
