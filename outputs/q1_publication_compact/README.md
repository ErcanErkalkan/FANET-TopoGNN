# Compact Manuscript Artefact Bundle

Repository: `ErcanErkalkan/FANET-TopoGNN`

Repository URL: <https://github.com/ErcanErkalkan/FANET-TopoGNN>

This directory contains the compact manuscript-facing artefacts used to cross-check the reported results. It is intentionally tracked even though other generated `outputs/` folders remain ignored.

## Reproduce

From the repository root:

```bash
python main.py --config configs/q1_publication_compact.json
```

To continue an interrupted run:

```bash
python main.py --config configs/q1_publication_compact.json --resume
```

## Contents

- `summary.json` and `manuscript_summary.json`: compact run summaries.
- `*.csv` and `*.tex`: manuscript-facing tables and aggregate metrics.
- `figures/`: generated PDF and PNG figure artefacts.
- `per_seed/`: per-seed tables, figures, and resume artefacts for auditability.
- `artifact_manifest.txt`: manifest emitted by the pipeline for this output directory.

The full directory is included in GitHub releases and Zenodo archives so the manuscript claim that the repository contains `outputs/q1_publication_compact/` remains directly verifiable.
