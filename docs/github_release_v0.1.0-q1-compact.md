# FANET-TopoGNN compact manuscript artefacts

## Release fields

Tag: `v0.1.0-q1-compact`

Target: `main`

Release title: `FANET-TopoGNN compact manuscript artefacts`

Pre-release: unchecked

Set as latest release: checked

## Release notes

This release archives the executable FANET-TopoGNN / Kinetic-TopoGuard reproducibility suite and the compact manuscript-facing artefact bundle used to cross-check the reported results.

Repository: `ErcanErkalkan/FANET-TopoGNN`

Repository URL: <https://github.com/ErcanErkalkan/FANET-TopoGNN>

Archived release DOI: <https://doi.org/10.5281/zenodo.20226053>

### Included artefacts

- `outputs/q1_publication_compact/summary.json`
- `outputs/q1_publication_compact/manuscript_summary.json`
- `outputs/q1_publication_compact/manuscript_tables.tex`
- `outputs/q1_publication_compact/claims_summary.md`
- `outputs/q1_publication_compact/report.md`
- `outputs/q1_publication_compact/artifact_manifest.txt`
- Aggregate CSV and LaTeX tables for metrics, risk metrics, lead time, dataset summary, network metrics, ablations, and statistical tests
- Generated PDF and PNG figures under `outputs/q1_publication_compact/figures/`
- Per-seed tables, figures, and resume artefacts under `outputs/q1_publication_compact/per_seed/`

### Reproducibility command

```bash
python main.py --config configs/q1_publication_compact.json
```

To continue an interrupted run:

```bash
python main.py --config configs/q1_publication_compact.json --resume
```

### Zenodo

This release is intended for Zenodo archival. The repository includes:

- `.zenodo.json` for Zenodo metadata
- `CITATION.cff` for citation metadata
- `docs/zenodo_release.md` with the Zenodo publication checklist

Zenodo archived DOI: <https://doi.org/10.5281/zenodo.20226053>
