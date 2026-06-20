# FANET-TopoGNN v0.1.12 publication consistency artefacts

## Release fields

Tag: `v0.1.12`

Target: `main`

Release title: `FANET-TopoGNN v0.1.12 publication consistency artefacts`

Pre-release: unchecked

Set as latest release: checked

## Release notes

This release archives the executable FANET-TopoGNN / Kinetic-TopoGuard reproducibility suite and the compact manuscript-facing artefact bundle after the consistency cleanup.

Repository: `ErcanErkalkan/FANET-TopoGNN`

Repository URL: <https://github.com/ErcanErkalkan/FANET-TopoGNN>

Archived release DOI: <https://doi.org/10.5281/zenodo.20226053>

### Included artefacts

- `outputs/publication_compact/summary.json`
- `outputs/publication_compact/runtime_profile.json`
- `outputs/publication_compact/manuscript_summary.json`
- `outputs/publication_compact/manuscript_tables.tex`
- `outputs/publication_compact/claims_summary.md`
- `outputs/publication_compact/report.md`
- `outputs/publication_compact/artifact_manifest.txt`
- Aggregate CSV and LaTeX tables for metrics, risk metrics, topology-change lead time, dataset summary, network metrics, ablations, and statistical tests
- Generated PDF and PNG figures under `outputs/publication_compact/figures/`
- Per-seed tables, figures, and resume cache artefacts under `outputs/publication_compact/per_seed/`

### Consistency cleanup

- Controller-level connectivity, PDR, delay, reroute, DTN, and relay metrics are computed from the physical-radio snapshot graph.
- The manuscript graph definition distinguishes the geometric candidate set from the post-filter physical edge set.
- The lead-time artefacts and table labels describe topology-change lead time rather than direct fragmentation-onset lead.
- The public artefact bundle contains the CSV, JSON, LaTeX, figure, per-seed, resume-cache, and runtime-profile files used for the manuscript cross-check; transient rerun logs remain excluded.

### Reproducibility command

```bash
python main.py --config configs/publication_compact.json
```

To continue an interrupted run:

```bash
python main.py --config configs/publication_compact.json --resume
```

### Zenodo

This release is intended for Zenodo archival. The repository includes:

- `.zenodo.json` for Zenodo metadata
- `CITATION.cff` for citation metadata
- `docs/zenodo_release.md` with the Zenodo publication checklist

Zenodo archived DOI: <https://doi.org/10.5281/zenodo.20226053>
