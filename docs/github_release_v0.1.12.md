# DRAFT: FANET-TopoGNN v0.1.12 publication consistency artefacts

This is a release draft, not an archived release record. DOI
`10.5281/zenodo.20226053` identifies `v0.1.0-q1-compact` and must not be cited as
the archive of `v0.1.12`.

## Release fields

Tag: `v0.1.12`

Target: `main`

Release title: `FANET-TopoGNN v0.1.12 publication consistency artefacts`

Pre-release: unchecked

Set as latest release: checked

## Release notes

This draft describes a future release of the executable FANET-TopoGNN / Kinetic-TopoGuard reproducibility suite and the compact manuscript-facing artefact bundle after consistency cleanup.

Repository: `ErcanErkalkan/FANET-TopoGNN`

Repository URL: <https://github.com/ErcanErkalkan/FANET-TopoGNN>

Existing earlier archive DOI (`v0.1.0-q1-compact`): <https://doi.org/10.5281/zenodo.20226053>

### Included artefacts

- `outputs/publication_compact/summary.json`
- `outputs/publication_compact/runtime_profile.json` (separates author-recorded runtime environment from the reference reproducibility environment)
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
- The binary risk label has exact-horizon semantics (`frag_at_horizon`): it records whether `beta0(t+h) > 1`, not whether fragmentation occurred at any intermediate tick in the horizon window.
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

This release requires a new Zenodo version before it can be described as archived. The repository includes:

- `.zenodo.json` for Zenodo metadata
- `CITATION.cff` for citation metadata
- `docs/zenodo_release.md` with the Zenodo publication checklist

Existing Zenodo DOI for `v0.1.0-q1-compact`: <https://doi.org/10.5281/zenodo.20226053>

Do not attach that DOI to `v0.1.12`; record the new version DOI only after Zenodo mints it.
