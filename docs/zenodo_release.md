# Zenodo Release Guide

Repository: `ErcanErkalkan/FANET-TopoGNN`

Repository URL: <https://github.com/ErcanErkalkan/FANET-TopoGNN>

Archived release DOI: <https://doi.org/10.5281/zenodo.20226053>

This repository includes Zenodo metadata in `.zenodo.json` and citation metadata in `CITATION.cff`. The compact manuscript artefact bundle is tracked under `outputs/q1_publication_compact/` so GitHub releases and Zenodo archives include the CSV, JSON, LaTeX, per-seed, and figure files used for the manuscript cross-check. Transient rerun logs are intentionally excluded.

## Recommended Release

Use this release tag for the current compact benchmark bundle:

```bash
git tag -a v0.1.0-q1-compact -m "FANET-TopoGNN compact manuscript artefacts"
git push origin v0.1.0-q1-compact
```

Then create a GitHub release from that tag:

- Title: `FANET-TopoGNN compact manuscript artefacts`
- Target: `v0.1.0-q1-compact`
- Notes: `Includes the executable FANET-TopoGNN reproducibility suite and outputs/q1_publication_compact artefacts used for manuscript cross-checking.`

## Zenodo Steps

1. Sign in to Zenodo.
2. Open the GitHub integration page in Zenodo.
3. Authorize GitHub access if Zenodo asks for it.
4. Enable the repository `ErcanErkalkan/FANET-TopoGNN`.
5. Create the GitHub release from `v0.1.0-q1-compact`.
6. Wait for Zenodo to archive the release and mint a DOI.
7. Record the DOI in the manuscript data-availability statement and repository citation metadata.

Current DOI: <https://doi.org/10.5281/zenodo.20226053>

Official Zenodo documentation:

- <https://help.zenodo.org/docs/github/>
- <https://help.zenodo.org/docs/github/enable-repository/>
- <https://help.zenodo.org/docs/github/archive-software/github-upload/>

## After DOI Minting

Use the following manuscript text with the final DOI:

```latex
All data and code are openly available from the GitHub repository
\path{ErcanErkalkan/FANET-TopoGNN} and the archived Zenodo release
\url{https://doi.org/10.5281/zenodo.20226053}.
```

If Zenodo does not import all metadata automatically, edit the draft record and copy values from `.zenodo.json`.
