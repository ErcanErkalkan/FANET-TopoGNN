# Zenodo Release Guide

Repository: `ErcanErkalkan/FANET-TopoGNN`

Repository URL: <https://github.com/ErcanErkalkan/FANET-TopoGNN>

Archived release DOI: <https://doi.org/10.5281/zenodo.20226053>

That DOI identifies `v0.1.0-q1-compact` (published 2026-05-16). The repository
working-tree Python package version is `0.1.12`; it is not archived by the existing
DOI.

This repository includes Zenodo metadata in `.zenodo.json` and citation metadata in `CITATION.cff`. The compact manuscript artefact bundle is tracked under `outputs/publication_compact/` so GitHub releases and Zenodo archives include the CSV, JSON, LaTeX, per-seed, and figure files used for the manuscript cross-check. Transient rerun logs are intentionally excluded.

## Recommended Release

Use this release tag only when publishing a new Zenodo version for the current compact benchmark bundle:

```bash
git tag -a v0.1.12 -m "FANET-TopoGNN v0.1.12 publication consistency artefacts"
git push origin v0.1.12
```

Then create a GitHub release from that tag:

- Title: `FANET-TopoGNN v0.1.12 publication consistency artefacts`
- Target: `v0.1.12`
- Notes: `Includes the executable FANET-TopoGNN reproducibility suite and the cross-checked outputs/publication_compact artefacts used for manuscript consistency validation.`

## Zenodo Steps

1. Sign in to Zenodo.
2. Open the GitHub integration page in Zenodo.
3. Authorize GitHub access if Zenodo asks for it.
4. Enable the repository `ErcanErkalkan/FANET-TopoGNN`.
5. Create the GitHub release from `v0.1.12`.
6. Wait for Zenodo to archive the release and mint a DOI.
7. Verify that Zenodo identifies the new version as `v0.1.12`.
8. Only then record the newly minted version DOI in manuscript and citation metadata.

Existing earlier-version DOI: <https://doi.org/10.5281/zenodo.20226053>

Official Zenodo documentation:

- <https://help.zenodo.org/docs/github/>
- <https://help.zenodo.org/docs/github/enable-repository/>
- <https://help.zenodo.org/docs/github/archive-software/github-upload/>

## After DOI Minting

Use the following manuscript text with the final DOI:

```latex
All data and code are openly available from the GitHub repository
\path{ErcanErkalkan/FANET-TopoGNN}. The archived snapshot
\texttt{v0.1.0-q1-compact} is available at
\url{https://doi.org/10.5281/zenodo.20226053}; later working-tree changes are not
contained in that record.
```

If Zenodo does not import all metadata automatically, edit the draft record and copy values from `.zenodo.json`.
