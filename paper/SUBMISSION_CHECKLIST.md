# Submission Checklist

## Closed in manuscript

- Title reframed around applied AI and UAV engineering decision support.
- Abstract explicitly separates the AI contribution from the engineering application.
- Undefined title/abstract abbreviations removed or expanded.
- Highlights use five items and stay within the 85-character limit.
- Venue-specific metadata was removed for a neutral manuscript package.
- Graphical abstract regenerated as a vector workflow graphic, not a generated bitmap image.
- Double-anonymized entry point added: `main_anonymized.tex`.
- Separate title page added: `title_page.tex`.
- Data/code availability now has anonymized-submission and public-release wording.

## Closed in code and artefact pipeline

- Added expanded 20-seed submission profile: `configs/paper_like_submission.json`.
- Kept checklist claims aligned with the implemented compact release: Shallow ML, GCN, GAT, GraphSAGE, PI+MLP, FANET-TopoGNN variants, T-GCN, STGCN, TGN, union--find, and Kinetic-TopoGuard; the optional heuristic implementation is not reported unless a selected configuration exports its artefacts. The shallow selector is documented as exactly linear regression, ridge regression, random forest, gradient boosting, and MLP; additional shallow families are not claimed without matching code, configuration, and output artefacts.
- Documented the environment split: the base stack supports non-deep runs and surrogate checks; `requirements-deep.txt` / `.[deep]` is required before neural rows are interpreted as actual PyTorch baselines.
- Removed checklist items for baselines, calibration metrics, diagnostic curves, and utility outputs that are not exported by the released compact artefacts.

## Required before manuscript package finalization

- Install the deep-learning dependencies and run the expanded 20-seed submission profile:

```powershell
python -m pip install -r requirements-deep.txt
python main.py --config configs/paper_like_submission.json --resume
```

- Sync the generated artefacts into the manuscript:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\sync_manuscript_assets.ps1 -Profile paper_like_submission -ManuscriptDir paper
```

- Compile `main_anonymized.tex` for the blinded manuscript.
- Compile `title_page.tex` only if a separate title page is required.
- Verify the anonymized PDF contains no author name, affiliation, personal GitHub handle, acknowledgements, or direct DOI that identifies the author during anonymized evaluation.
- Archive the final public benchmark artefact after formal release, or provide an anonymized package when blind evaluation is required.
