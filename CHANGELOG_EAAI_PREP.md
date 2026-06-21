# EAAI submission preparation changelog

## 1. Completed evidence work

- Completed the focused 20-seed primary comparison with 20 unique seeds and 432,000 simulated snapshots.
- Verified the compact neural-family rows through a fresh PyTorch 2.11.0 CPU rerun; all nine neural task entries record `pytorch` and `surrogate_used=false`.
- Added an external transfer study based on synchronized real three-UAV forestry flight motion from the public dataset at DOI `10.5281/zenodo.14701641`.
- Archived source file sizes and MD5 checksums, the derived-trace SHA-256, 107,730 prediction rows, raw and physically clipped metrics, and the external protocol.
- Kept the external-validity boundary explicit: the public bags provide measured motion but no measured RF, packet reception, or medium-access ground truth.

## 2. Manuscript and bibliography

- Rebuilt the anonymous and identified manuscripts at 49 pages, below the 50-page submission limit.
- Added focused 20-seed results, paired statistical tests, real-motion transfer results, provenance statements, and the revised evidence boundary.
- Kept Union-Find labeled as a current-graph diagnostic rather than a learned forecasting model.
- Repaired the PersLay entry from the official PMLR AISTATS record, including proceedings title, volume, editors, publisher, pages 2786-2796, year, and URL.
- Added the external dataset citation from the official Zenodo record.
- Updated the title page, CRediT statement, funding, competing-interest, data/code, and AI-use declarations.

## 3. Reproducibility and packaging

- Added smoke, 20-seed, provenance-promotion, external-download, trace-extraction, external-evaluation, table-generation, readiness-audit, and anonymous-package launchers.
- Added `pytest.ini` so the repository test suite does not collect duplicated tests from the generated supplementary directory.
- Added an allowlist-based anonymous package builder with workspace path guards, identity scanning, evidence checks, PDF page-count validation, manifest hashing, and ZIP CRC validation.
- Rebuilt `anonymous_supplementary.zip` with 561 files (5.79 MiB), including the current 49-page anonymous manuscript, all 20 seed outputs, derived public flight trace, external predictions, and compact provenance records.
- Excluded raw ROS bags, resume caches, bytecode, LaTeX build artefacts, Git metadata, title-page files, personal metadata, and the identity-bearing public compact README.

## 4. Verification completed

```text
python -m compileall -q fanet scripts tests main.py
python -m pytest -q --basetemp .pytest_tmp_final_root
python scripts/audit_submission_readiness.py
python scripts/build_anonymous_supplementary.py
pdflatex / bibtex rebuilds for main, main_anonymized, and title_page
pdfinfo, pdftoppm/contact-sheet visual QA, pdftotext identity scans
ZIP entry, forbidden-file, manifest, and CRC checks
```

- Full tests: 12 passed; only two Matplotlib API deprecation warnings remain.
- Submission-readiness audit: all checks passed.
- Anonymous PDF: 49 pages; identity scan passed.
- Identified manuscript: 49 pages; title page: 2 pages.
- All rendered anonymous manuscript and title-page contact sheets were visually inspected; no clipping or unresolved references were found.
- Anonymous ZIP: 561 entries, no raw bags/resume caches/bytecode/build logs, CRC passed.

## 5. Scientific boundary

The completed evidence now has three layers: a three-seed compact neural-family benchmark, a focused 20-seed primary learned comparison, and a real-field-motion external transfer test. It does not claim measured wireless-link, packet-level, hardware-in-the-loop, onboard-runtime, or deployment validation because those labels are absent from the public flight source.

## 6. Portal-only tasks

- Upload `paper/main_anonymized.pdf`, `anonymous_supplementary.zip`, and `paper/title_page.pdf` to the correct separate portal fields.
- Use `EAAI_PORTAL_METADATA.md` and `EAAI_COVER_LETTER.md` for the prepared metadata and cover letter.
- The corresponding author must personally confirm originality, concurrent-submission, competing-interest, authorship, conflict, and reviewer-related attestations in the authenticated journal portal.

## Readiness

**READY FOR AUTHOR PORTAL ATTESTATIONS** - all repository-completable evidence, provenance, bibliography, metadata, build, anonymity, test, and packaging work is complete. Only authenticated account actions and author attestations remain.
