# Final EAAI submission checklist

- [x] Format-normalized length verified: 17-page preprint and 10-page Elsevier two-column proof
- [x] Anonymous 17-page manuscript contains no author identity
- [x] Separate two-page title/declaration file exists
- [x] Twenty-seed primary study contains all configured seeds and current event metrics
- [x] Five-seed PyTorch neural-family aggregate exists
- [x] Six-horizon and full 2^3 factorial studies exist
- [x] Forestry, AERPAW, WiNES 60 GHz, and MILUV evidence layers exist with source provenance
- [x] Validation-selected operating points, packet-load study, and host-loop latency study exist
- [x] Offline dashboard exists and states that it is not a live digital twin
- [x] Current PDFs compile without undefined references or overfull boxes
- [x] Obsolete packages, audits, smoke outputs, raw ROS bags, caches, logs, and temporary renders removed
- [ ] Regenerate `anonymous_supplementary.zip` from current evidence
- [ ] Regenerate `submission_readiness_audit.json` after packaging
- [ ] Run the final test suite after package regeneration
- [ ] Author completes account-only attestations and conflict checks in the journal portal

Current repository status: the scientific outputs and PDFs are current. The previous anonymous ZIP and readiness JSON were deliberately removed because they predated the final event, MILUV, factorial, packet, latency, and manuscript revisions. Use `python scripts/build_anonymous_supplementary.py` and then `python scripts/audit_submission_readiness.py` to create current submission artifacts.
