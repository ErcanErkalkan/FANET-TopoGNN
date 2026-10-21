# Step 1: versioned strict full-horizon label integrity (21 October 2026)

## Scope and evidence boundary
This branch is a candidate patch. It does not alter or supersede the locked EAAI
20-seed hypothesis, historical R1-R5 result files, or the dated R2 600/600
**coverage-only** log audit. No new model-superiority or field-PDR result is
claimed. Clean environment execution and comparison-statistic rechecks remain OPEN.

## Failure mechanism and remedy
Legacy simulation records use `min(t+h, final_index)`; terminal observations
therefore have fewer than h actual future ticks but are assigned a purported
h-horizon label. A new opt-in sim field `horizon_label_policy` controls behavior:
- `legacy_clamp` (default): reproduce prior output semantics; historical only.
- `strict_full_horizon`: drop right-censored rows that have no t+h graph.
  Require positive h and exact timestamp difference h.

`relabel_forecast_horizon(..., strict_full_horizon=True)` applies the same
rule to FULL, UNTRUNCATED trajectory sequences, using exact time keys.
Do not use previously strict-truncated sequences as multi-horizon sources.

New experiments MUST set the strict policy explicitly and use a distinct
`experiment_name` and `output_dir`. The generated summary records the
effective policy. Existing historical caches and CSVs must not be overwritten.

## Executable checks
```bash
python -m pip install -e .
python -m pip install pytest
pytest -q tests/test_strict_horizon.py
python scripts/audit_strict_horizon.py --config configs/strict_v4_smoke.json --seed 7
python main.py --config configs/strict_v4_smoke.json
```
Expected smoke dataset: 3 runs x (12-2) = 30 fully labeled rows; the auditor
must compare target beta0 with the original complete t+h simulation state,
check exact time indices, labels, run IDs and row conservation.

## Release gates after Step 1 code review
1. Execute the regression tests and audit in a clean pinned environment.
2. R1: freeze the strict-v3/v4 label contract, compare full numerical replay
   and verify code/data dependencies across all 360 cases.
3. R2: collect per-model metric JSON for 600 records, verify 120 full cells,
   recompute seed-paired tests and trace all runs to one commit/environment.
4. R3: recompute event-based scores and censored evaluation intervals.
   Tail suppression can also censor end-of-run onset events; report this.
5. R4: retrain and rerun every selected neural family on homogeneous labels,
   verifying actual PyTorch execution rather than surrogate placeholders.
6. R5: rerun stateful relay and packet tests with frozen corrected forecasts
   and unchanged paired traffic/channel realizations.
7. Keep previous locked metrics and prior DOI intact. Create a NEW archive
   only after complete evidence and anonymity/provenance audit clearance.

**Release decision: HOLD** until each applicable release gate is satisfied.
