# Expanded experiment rerun plan

The current compact manuscript artefacts report the benchmark with three random seeds. This is enough for reproducibility checking and debugging, but not enough for strong inferential claims. No additional seed results are invented in the manuscript. The following rerun is required before replacing the preliminary compact tables with expanded-profile evidence.

## Minimum rerun

- Config: `configs/paper_like_submission.json`.
- Seeds: 20 independent random seeds.
- Keep all existing factors unchanged:
  - UAV counts: 10, 20, 30
  - Mobility models: Random Waypoint, Gauss-Markov, task-oriented mission mobility
  - Graph policies: fixed and adaptive
  - Radio scenarios: nominal and degraded
  - Forecast horizon: current 6-step horizon
  - Models: union-find, shallow ML, Kinetic-TopoGuard, GCN, GAT, GraphSAGE, PI+MLP, FANET-TopoGNN variants, T-GCN, STGCN, and TGN
  - Shallow candidates: linear regression, ridge regression, random forest, gradient boosting, and MLP only

Run from the repository root after installing the deep-learning dependencies, because this profile includes neural GNN and temporal baselines:

```powershell
python -m pip install -r requirements-deep.txt
python main.py --config configs/paper_like_submission.json --resume
```

A base-environment run remains useful as a smoke or surrogate check, but neural rows from such a run must not be reported as actual PyTorch GNN or temporal-neural results.

Then sync manuscript assets:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\sync_manuscript_assets.ps1 -Profile paper_like_submission -ManuscriptDir paper
```

## Required additional analysis

1. **Expanded confidence intervals**
   - Report mean +/- 95% CI across seeds.
   - Report median and IQR where distributions are non-Gaussian.
   - Avoid strong claims when confidence intervals overlap.

2. **Lead-time saturation repair**
   - Add horizon sweep: 2, 4, 6, 8, 10 steps.
   - Report whether median lead remains saturated at each horizon.
   - Add lead-time distribution plots instead of only median lead.

3. **Useful-warning metrics**
   - Precision-adjusted useful lead at fixed warning precision.
   - False-alarm-adjusted lead.
   - Lead-time versus precision/recall/F1 trade-off curves.
   - Controller-action rate versus warning precision.
   - Optional future calibration and threshold diagnostics only if their code paths and exported artefacts are added and verified before manuscript use.

4. **External validity extension**
   - Add at least one of the following before strong deployment claims:
     - real UAV flight traces,
     - hardware-in-the-loop traces,
     - packet-level ns-3 or OMNeT++ coupled simulation,
     - activated SINR/interference sensitivity benchmark.

5. **Ablation extension**
   - Equal-learner factorial ablation:
     - kinematic only,
     - graph-statistical only,
     - persistence-image only,
     - kinematic + graph-statistical,
     - kinematic + persistence-image,
     - graph-statistical + persistence-image,
     - full model.

6. **Baseline extension**
   - Report only tabular, temporal, or heuristic baselines that are implemented and exported by the selected configuration.
   - Add any new booster, regularised-linear, or sequence baseline to the manuscript only after its code, configuration entry, and output artefacts are present in the release.
   - Keep CatBoost optional unless the dependency is installed and archived in the environment manifest.

## Reporting rule

Only replace the manuscript tables after the expanded 20-seed rerun has completed and the exported logs, CSV files, LaTeX tables, summary JSON, and figure files actually produced by the selected profile have been verified. If the rerun is not completed, keep the current wording that treats the three-seed benchmark as preliminary compact-benchmark evidence.
