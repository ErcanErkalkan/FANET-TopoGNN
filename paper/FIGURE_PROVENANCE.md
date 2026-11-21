# Source and interpretation of the current five figures

The figures are part of the **same existing development simulation**, not new experiments. Rebuild them from the frozen derived tables using `python paper/make_figures.py` (requires NumPy and Matplotlib). This command writes five vector-PDF figures and their 200-dpi PNG previews under `paper/manuscript/figures/`. Generated PDF metadata is fixed so the content hashes are stable in the pinned rendering environment.

| Figure | Type | Main evidence or source | Important limitation |
|---|---|---|---|
| Fig. 1. Warning pipeline | Schematic of frozen source-code behavior | `core/fanet/models.py`, `core/fanet/training.py`, `frozen/r9_9_calibration_alert.py`, and frozen protocol | Not a simulation trace or independent validation |
| Fig. 2. Endpoint versus onset | Explicitly conceptual sequence | Endpoint/onset definitions in `paper/manuscript/main.tex` | Invented didactic case; not claimed to come from a run |
| Fig. 3. Node count × mobility | Two numeric heatmaps | `data_derived/wn_r9_10_by_mobility_node_count.csv` | Same 45 groups and 180 paired development runs; not 9 independent cohorts |
| Fig. 4. Development block × mobility | Numeric heatmap | `data_derived/wn_r9_10_all_strata.csv`, rows with `scope=replicate_mobility` | Each cell 3 trajectory groups and 4 simulated minutes; budget is a stipulated analytical reference |
| Fig. 5. Lead and hypothetical latency | Bar and line charts | `data_derived/wn_r9_10_lead_distribution.csv`, `data_derived/wn_r9_10_action_latency_sensitivity.csv` | Timing sufficiency is not an intervention success rate |

The primary fixed threshold is 0.15. No model fitting, threshold search, event rematching, unblinding, new packet simulation, or route intervention is performed by the figure generator.

For technical checks run `python run.py check`, `python run.py verify` and `python run.py test` in the extracted canonical archive. Figure specifications and CSV invariants are checked in `audit/tests/test_current_figures.py`. Manuscript formatting is described only in `paper/manuscript/FORMAT_REQUIREMENTS.md`.
