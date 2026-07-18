# Supplementary Material S1: detailed table index

The manuscript intentionally keeps eight decision-critical tables in the main text. The
following detailed tables and machine-readable sources are supplied in Supplementary
Material S1 (`EAAI_anonymous_supplementary.zip`).

| Main-text topic | Detailed S1 artefact |
|---|---|
| Earlier 20-seed benchmark | `tables/detailed/confirmatory_metrics_table.tex`; `outputs/paper_like_submission/seed_summary.csv` |
| Complete five-seed neural model family | `tables/detailed/neural_seed_extension_full_table.tex`; `outputs/publication_neural_5seed_extension/neural_seed_extension_full.csv` |
| Run-block feature attribution | `tables/detailed/explainability_table.tex`; companion files in `outputs/publication_neural_5seed_extension/` |
| Forecast-horizon sweep | `tables/detailed/horizon_sweep_table.tex`; `outputs/horizon_sweep/horizon_sweep_summary.csv` |
| Event window/refractory/density sensitivity | `outputs/event_protocol_sensitivity/event_protocol_sensitivity_table.tex`; companion CSV, figure, and protocol JSON |
| Forestry-motion transfer | `tables/detailed/external_metrics_table.tex`; `outputs/external_validation/` |
| AERPAW cellular transfer | `tables/detailed/aerpaw_cellular_table.tex`; `outputs/aerpaw_cellular_validation/` |
| 60 GHz peer-link transfer | `tables/detailed/uav_to_uav_mmwave_table.tex`; `outputs/uav_to_uav_mmwave_validation/` |
| MILUV zero-shot and adaptation | `tables/detailed/miluv_validation_table.tex`; `tables/detailed/miluv_adaptation_table.tex`; `outputs/miluv_validation/` |
| Operating policies | `tables/detailed/operating_point_table.tex`; `outputs/operating_point/` |
| One-step packet-load sweep | `tables/detailed/packet_level_table.tex`; `outputs/packet_level_controller/packet_level_summary.csv` |

The five-seed PyTorch extension is descriptive model-family context, not confirmatory
evidence. A 20-new-seed neural extension must be run only in an environment with a verified
PyTorch backend; no surrogate or fabricated neural results are included.
