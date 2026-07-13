# Accepted-Article Gap Closure

Reference: `1-s2.0-S0952197626017422-main.pdf`, *Engineering Applications of Artificial Intelligence* 181 (2026) 115458.

## Format-normalized comparison

- Accepted bridge article: 13 publisher-formatted two-column pages and 10,931 extracted words.
- Current FANET manuscript: 17 preprint pages and 5,604 extracted words; the anonymous version is 17 pages and 5,581 words.
- The same source compiled with Elsevier `5p,times,twocolumn` is 10 PDF pages and 5,686 extracted words, about 52% of the accepted article's extracted word count.
- The defensible comparison is therefore 10 versus 13 two-column PDF pages, not 17 versus 13. The manuscript is not overlong after format normalization.

## Closed evidence gaps

| Gap | Executed resolution | Current evidence |
| --- | --- | --- |
| Fragmentation-event semantics | Connected-to-fragmented transitions, one-to-one alert matching, missed events, and false events per minute | `fanet/evaluation.py`; 20-seed event metrics |
| Radio realism | Run-persistent hardware offsets and temporally correlated shadowing/fading with paired policy realizations | `fanet/radio.py`; current v6 result caches |
| Physical controller limits | Relay speed, acceleration, and link-budget constraints; analytical reachability proxy separated from packet PDR | Controller outputs and SimPy packet study |
| Primary statistical evidence | Twenty independent seeds and run-clustered confidence intervals | 432,000 snapshots in `outputs/paper_like_submission/` |
| Model-family context | Five-seed PyTorch execution for all reported neural families | `outputs/publication_neural_5seed_extension/` |
| Horizon dependence | Six executed horizons from 0.2 to 2.0 s | `outputs/horizon_sweep/` |
| Feature attribution | Full 2^3 graph/topology/kinematic factorial with the same learner | `outputs/factorial_feature_ablation/` |
| Real-data transfer | Forestry motion, AERPAW cellular, WiNES 60 GHz peer-link, and MILUV three-UAV UWB topology studies | Corresponding output directories and source manifests |
| Operational validation | Validation-selected false-event budgets, packet-load sweep, and complete host-loop timing | `outputs/operating_point/`, `outputs/packet_level_controller/`, `outputs/end_to_end_latency/` |
| Integrated artifact | Offline evidence replay with explicit non-live scope | `outputs/digital_twin_dashboard/` |

## Current quantitative interpretation

- Twenty-seed confirmation: current-state persistence has the lowest count MAE (0.132); Kinetic-TopoGuard has MAE 0.133 and event F1 0.483, versus shallow ML MAE 0.204 and event F1 0.414. The claim is improved warning trade-off, not best point-count accuracy.
- Validation-selected operating policies reach event F1 0.421 at 1.85 false events/min and event F1 0.351 at 0.90 false events/min.
- The five-seed extension gives Kinetic-TopoGuard MAE 0.120 and event F1 0.470. TGN reaches event F1 0.464 but produces 6.39 false events/min.
- The 60 GHz held-out peer-link model improves link F1 from 0.731 to 0.923 and PR-AUC to 0.985. Its use on forestry motion is a transported sensitivity, not same-site RF calibration.
- AERPAW supports chronological LTE link-state transfer, but throughput transfer is negative: robust splits have negative R2.
- MILUV supplies measured three-UAV motion and UWB topology. At the primary threshold Kinetic-TopoGuard event F1 is 0.08 and shallow ML event F1 is 0.40, so neither transfer is deployment-ready.
- Packet simulation shows no consistent Kinetic-TopoGuard PDR advantage. Complete host-loop P95 latency is 25.68--30.87 ms, so the study does not claim a 20 ms tail bound.

## Remaining scientific boundary

No retained dataset provides synchronized outdoor multi-UAV motion, inter-UAV RF, MAC contention, routing, IP packet delivery, onboard timing, and closed-loop actuation in one campaign. The repository therefore does not claim field IP-PDR, hardware-in-the-loop control, onboard-runtime validation, safe autonomous intervention, or deployment readiness. These are physical-study requirements, not gaps that can be closed by prose.

## Current artifacts

- `paper/main.pdf`: identified 17-page preprint.
- `paper/main_anonymized.pdf`: anonymous 17-page manuscript.
- `paper/main_5p.pdf`: 10-page two-column comparison proof.
- `paper/title_page.pdf`: separate two-page author/declaration file.
- `scripts/build_anonymous_supplementary.py`: regenerates the anonymous package from current outputs.
- `scripts/audit_submission_readiness.py`: regenerates the final readiness report after packaging.
