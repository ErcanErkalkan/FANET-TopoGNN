# Engineering Applications of Artificial Intelligence portal metadata

Synchronized with the manuscript source, title page, route decision, citation metadata,
and submission text audit on 2026-07-18.

## Submission identity

- Journal: Engineering Applications of Artificial Intelligence
- Article type: Original research article
- Manuscript title: A Reproducible Benchmark and Cross-Domain Stress Test for Fragmentation-Event Forecasting in Unmanned Aerial Vehicle Networks
- Running topic: Reproducible stress testing of fragmentation-event forecasting
- Language: English

## Author and affiliation

- Given name: Ercan
- Family name: Erkalkan
- Corresponding author: Yes
- E-mail: ercan.erkalkan@marmara.edu.tr
- ORCID: 0000-0001-9259-7112
- Institution: Marmara University
- Unit: Vocational School of Technical Sciences, Department of Electronics and Automation, Artificial Intelligence Operator Program
- Address: Mehmet Genc Campus, 34865 Kartal Istanbul, Turkiye

The author must use the portal's ORCID authorization control if account-level
confirmation is requested.

## Abstract

Rapid motion and correlated radio variation can fragment an unmanned aerial vehicle (UAV) network before reactive monitoring leaves time for a safe response. We test whether short-horizon warnings remain useful when false alerts, packet traffic, controller cost, domain shift, and computation are considered jointly. The artificial-intelligence contribution is a leakage-controlled benchmark of current-state, topological, kinematic, source-gated, shallow, and neural predictors with a locked 20-seed evaluation. The engineering application couples validation-selected warnings to bounded simulated relay and packet-event models; it is not field control. Under the prespecified multiplicity-controlled rule, Source-Gated Kinetic-TopoGuard does not outperform Current-state ExtraTrees: the paired event F1 difference is -0.0215 (seed-bootstrap 95% confidence interval [-0.0417, -0.0020]; Holm-adjusted permutation p = 0.1030), while false alerts increase by 2.10 events per minute. In the 20-seed equal-learner factorial, adding graph, topology, and kinematic sources reduces event F1 by 0.0426, although calibration improves. The residual count branch selects zero scale in 95% of seeds, and none of 360 paired closed-loop comparisons satisfies the predeclared engineering-benefit rule. The single compatible measured three-vehicle ultra-wideband sequence contains only one or two fragmentation events per threshold; all frozen and adapted models yield zero event F1 at the primary threshold. We treat this as a sparse stress test, not strong external validation. Measured host-loop latency excludes sensing, telemetry, autopilot communication, and actuation. The benchmark exposes where richer predictors fail to yield engineering benefit.

## Keywords

1. Fragmentation-event forecasting
2. Unmanned aerial vehicle networks
3. Reproducible benchmarking
4. Cross-domain stress testing
5. Network resilience
6. Engineering decision support

## INSPEC classification (maximum six)

Use these five standard INSPEC outline codes; do not add free-form or fabricated
subcodes if the portal offers the official controlled vocabulary.

1. B60 — Communications
2. C10 — Systems and control theory
3. C30 — Control technology
4. C40 — Numerical analysis and theoretical computer topics
5. C70 — Computer applications

## Highlights

Upload the editable `EAAI_HIGHLIGHTS.txt` file. Its count, character lengths,
acronyms, and equality with the manuscript highlights are checked by
`scripts/validate_submission_text.py`.

## Declarations

- Funding: This research received no specific grant from funding agencies in the public, commercial, or not-for-profit sectors.
- Competing interests: The author declares that there are no known competing financial interests or personal relationships that could have influenced the work reported in this article.
- Separate competing-interest file: `submission/EAAI_Declaration_of_Competing_Interest.docx`
- Human participants: Not applicable.
- Animal subjects: Not applicable.
- Informed consent: Not applicable.
- CRediT: Ercan Erkalkan - Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Visualization, Writing - original draft, Writing - review and editing.
- Generative AI disclosure: During manuscript preparation, the author used OpenAI GPT-based tooling for language refinement, software-development assistance, and local reproducibility cross-checks. The author reviewed and verified all resulting manuscript and code changes, executed the reported analyses, and takes responsibility for the submitted work.
- Graphical abstract provenance: The graphical abstract was assembled programmatically from project results with Matplotlib and vector primitives. No generative-AI or AI-assisted image-generation or image-alteration tool was used for this artwork.

## Data and code availability

Code, configurations, derived external-validation data, and generated artifacts
are available from `https://github.com/ErcanErkalkan/FANET-TopoGNN`. The archived
software record `https://doi.org/10.5281/zenodo.20226053` identifies archived release
`v0.1.0-q1-compact`; the current working-tree Python package version is `0.1.12`.
These scopes must not be reported as the same release. The source forestry flight
dataset is available under CC BY 4.0 at
`https://doi.org/10.5281/zenodo.14701641`. The MILUV dataset is available at
`https://doi.org/10.25452/figshare.plus.28386041.v1`. AERPAW Dataset 22/23 and
the WiNES UAV-to-UAV 60 GHz channel data are identified in the manuscript and
machine-readable protocols.

For double-anonymous review, the manuscript uses a generic masked availability
statement. Do not place the public repository, software DOI, title page, author
metadata, or identified declarations in anonymous-review fields.

## File-to-field map

- Main manuscript: `paper/main_anonymized.pdf`
- Title page and declarations: `paper/title_page.pdf`
- Editable highlights: `EAAI_HIGHLIGHTS.txt`
- Supplementary Material S1: `submission/EAAI_anonymous_supplementary.zip`
- Declaration of Competing Interest: `submission/EAAI_Declaration_of_Competing_Interest.docx`
- Cover letter: `EAAI_COVER_LETTER.md`
- Graphical abstract: `paper/figures/plantuml/graphical_abstract.pdf`
- Graphical abstract provenance note: `submission/GRAPHICAL_ABSTRACT_PROVENANCE.txt`
- Named figure files: `submission/EAAI_Figure_Files.zip`
- Source manuscript, only if requested: `paper/main.tex` and its dependencies

## Portal-only attestations

- Confirm that the work is original, approved by the author, and not under consideration elsewhere.
- Authorize the ORCID link through the signed-in account.
- Select classifications from the portal's current controlled vocabulary.
- Check reviewer conflicts personally before entering any nomination.
- Keep identified metadata out of anonymous-review upload fields.
