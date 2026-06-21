# Engineering Applications of Artificial Intelligence portal metadata

Reviewed against the manuscript, title page, CFF, Zenodo metadata, official
ORCID record, and Elsevier highlight guidance on 2026-06-21.

## Submission identity

- Journal: Engineering Applications of Artificial Intelligence
- Article type: Original research article (select the portal's equivalent standard research-paper label)
- Manuscript title: Motion-Conditioned Topological Artificial Intelligence for Fragmentation-Risk Forecasting in Unmanned Aerial Vehicle Networks
- Running topic: Topological artificial intelligence for UAV-network fragmentation warning
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

The ORCID public record returns `ERCAN ERKALKAN` and the same verified primary
institutional e-mail. The author must use the portal's ORCID sign-in/authorize
button if it requests account-level confirmation.

## Abstract

Unmanned aerial vehicle (UAV) networks can fragment within short control
horizons because motion changes pairwise distance, radio margin, and multi-hop
component structure. This paper proposes Kinetic-TopoGuard, a motion-conditioned
topological artificial intelligence framework that combines horizon projection,
zero-dimensional persistence-image descriptors, residual forecasting, and
calibrated risk scoring. The method forecasts the future number of connected
components and converts that estimate into an interpretable fragmentation
warning for relay, routing, and delay-tolerant connectivity management. A
verified PyTorch compact benchmark evaluates tabular, graph, topological, and
temporal baselines under fixed/adaptive topology control and nominal/degraded
radio conditions. In a focused 20-seed simulation, Kinetic-TopoGuard reduces
mean absolute error from 0.363 to 0.308 and raises fragmentation-risk F1 from
0.503 to 0.587 relative to the validation-selected shallow comparator. Transfer
to a public 399-second three-UAV forestry flight trace yields mean absolute
errors of 0.152-0.219 across three predeclared communication-radius
sensitivities, compared with 0.709-1.542 for the shallow comparator. The field
dataset contains motion but no radio or packet ground truth; consequently, the
external study supports mobility-source transfer, not field-measured
wireless-network or deployment readiness.

## Keywords

1. Topological artificial intelligence
2. Persistent homology
3. Unmanned aerial vehicle networks
4. Fragmentation-risk forecasting
5. Calibration
6. Engineering decision support

## Highlights

1. Topological artificial intelligence forecasts fragmentation risk.
2. Motion projection improves short-horizon topology encoding.
3. Twenty seeds support the primary forecasting comparison.
4. Real three-drone motion provides an external transfer test.
5. Backend records verify the implemented neural comparisons.

All five highlights are at most 85 characters and contain no undefined acronym.

## Declarations

- Funding: This research received no specific grant from funding agencies in the public, commercial, or not-for-profit sectors.
- Competing interests: The author declares that there are no known competing financial interests or personal relationships that could have influenced the work reported in this article.
- Human participants: Not applicable.
- Animal subjects: Not applicable.
- Informed consent: Not applicable.
- CRediT: Ercan Erkalkan - Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing - original draft, Writing - review and editing, Visualization.
- Generative AI disclosure: During manuscript preparation, the author used OpenAI GPT-based tooling for language refinement and local reproducibility cross-checks. The author reviewed and verified all resulting manuscript and code changes and takes responsibility for the submitted work.

## Data and code availability

Code, configurations, derived external-validation data, and generated artifacts
are available from `https://github.com/ErcanErkalkan/FANET-TopoGNN` and the
archived release `https://doi.org/10.5281/zenodo.20226053`. The source forestry
flight dataset is available under CC BY 4.0 at
`https://doi.org/10.5281/zenodo.14701641`. For double-anonymous review, use the
identity-scanned supplementary ZIP and do not expose the public repository or
software DOI in the anonymous manuscript fields.

## File-to-field map

- Main manuscript: `paper/main_anonymized.pdf`
- Title page / author information: `paper/title_page.pdf`
- Supplementary material: `anonymous_supplementary.zip`
- Cover letter: `EAAI_COVER_LETTER.md`
- Graphical abstract: `paper/figures/plantuml/graphical_abstract.pdf`
- Source manuscript, only if requested: `paper/main.tex` plus its figure/table dependencies

## Portal-only attestations

- Confirm that the work is original, not under consideration elsewhere, and approved by the author.
- Authorize the ORCID link through the signed-in account.
- Select classifications from the portal's current controlled vocabulary.
- Nominate reviewers only after the author personally checks recent collaboration, institutional, supervisory, and personal conflicts. Reviewer identities are intentionally not fabricated by the package.
- Keep the title page and public repository/DOI metadata out of anonymous-review upload fields.
