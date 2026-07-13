# Engineering Applications of Artificial Intelligence portal metadata

Reviewed against the manuscript, title page, CFF, Zenodo metadata, official
ORCID record, and Elsevier highlight guidance on 2026-07-01.

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

Rapid motion and correlated radio variation can partition a flying ad hoc
network before a reactive connectivity monitor leaves enough time for
intervention. This paper presents Kinetic-TopoGuard, a short-horizon predictor
of the future number of connected components and of connected-to-fragmented
transition risk. The method combines current graph state, velocity-projected
link margins, zero-dimensional persistence images, graph summaries, residual
regression, and validation-selected risk thresholds. At the 0.6-second horizon,
Kinetic-TopoGuard does not improve count MAE over persistence (0.133 versus
0.132), but it yields event F1=0.483 with 3.53 false events/minute, compared
with 0.414 and 13.24 for shallow ML. AERPAW chronological LTE link-state F1
rises from 0.828/0.679 to 0.961/0.980, while robust throughput tests are
negative. A measured 60 GHz peer-link model raises held-out link-viability F1
from 0.731 to 0.923. Direct MILUV three-UAV UWB topology transfer exposes
substantial domain shift: Kinetic-TopoGuard event F1 is 0.08 at the primary
threshold. The study does not claim a live digital twin, measured outdoor
FANET IP packet delivery, or field deployment.

## Keywords

1. Topological artificial intelligence
2. Persistent homology
3. Unmanned aerial vehicle networks
4. Fragmentation-risk forecasting
5. Calibration
6. Engineering decision support

## Highlights

1. Fragmentation transitions are scored with one-to-one event matching.
2. Motion, topology, and graph sources are isolated in a full factorial.
3. Twenty seeds and six horizons test the primary forecasting claim.
4. Measured three-UAV UWB topology provides direct multi-UAV transfer evidence.
5. Packet, latency, and physical relay constraints bound operational claims.

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
`https://doi.org/10.5281/zenodo.14701641`. MILUV is available at
`https://doi.org/10.25452/figshare.plus.28386041.v1`; AERPAW and WiNES source
records are listed in the manuscript and supplementary protocols. For
double-anonymous review, use the
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
