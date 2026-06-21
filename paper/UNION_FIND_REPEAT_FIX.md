# Union-Find Repetition Fix Report

## Goal
Reduce repeated explanations of the Union-Find reference and remove defensive repetition from captions, results discussion, and conclusion while preserving the necessary methodological clarification in one central place.

## Main edits
- Kept the full Union-Find clarification only in the method subsection: `Union-find current-graph upper-bound reference`.
- Removed Union-Find/oracle wording from the abstract and research question.
- Shortened the empirical contribution item so it no longer repeats the Union-Find explanation.
- Removed the defensive `Metric-specific interpretation after claim rebalancing` table.
- Shortened the scalar performance discussion after the generated tables.
- Removed the long Union-Find explanation from the MAE figure caption.
- Removed the Union-Find explanation from the lead-time paragraph.
- Removed the Union-Find explanation from the risk-F1 paragraph and risk-F1 figure caption.
- Shortened the discussion and conclusion so they no longer repeat the same Union-Find warning.
- Shortened generated table captions and renamed table rows to `Union-Find reference` without explanatory caption text.

## Phrase-count reduction

### `main.tex`
| Phrase | Before | After |
|---|---:|---:|
| `union--find` | 13 | 3 |
| `current-graph` | 13 | 2 |
| `upper-bound` | 5 | 1 |
| `oracle` | 4 | 1 |

### `tables/generated/manuscript_tables.tex`
| Phrase | Before | After |
|---|---:|---:|
| `Union-Find` | 3 | 3 |
| `union--find` | 2 | 0 |
| `current-graph` | 5 | 0 |
| `upper-bound` | 2 | 0 |
| `oracle` | 0 | 0 |

## Page count
- Previous PDF: 63 pages
- Updated PDF: 61 pages
- Reduction: 2 pages

## Verification
- PDF was rebuilt successfully with `pdflatex`.
- Final PDF renders successfully: 61 pages rendered for visual inspection.
