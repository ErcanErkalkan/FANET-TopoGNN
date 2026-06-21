# Seed Repetition Fix Report

## Objective
Reduce repeated emphasis on the three-random-seed limitation. The issue should remain in the proper methodological places but should not weaken the contribution narrative in Highlights or Contributions.

## Main edits
- Removed the three-seed limitation from the Contributions item on the reproducible benchmark.
- Removed repeated seed-count wording from the runtime paragraph.
- Removed repeated seed-count wording from the simulator overview paragraph.
- Removed repeated seed-count wording from the graph-construction paragraph.
- Removed repeated seed-count wording from the opening Results paragraph.
- Removed repeated seed-expansion wording from the practical-interpretation and conclusion paragraphs.

## Where the limitation remains
- Methods / Experimental Setup: seed count, split construction, CI computation, and expanded-seed protocol.
- Results: one statistical-power interpretation paragraph.
- Limitations: one concise limitation statement.

## Phrase-count check in main.tex
| Phrase | Before | After |
|---|---:|---:|
| `three random seeds` | 4 | 3 |
| `three seeds` | 7 | 2 |
| `three-seed` | 3 | 1 |
| `across three seeds` | 3 | 0 |
| `more seeds` | 3 | 1 |
| `seed` overall | 42 | 28 |

## Build check
- PDF build: successful with pdflatex.
- Final page count: 60 pages.
- Render check: 60 pages rendered successfully.
