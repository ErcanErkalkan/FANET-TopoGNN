# Lead-time / positive-lead repetition cleanup

## Objective
Reduce the spread of the lead-time / positive-lead / 600 ms ceiling discussion so that the substantive interpretation remains concentrated in the **Early-Warning Behaviour** subsection, while the metric definition remains concise in Experimental Setup.

## Main edits

1. **Contributions shortened**
   - Removed the numeric median lead-time statement from the Networking impact contribution.
   - Replaced it with a brief statement that Kinetic-TopoGuard uses a conservative warning policy.

2. **Research question and problem formalisation simplified**
   - Replaced explicit lead-time wording with broader warning-behaviour wording.
   - Removed redundant early-warning lead-time framing from the contribution list.

3. **Risk-calibration paragraph simplified**
   - Removed repeated “positive median topology-change lead” wording.
   - Retained the distinction between scalar beta0 accuracy, risk detection, and warning behaviour.

4. **Topology-change metric subsection compressed**
   - Kept the formal event-level definition and formula.
   - Removed interpretive discussion that repeated the Results section.
   - Removed the scaled-lead-time paragraph because it was not used as a central result.

5. **Results interpretation centralized**
   - The main discussion of the 600 ms horizon ceiling, positive lead, and useful-warning interpretation now remains in **Early-Warning Behaviour**.

6. **Limitations and Conclusion shortened**
   - Removed repeated positive-lead / precision-adjusted-lead details.
   - Kept only a concise statement that stronger validation needs horizon sweeps.

## Phrase count check

| Phrase | Before | After |
|---|---:|---:|
| `lead-time` | 7 | 5 |
| `lead time` | 11 | 2 |
| `positive lead` | 5 | 1 |
| `topology-change lead` | 10 | 4 |
| `ceiling` | 4 | 3 |
| `saturated` | 3 | 0 |
| `saturation` | 1 | 0 |
| `\\SI{600}{ms}` | 4 | 3 |

## Page count

- Before: 60 pages
- After: 59 pages

## Build status

- `pdflatex` completed successfully.
- The regenerated PDF has 59 pages.
- All pages were rendered to PNG for visual sanity checking.
