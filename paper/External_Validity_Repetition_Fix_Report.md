# External-validity repetition fix report

## Goal
The manuscript repeated external-validity, hardware-in-the-loop, real-flight validation, and onboard/embedded profiling limitations in several places. The update centralizes those limitations in the dedicated Limitations subsection and the Conclusion/Future Work section, while shortening the Introduction and latency-related text.

## Main edits

1. **Abstract shortened**
   - Removed the final sentence listing onboard profiling, real-flight validation, and richer MAC/physical models.
   - Kept only a concise statement that the evaluation is simulation-based under the tested settings.

2. **Introduction shortened**
   - Removed repeated deployment-boundary language from the end of the networking-impact paragraph.
   - The introduction now ends that paragraph with a short reference to the metric-specific Results section.

3. **Simulator-scope repetition removed**
   - Removed the separate external-validity table from the methods section.
   - Kept the physical/MAC/interference modelling scope concise and redirected detailed implications to the Limitations subsection.

4. **Latency sections shortened**
   - Removed repeated embedded-device / edge-device deployment caveats from the host-side profiling paragraph.
   - Kept only host-side CPU inference timing and the 20 ms budget comparison.

5. **Limitations kept strong**
   - Consolidated the main limitation statement in the Limitations subsection.
   - It now explicitly mentions synthetic traces, lack of full MAC/packet contention, inactive interference-limited evaluation, hardware-in-the-loop traces, real-flight validation, and onboard profiling.

6. **Conclusion kept strong**
   - Future work still clearly calls for horizon sweeps, real or hardware-in-the-loop flight traces, and packet-level MAC/interference studies.

## Phrase-count check

| Phrase | Before | After |
|---|---:|---:|
| `external validity` | 2 | 1 |
| `hardware-in-the-loop` | 4 | 2 |
| `real flight` | 2 | 0 |
| `real-flight` | 0 | 1 |
| `embedded-device` | 2 | 0 |
| `edge-device` | 1 | 0 |
| `field-validated` | 1 | 0 |
| `deployment claims` | 1 | 0 |

## Build check

- `pdflatex` completed successfully.
- Updated PDF page count: **59 pages**.
- Render check completed for all **59 pages**.
