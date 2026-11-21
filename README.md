# FANET-TopoGNN — current research snapshot

The default branch **main** contains the single active project snapshot and is **publisher- and journal-neutral**.

Research question: can one selected fragmentation-warning threshold maintain a specified false-alarm budget across different flying ad hoc network mobility regimes, node counts and development blocks?

## Complete reproducible snapshot

Source files are browsable on this branch. Large recorded CSV datasets and PDF evidence are included in **one complete project ZIP**, tracked as seven transport chunks under `snapshot/` (these are *not* seven versions).

```bash
python assemble_snapshot.py
```

Expected SHA-256: `8b06e9235a46ae9fc85887f901f0619e37b86f6216ca879afc8f743a24d7267d`.

Extract the reconstructed `FANET-TopoGNN_Current.zip` in a separate directory, then run:

```bash
python run.py check
python run.py verify
python run.py test
```

The extracted ZIP is the **authoritative complete research package**. A source-only checkout deliberately omits large duplicate experiment CSVs, so the `run.py` verification commands require the complete extracted archive.

## Findings (descriptive development simulation)

- 45 trajectory groups, 180 paired radio/graph runs, 60 simulated minutes.
- Fixed TopoGNN threshold 0.15: 183 of 1,114 fragmentation onsets matched; event F1 = 0.264.
- 88 false alarms (1.47 per simulated minute overall), but six of 15 block-by-mobility strata exceed the stipulated two/minute reference.
- At 30 nodes, all 82 fragmentation events were missed with no emitted alarm.
- History-4 is a post-hoc exploratory comparator and exceeded the reference budget.

These are **simulated graph-connectivity results**, not packet-delivery outcomes, independent blind evidence, prospective clinical/safety deployment, or demonstrated model superiority.

## Contents

`core/`: simulator and model. `frozen/`: locked scientific inputs. `audit/`: scoring and verification. `data_derived/`: derived summary tables. `paper/manuscript/`: one provisional journal-neutral draft. `snapshot/`: the integrity-checked current full research artifact.

Freeze-stage identifiers are **scientific provenance**, not alternate active software releases. No claim is made that a publisher has accepted or reviewed this work.

Research status: **AUTHOR REVIEW / HOLD**; authorship, external data release permissions and submission require author confirmation.
