# Contribution Log — Group G12

Maintained as required by Section 27 of the assignment guide. Each entry names
the student, what was done, and where the evidence lives. Git and Overleaf
histories are the primary evidence; this log summarises them.

| Member | Roll no. | Role |
|---|---|---|
| Faheem | 23i-0728 | Primary Researcher |
| Irtaza Kazmi | 23i-6001 | Co-Researcher |

## Log

| Date | Student | Contribution | Evidence |
|---|---|---|---|
| 2026-08-30 | Faheem | Built cold-start thermal characterisation harness (NVML, 20 Hz) and analysis script; ran pilot and Q4 runs | `138e538`, `src/thermal_tc.py`, `src/analyze_tc.py` |
| 2026-08-30 | Faheem | Ran Q8 characterisation; identified Q4/Q8 convergence to a common throughput floor | `ec53355`, `experiments/raw/tc_q8_*` |
| 2026-08-30 | Faheem | Ran FP16 characterisation; documented it as confounded by CPU offload | `074f49f`, `experiments/raw/tc_fp16_*` |
| 2026-09-13 | Faheem | Ran 0.5B control run, ruling out a serving-stack overhead artefact | `8c1317e`, `experiments/raw/tc_s05_*` |
| 2026-09-13 | Faheem | Wrote Cutoff 1 proposal | `60399de`, `bfc98d0`, `proposal/` |
| 2026-09-13 | Faheem | Verified all references against arXiv, Crossref and publisher records; aligned README and proposal; created LaTeX manuscript source | this commit, `paper/` |
| 2026-09-13 | Faheem | Set up public repository; invited Irtaza Kazmi as collaborator (write access) | GitHub settings |

<!-- Add a row for every working session. Both members should appear regularly. -->
