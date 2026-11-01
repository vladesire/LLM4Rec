# Sequential Kaggle research log

Student uploads all CSVs and handles remote pushes. Four additional scored attempts R1–R4 are authorized **one at a time**; no future round starts without feedback and an instruction to proceed. Current quota is not accessible to the agent; confirm four additional attempts remain before R1 upload.

| Round | CSV | SHA-256 | Public | Private | Status |
|---|---|---|---|---|---|
| R0 | `submissions/R0_global_popularity.csv` | `73d71295c19c61c3702a3c8e0fae951f27a307ce9b05cd39fd37e9b93e0996e0` | unknown | approximately 0.27, user-reported | SCORED_APPROXIMATE |
| R1 | `submissions/R1_personal_completion.csv` | `95090a22e6eaf0993c3c83512c709393a5ab8daa8439c0a274d955f71ab0ead6` | null | null | AWAITING_USER_SCORE |

R0 is the only scored incumbent. Exact score evidence/submission identifier has not been inspected. Its code commit is `6a26a69`; original CSV, configuration and metrics are preserved. R0's original local 0.595717 is a historical broad-label proxy, not an estimate of Kaggle accuracy and not comparable with the corrected target.

R1: >=80% continuous-session qualification; personal historical distinct-broadcast frequency; latest qualifying overlap-end then ID ties; fixed qualifying-user popularity fallback. No decay, opportunity adjustment or fusion. On the corrected 1–29 June development task, R1 MAP@5=0.250764 versus replayed R0 strategy 0.173207. Content and ALS independently score 0.187332 and 0.152013 with count inputs. Details, cohort counts, sensitivities, checks, runtime and five examples are under `experiments/R1/`.

Reproduce R1 with `.venv/bin/python scripts/run_r1.py --reproduce`; verify with `.venv/bin/python scripts/verify_r1.py`. The corrected 29 June–27 July audit remains sealed until R4. All development and future Kaggle-driven selection is adaptive.

R1 code commit: pending local commit recording. Exact source/config/contract hashes are already in its manifest.

Next: wait for the student's R1 public/private scores and comments. A provisional R2 branch is recorded in `experiments/R1/HANDOFF.md`; R2 has not been run or generated. R1 is not designated the final file before its result. After R4 feedback, choose the best actually returned private score among all submissions, with a documented simplicity preference for exact ties.
