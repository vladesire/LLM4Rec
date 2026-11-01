# Sequential Kaggle research log

Student uploads all CSVs and handles remote pushes. Four additional scored attempts R1–R4 are authorized **one at a time**; no future round starts without feedback and an instruction to proceed. R1 is scored; R2 awaits upload/result. Current account quota is not accessible to the agent. R3/R4 remain conditional and have not been executed.

| Round | CSV | SHA-256 | Public | Private | Status |
|---|---|---|---|---|---|
| R0 | `submissions/R0_global_popularity.csv` | `73d71295c19c61c3702a3c8e0fae951f27a307ce9b05cd39fd37e9b93e0996e0` | unknown | approximately 0.27, user-reported | SCORED_APPROXIMATE |
| R1 | `submissions/R1_personal_completion.csv` | `95090a22e6eaf0993c3c83512c709393a5ab8daa8439c0a274d955f71ab0ead6` | 0.39334 | 0.38480 | SCORED_USER_REPORTED |
| R2 | `submissions/R2_recency_habit.csv` | `7fbefef69e439c7a027caf2147ed60ff92086068cc7e9babd3f9565b705afe38` | null | null | AWAITING_USER_SCORE |

R1 is the scored incumbent: private 0.38480, public 0.39334, exact numbers reported by the student; original screenshot/submission identifier has not been inspected. R0 remains approximately 0.27. Its code commit is `6a26a69`; original CSV, configuration and metrics are preserved. R0's original local 0.595717 is a historical broad-label proxy, not an estimate of Kaggle accuracy and not comparable with the corrected target.

R1: >=80% continuous-session qualification; personal historical distinct-broadcast frequency; latest qualifying overlap-end then ID ties; fixed qualifying-user popularity fallback. No decay, opportunity adjustment or fusion. On the corrected 1–29 June development task, R1 MAP@5=0.250764 versus replayed R0 strategy 0.173207. Content and ALS independently score 0.187332 and 0.152013 with count inputs. Details, cohort counts, sensitivities, checks, runtime and five examples are under `experiments/R1/`.

Reproduce R1 with `.venv/bin/python scripts/run_r1.py --reproduce`; verify with `.venv/bin/python scripts/verify_r1.py`. The corrected 29 June–27 July audit remains sealed until R4. All development and future Kaggle-driven selection is adaptive.

R1 code commit: `9c7b3f72f67b4801ae1a1191c8528b4ffd47410b`. This commit contains the implementation and initial R1 evidence; the following metadata commit records its identity. Exact source/config/contract hashes are in the manifest.

R1 feedback supports the combined personal/completion package over the approximate R0 result; it does not isolate the contribution of each change. Both public and private scores improve; no subgroup explanation is inferred from their difference. The student authorized R2. R2 now tests modest recency and schedule/channel-habit additions on the unchanged development contract. After R4 feedback, choose the best actually returned private score among all submissions, with a documented simplicity preference for exact ties.

R2 selected a 56-day decay and 75% repeat/25% channel-time-habit blend from eleven saved development variants. Primary MAP@5=0.257532 versus R1 0.250764 on identical corrected labels/population. Schedule-rate additions were locally unhelpful and excluded. Target sensitivities retain the gain direction, but development and selection are adaptive. The corrected later audit remains sealed. Exact formulas, ablations, bootstrap limitations, runtime and checks are in `experiments/R2/HANDOFF.md`.

Reproduce R2 with `.venv/bin/python scripts/run_r2.py --reproduce`; verify with `.venv/bin/python scripts/verify_r2.py`. R2 code commit: pending local commit recording. Source/config/contract hashes are in its manifest.

**STOP: AWAITING_USER_SCORE for R2.** Return public/private scores and comments before R3. R1 remains the best scored incumbent; R2 is an unscored challenger.
