**Current: R4 AWAITING_USER_SCORE; scored incumbent R3 private 0.42671. Earlier narrative statuses are historical.**

# Sequential Kaggle research log

Student uploads all CSVs and handles remote pushes. Four additional scored attempts R1–R4 are authorized **one at a time**; no future round starts without feedback and an instruction to proceed. R1/R2 are scored; R3 awaits upload/result. Current account quota is not accessible to the agent. R4 remains conditional and has not been executed.

| Round | CSV | SHA-256 | Public | Private | Status |
|---|---|---|---|---|---|
| R0 | `submissions/R0_global_popularity.csv` | `73d71295c19c61c3702a3c8e0fae951f27a307ce9b05cd39fd37e9b93e0996e0` | unknown | approximately 0.27, user-reported | SCORED_APPROXIMATE |
| R1 | `submissions/R1_personal_completion.csv` | `95090a22e6eaf0993c3c83512c709393a5ab8daa8439c0a274d955f71ab0ead6` | 0.39334 | 0.38480 | SCORED_USER_REPORTED |
| R2 | `submissions/R2_recency_habit.csv` | `7fbefef69e439c7a027caf2147ed60ff92086068cc7e9babd3f9565b705afe38` | 0.39904 | 0.39378 | SCORED_USER_REPORTED |
| R3 | `submissions/R3_boosted_reranker.csv` | `64e45fc3fc75a6bbd5481a441e37ba1fd139608863cd42862fed1d687df2935b` | 0.43102 | 0.42671 | SCORED_USER_REPORTED |
| R4 | `submissions/R4_count_distribution.csv` | `e3e6e7a74d12c3ffb8bc15054b4a5373007d1f1e1eb0e019fbc89c4fa08da2d6` | null | null | AWAITING_USER_SCORE |

R2 is the scored incumbent: private 0.39378, public 0.39904, exact numbers reported by the student; original screenshot/submission identifier has not been inspected. R1 has private 0.38480; R0 remains approximately 0.27. R0's code commit is `6a26a69`; original CSV, configuration and metrics are preserved. Its local 0.595717 is a historical broad-label proxy, not an estimate of Kaggle accuracy and not comparable with the corrected target.

R1: >=80% continuous-session qualification; personal historical distinct-broadcast frequency; latest qualifying overlap-end then ID ties; fixed qualifying-user popularity fallback. No decay, opportunity adjustment or fusion. On the corrected 1–29 June development task, R1 MAP@5=0.250764 versus replayed R0 strategy 0.173207. Content and ALS independently score 0.187332 and 0.152013 with count inputs. Details, cohort counts, sensitivities, checks, runtime and five examples are under `experiments/R1/`.

Reproduce R1 with `.venv/bin/python scripts/run_r1.py --reproduce`; verify with `.venv/bin/python scripts/verify_r1.py`. The corrected 29 June–27 July audit remains sealed until R4. All development and future Kaggle-driven selection is adaptive.

R1 code commit: `9c7b3f72f67b4801ae1a1191c8528b4ffd47410b`. This commit contains the implementation and initial R1 evidence; the following metadata commit records its identity. Exact source/config/contract hashes are in the manifest.

R1 feedback supports the combined personal/completion package over the approximate R0 result; it does not isolate the contribution of each change. Both public and private scores improve; no subgroup explanation is inferred from their difference. The student authorized R2. R2 now tests modest recency and schedule/channel-habit additions on the unchanged development contract. After R4 feedback, choose the best actually returned private score among all submissions, with a documented simplicity preference for exact ties.

R2 selected a 56-day decay and 75% repeat/25% channel-time-habit blend from eleven saved development variants. Primary MAP@5=0.257532 versus R1 0.250764 on identical corrected labels/population. Schedule-rate additions were locally unhelpful and excluded. Target sensitivities retain the gain direction, but development and selection are adaptive. The corrected later audit remains sealed. Exact formulas, ablations, bootstrap limitations, runtime and checks are in `experiments/R2/HANDOFF.md`.

Reproduce R2 with `.venv/bin/python scripts/run_r2.py --reproduce`; verify with `.venv/bin/python scripts/verify_r2.py`. R2 code commit: `b26d33c6f245083849573d313b4b450286fd6359` (implementation/evidence commit; followed by identity metadata). Source/config/contract hashes are in its manifest.

R2 feedback improved by 0.00570 public/0.00898 private over R1. The student authorized R3; this supports testing signal combination while leaving causal contributions and transfer uncertain.

R3: 200-item candidate pool, micro recall 0.853538 and oracle MAP@5 ceiling 0.599987. A chronological binary boosted reranker selected the 23-feature version dropping ALS/content score features while retaining those candidate sources. Development MAP@5=0.271132 versus R2 0.257532; full and two fixed-blend outcomes are preserved. The later corrected audit remains sealed. R3 code commit: `9b2855a12482a28fb9916e9972d761d4f76c0877` (implementation/evidence commit; followed by identity metadata). Reproduce with `.venv/bin/python scripts/run_r3.py --reproduce`; verify with `.venv/bin/python scripts/verify_r3.py`.

**STOP: AWAITING_USER_SCORE for R3.** Return public/private scores and comments before R4. R2 remains the best scored incumbent; R3 is an unscored challenger. No R4 CSV or model has been generated.


R3 returned public 0.43102/private 0.42671 (user report; screenshot/ID not inspected) and is the scored incumbent, private gain 0.03293 over R2. R4 was explicitly authorized as a complementary count-distribution experiment.

R4: Beta-Binomial kappa100, 512 draws, expected-AP greedy, 10% reciprocal-rank blend with 90% unchanged R3. Development 0.271272 vs0.271132; corrected late audit0.331476 vs0.331066, used for selection. Tiny gains and a bootstrap crossing zero; exploratory, not established superiority. Details in experiments/R4/HANDOFF.md. R4 code commit: pending. STOP for R4 scores before final designation.
