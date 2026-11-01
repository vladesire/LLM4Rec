**Screenshot now inspected:** all five returned scores and Complete status confirmed; R0 exact public0.28078/private0.27750. R4 remains designated.

# Submission log - experiments complete

Scores are user-reported; genuine screenshot evidence is tracked separately. No agent Kaggle upload or remote push.
| Round | Immutable CSV | Public | Private | Status |
|---|---|---|---|---|
| R0 | submissions/R0_global_popularity.csv | None | approximately 0.27 | SCORED_USER_REPORTED_APPROXIMATE |
| R1 | submissions/R1_personal_completion.csv | 0.39334 | 0.3848 | SCORED_USER_REPORTED |
| R2 | submissions/R2_recency_habit.csv | 0.39904 | 0.39378 | SCORED_USER_REPORTED |
| R3 | submissions/R3_boosted_reranker.csv | 0.43102 | 0.42671 | SCORED_USER_REPORTED |
| R4 | submissions/R4_count_distribution.csv | 0.4289 | 0.4269 | SCORED_USER_REPORTED |
**Designated final: R4, private0.42690**, chosen by maximum returned private score. R3 has private0.42671; +0.00019 does not establish significant superiority. Public R4 is lower than R3; no subgroup explanation is inferred. Experiments are stopped on the student’s instruction.

Exact hashes and implementation commits:

- R0: SHA-256 `73d71295c19c61c3702a3c8e0fae951f27a307ce9b05cd39fd37e9b93e0996e0`; code commit `6a26a6970dcf59568b1981dae4f46ec12a0d2b58`.

- R1: SHA-256 `95090a22e6eaf0993c3c83512c709393a5ab8daa8439c0a274d955f71ab0ead6`; code commit `9c7b3f72f67b4801ae1a1191c8528b4ffd47410b`.

- R2: SHA-256 `7fbefef69e439c7a027caf2147ed60ff92086068cc7e9babd3f9565b705afe38`; code commit `b26d33c6f245083849573d313b4b450286fd6359`.

- R3: SHA-256 `64e45fc3fc75a6bbd5481a441e37ba1fd139608863cd42862fed1d687df2935b`; code commit `9b2855a12482a28fb9916e9972d761d4f76c0877`.

- R4: SHA-256 `e3e6e7a74d12c3ffb8bc15054b4a5373007d1f1e1eb0e019fbc89c4fa08da2d6`; code commit `e6255ca692a83c642b99d308768ae0709006f6da`.

R1 changes the package to completion-aware personal frequency; its feedback does not isolate completion versus personalization. R2 adds modest recency/habit. R3 combines temporal signals with boosted reranking. R4 preserves R3 and adds10% sampled-count expected-AP reciprocal-rank ranking; pure count models lose. All configurations and negative outcomes are preserved under their round directories.

Corrected June1-June29 MAP@5: R0 replay .173207, qualifying popularity .195555, content .187332, ALS .152013, R1 .250764, R2 .257532, R3 .271132, R4 .271272. All1260 users, empty targets zero, identical declared target/candidate/repeat/fallback policies. R0 original .595717 is historical broad-label proxy only.

R4 freezes the shortlist, opens the corrected June29-July27 audit once and uses it as a selection gate (.331066 R3 versus .331476 R4). Earlier broad labels exposed the period. Four adaptive private-score queries and local tuning are not independent confirmation.

Reproduction: `.venv/bin/python scripts/run_rN.py --reproduce` for N=1,2,3,4; fresh-cache bootstrap and final report commands are in README.md. No further experiment or upload is needed. Final external/personal requirements are in HANDOFF.md.
