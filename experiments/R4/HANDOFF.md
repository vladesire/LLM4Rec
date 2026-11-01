# R4: count-distribution complement to R3

**AWAITING_USER_SCORE.** R3 is the scored incumbent: public 0.43102/private 0.42671, exact user report, screenshot/ID not inspected. Its CSV, configuration and inference model remain unchanged. No upload or remote push was performed.

## Selected challenger

File: `submissions/R4_count_distribution.csv`

SHA-256: `e3e6e7a74d12c3ffb8bc15054b4a5373007d1f1e1eb0e019fbc89c4fa08da2d6`

**90% R3 / 10% count-distribution expected-AP ranking**, combined through reciprocal ranks (constant 60). This is an exploratory complement, not established superiority or a promised private-score gain. The count model alone loses to R3. Novelty is not the justification: the conservative blend has tiny positive paired differences on development and the subsequently opened audit, and retains most of R3's ranking.

## Frozen contract and model

The R1 completion/frequency target, 80% continuous-session qualification, distinct broadcasts, original programme durations, target-ID ties, eligible IDs, repeat policy, fallback and all 1,260 users remain unchanged. Features use only each cutoff's past. Candidate generation and the recommendation pool are the unchanged R3 200-item pool.

Historical opportunity A_ui counts distinct labelled broadcasts intersecting the interval from the user's first historically observed viewing to cutoff. Qualifying n_ui uses the same interval and original continuity rules. Every checked pair satisfies 0<=n_ui<=A_ui. These opportunities are availability proxies, not observed exposure. No decay is used in Beta sufficient statistics.

Population channel rates divide qualifying watches by aligned user-channel opportunities. User-channel rates shrink toward these rates with 20 opportunity observations; the programme prior averages channel propensities using its target schedule's channel mix. Rates are bounded to [0.000001,0.999999] for valid positive Beta shapes. This is a per-airing rate, never R3 membership probability.

Posterior a=kappa*mu+n, b=kappa*(1-mu)+A-n. Future count N is Beta-Binomial(M,a,b), where M is actual target-window scheduled broadcasts. Mean is M*a/(a+b). Kappa {5,20,100} was compared on the earlier May4–June1 inner window; 100 won membership MAP. Assumptions include stable exchangeable airings, independent programme rates, sparse labelled schedules, ambiguous device/session identity and incomplete observation/exposure.

## Sampling and list decision

512 draws, seed 42. Separate user-seeded Beta and Binomial streams generate bounded counts. Sample targets contain only positive-count programmes, up to five by count-descending/ID-ascending order. Empty draws remain empty and contribute zero to the simulation objective.

The modeling pool includes every scheduled programme with personal qualifying history, the top 500 predictive means and all 200 recommendation candidates. Competitors outside the recommendation pool can occupy sampled target slots. This is still a truncated catalogue model. Expanding the mean shortlist to 1,000 for 30 users changed any recommendation membership probability by at most 0.013672. Shared draws within each model compare mean/membership/decision rules; appended expansion rows preserve base-item draws. Independent samples do not learn all user-behavior correlations.

Expected-AP moments use the sampled set size denominator, with empty samples retained as zeros. Greedy selection considers the top 25 membership candidates plus R3's Top-5. The exact moment identity was tested against direct AP for every five-item permutation of a six-item toy universe containing empty/short targets; greedy was compared with exhaustive optimum. Counts are checked against future availability. Beam search was unnecessary.

Distribution-only list stability at 512 versus 2,048 draws was limited (43.3% membership lists, 40% greedy lists identical across the first 30 users). The **final 10% blend was identical for all 30** in the same check. These are limited-sample stability diagnostics, not proof for every user.

## Actual ablations and audit

| Rule | June development MAP@5 |
|---|---:|
| A: unchanged R3 | 0.271132 |
| B: predictive mean | 0.253992 |
| C: sampled membership | 0.256944 |
| D: expected-AP greedy | 0.257600 |
| E: 10% distribution / 90% R3 | **0.271272** |
| E: 25% distribution | 0.271253 |
| E: 50% distribution | 0.269909 |

Distributional conversion helps this count predictor locally; expected-AP adds a small further development improvement. Neither replaces R3 successfully. On the inner window, membership beat mean but greedy did not consistently beat membership. Do not attribute the final package gain solely to uncertainty or metric optimization.

The shortlist and choice rule were saved before the corrected June29–July27 audit was opened. That corrected audit had not been exposed in R3; old broad labels had been examined earlier. The audit compared only R3 with the frozen 10% challenger: **0.331066 versus 0.331476**. It was used as a selection gate, so it is no longer an untouched estimate. Empty users: 52; nonempty: 1,208; five-label users: 1,001. No post-audit mixture search was run.

Development delta is only +0.000140. A 2,000-user-bootstrap interval is approximately [-0.001468,0.001834], ignoring adaptive selection and temporal shift. Only 34 users improve, 31 worsen and 1,195 are unchanged. This is weak evidence; R4 tests a small complement rather than a confidently better replacement. Production ordered lists differ from R3 for 328 users.

The unchanged R3 pool retrieves 3,281/3,844 development labels (micro recall 0.853538). Oracle MAP is 0.599987, while actual R3 is 0.271132, leaving oracle headroom 0.328854. Micro recall pools labels; oracle averages per-user attainable fractions and includes empty users as zero, explaining their different values. The oracle is not achievable performance evidence.

## Checks, calibration and provenance

CSV: 1,260 exact template users/order/schema, five distinct nonzero eligible integer IDs, no null/index, finite ranking scores. Previous R0–R3 files are unchanged. There are 1,191 distinct lists; five cold users receive the fixed fallback. Full production resampling reconstructs the exact saved CSV. Mathematical and bounded-count tests pass.

Rough 90% interval coverage is very high and zero-dominated, not persuasive calibration. A fixed 10,000-pair sample has only 31 personally known pairs; observed/predicted means are roughly 0.419/0.505 there, and 0.00160/0.00196 for new pairs. Activity-stratified likelihood/coverage diagnostics and exact sample sizes are saved in `additional_checks.json`; do not overinterpret the tiny known-item sample. Full per-user count totals/zero mass/coverage diagnostics are also retained.

Initial round runtime: 113.583 seconds, excluding extra mathematical/calibration/reconstruction checks. Original R3 inference and cached temporal features are reused. Input fingerprints remain those in R3. The background Lichman/Smyth PDF was requested through the web tool but timed out/unavailable on repeated attempts; it is a user-supplied motivation, not a paper claimed to have been read or a reproduced SWEET.TV method. This generative formulation and external planning assistance must be disclosed in the final report.

## Reproduction and stopping

```sh
.venv/bin/python scripts/verify_r4.py
.venv/bin/python -m unittest tests/test_count_r4.py -v
shasum -a 256 submissions/R4_count_distribution.csv
```

`verify_r4.py` regenerates production count samples and reconstructs the CSV without re-fitting or re-opening audit selection. `run_r4.py --reproduce` can rebuild all round evidence under ignored `cache/R4_reproduction/`; this repeats the frozen procedure, not a new audit-guided search. The normal runner stops on a frozen manifest. A second full clean-clone run is not claimed. Code commit, parameters, seeds and source hashes are recorded.

Submit this single file and return exact public/private scores, ID if available and comments. **STOP.** Only after feedback designate the best actually returned private score across all rounds; R4 need not win. The final report still needs consolidation, personal checks, GitHub URL, genuine score screenshot and unchanged native audit, with adaptive feedback/reused validation disclosed.
