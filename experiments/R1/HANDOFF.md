# R1 — completion-aware personal favourites

**Status: SCORED_USER_REPORTED.** The student returned public 0.39334/private 0.38480 and authorized R2. R1 is the scored incumbent. The original screenshot/submission ID has not been inspected. The following sections preserve R1's preparation evidence; current research state is in `SUBMISSION_LOG.md`.

## Hypothesis and changes

R0 recommended one global list. R1 ranks eligible programmes by each user's distinct historical broadcast counts with >=80% completion in one continuous channel session. Frequency ties use the latest qualifying overlap end, then ascending integer ID. There is no decay, opportunity factor or collaborative fusion. Missing slots use the fixed completion-qualified distinct-user popularity fallback.

This is a package comparison involving personal ranking and completion-aware training. A private-score gain would not isolate those contributions or prove the target proxy is exact. The full shared contract is `experiments/EVALUATION_CONTRACT.md`; selected participant ideas and installed API checks are attributed in `sources.md`.

## Development evidence

Features stop at **1 June 2020 00:00**, targets cover **[1 June,29 June 2020)**. Primary target: up to five programme IDs by qualifying-broadcast frequency; ascending ID breaks ties independently of predictions. All 1,260 template users remain in the primary mean, with 362 empty-target users counted as zero. There are 898 nonempty-target users and 642 with five labels; no unreachable labels were removed. Candidates: 3,929 nonzero scheduled IDs.

| Method | Primary MAP@5 | Nonempty MAP@5 | Five-label MAP@5 |
|---|---:|---:|---:|
| R0 strategy replay, fitted before 1 June | 0.173207 | 0.243030 | 0.248214 |
| Completion popularity / common fallback | 0.195555 | 0.274387 | 0.287326 |
| Content, qualifying-count input | 0.187332 | 0.262849 | 0.266184 |
| ALS, qualifying-count input | 0.152013 | 0.213292 | 0.202580 |
| **R1 personal frequency** | **0.250764** | **0.351851** | **0.366132** |

R0's original 0.595717 used a different broad-label target and later split. It is preserved as historical proxy evidence and is not a comparator in this table. The only scored incumbent is still R0, with approximate user-reported private 0.27; exact public/private evidence has not been inspected.

Content and ALS remain independent eligible methods. Their weighting now uses log1p(qualifying count), replacing R0's watched-second weights. ALS keeps 32 factors, 15 exact iterations, regularization 0.1, seed 42, confidence 1+10 log1p(count). User/item shapes, ID mapping and explicit repeat allowance were checked; no ALS search was run.

Sensitivity checks retain the same predictions: aggregate-across-sittings targets give R1 0.247316; descending-ID target ties give 0.225520. These checks expose specification sensitivity and do not replace the primary rule. The corrected 29 June–27 July audit remains sealed until R4; it was previously exposed under old broad labels and is not called pristine.

## CSV and production diagnostics

**File:** `submissions/R1_personal_completion.csv`

**SHA-256:** `95090a22e6eaf0993c3c83512c709393a5ab8daa8439c0a274d955f71ab0ead6`

The file is write-protected and the runner refuses to overwrite different bytes. It differs from R0. Production uses all eligible history before 27 July; this refit is separate from development evaluation.

- 1,260 rows; exact template schema/users/order; five distinct nonzero eligible integer IDs per row; no nulls or index; single-space separators.
- 1,231 distinct recommendation lists, 454 distinct recommended IDs; qualifying-history repeat share 98.65%.
- 27 users need some fallback; six need full fallback; 85 fallback slots. Five users have no qualifying history; one additional full-fallback user has no eligible qualifying history.
- Development R1 has 1,217 distinct lists, 95.86% repeat share; 90 users need some fallback and 19 need full fallback.
- Five production example lists are saved in `five_production_examples.json`, with no invented future outcomes. The original two formal report users are retained for every development method in `formal_two_user_examples.json` and their corrected histories CSV.

The initial round entry point took 19.009 seconds including preparation (11.490 seconds), models, original-row checks and output validation. Actual per-method times are in `comparison.csv` and `metrics.json`; setup is excluded. The explicit cached reproduction is separately recorded, without replacing original times.

## Actual checks and remaining uncertainties

Five focused tests passed: session changes, duplicate/overlapping fragments, split-crossing full denominators, distinct-airing frequency/target ties and checked AP arithmetic. Four worked cases were re-read from original view/schedule records: qualifying overlap, partial overlap, a session spanning broadcasts, and a cutoff-crossing session. `verify_r1.py` independently sorts production count/timestamp profiles and matches every CSV list, verifies raw timestamps, checks the real-user AP=0.55 arithmetic, and confirms preserved R0 bytes/source hashes.

Original full broadcast duration remains the denominator on both sides of every cutoff. No future viewing enters features. No corrected late-audit targets or metrics were constructed.

There are 351,499 sessions clipped at overlapping channel changes, removing 928,805,255 seconds from the single-stream interpretation. The logs have no device ID: simultaneous-device activity versus channel switching is unresolved. Exact hidden completion/session rules, target ties, empty-user treatment and four-week-to-twelve-week transfer also remain uncertain. Personal checks by the student are still pending.

## Reproduce and review

```sh
.venv/bin/python scripts/run_r1.py --reproduce
.venv/bin/python scripts/verify_r1.py
shasum -a 256 submissions/R1_personal_completion.csv
```

`--reproduce` writes new evidence under ignored `cache/R1_reproduction/`, checks the immutable CSV, and leaves original round evidence intact. Running the entry point without this flag after R1 is frozen only checks its identity and stops. The code commit is recorded in `manifest.json` and `SUBMISSION_LOG.md`.

## Student action and provisional branch

Confirm in Kaggle that **four additional attempts remain** before uploading R1. Account quota cannot be inspected through local credentials/tools; the earlier displayed number 100 is not treated as a verified current remaining allowance. Submit this exact file once and return exact public/private scores, submission ID if available, and comments.

No numerical gain is promised. Improvement would support the combined personal/completion package; failure could reflect proxy/session ambiguity, availability, history coverage or temporal transfer. Public/private disagreement will be recorded rather than assigned an invented cause.

If R1 improves and diagnostics remain credible, the provisional R2 branch is modest recency/schedule/habit ablations. If it is weak or fails to transfer, inspect usable history and qualification/parity first. These are plans only: **STOP until the student returns feedback and authorizes the next round.**
