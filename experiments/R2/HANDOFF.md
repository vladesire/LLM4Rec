# R2 — recency and channel/time habit blend

**Status: AWAITING_USER_SCORE.** R2 only; no R3 training, candidate experiments or CSV have been run. The corrected late audit remains sealed.

## Feedback and hypothesis

The student reported R1 public **0.39334**, private **0.38480**, and authorized R2. These exact reported numbers make R1 the scored incumbent. The original score screenshot and submission identifier have not been inspected. Both directions improve over approximate R0 private 0.27; this supports the combined personal/completion package without isolating its ingredients.

R2 asks whether modest recency and coarse channel/time habits improve personal recurring favourites. The selected package uses a **56-day half-life**, **75% repeat forecast**, and **25% channel/time habit forecast**. Schedule-rate adjustments were tested and excluded. No numerical Kaggle improvement is promised.

## Unchanged contract

Same >=80% continuous-session watches, distinct-broadcast counts, deterministic frequency Top-5 targets and ties, template users, target-window scheduled candidates, repeats allowed and completion-popularity fallback as R1. Features stop before **1 June 2020 00:00**; development is **[1 June,29 June)**. All 1,260 users remain in the primary mean: 898 nonempty, 642 with five labels, 362 empty as AP=0. R1's no-decay predictions and per-user AP replay exactly.

The corrected June 29–July 27 audit was not constructed or inspected. Production separately uses all eligible history before July 27, as authorized. R0's original 0.595717 remains a different historical broad-label proxy.

## Local ablations

| Variant | Development MAP@5 |
|---|---:|
| R1 / repeat, no decay | 0.250764 |
| Repeat, 28-day half-life | 0.248912 |
| Repeat, 56-day half-life | 0.252151 |
| Smoothed schedule-rate repeat | 0.249188 |
| Capped schedule-rate repeat | 0.245178 |
| Channel-only habit | 0.229281 |
| Channel/time habit | 0.242136 |
| 25% capped-schedule blend | 0.251452 |
| 50% capped-schedule blend | 0.250233 |
| 10% time-habit blend | 0.254880 |
| **25% time-habit blend — selected** | **0.257532** |

After the three recency variants, additions use the selected 56-day half-life. The three-way blend was not run because schedule additions did not improve on the best repeat-only reference. Selection followed the predeclared highest-primary-MAP rule; exact ties prefer simplicity. Eleven settings are adaptive development selection, not independent confirmation.

The selected score is 0.361348 on nonempty users and 0.379003 on five-label users. Its primary delta against R1 is **+0.006769**. The same predictions improve under aggregate-sittings targets (0.253405 versus 0.247316) and reversed target-ID ties (0.230278 versus 0.225520). Neither sensitivity replaces the primary rule.

A paired 2,000-replicate user bootstrap gives a conditional 95% interval of approximately [0.003774,0.009578] for the selected local delta. It ignores selection among settings and temporal shift, so it is not an independent significance or Kaggle-gain claim. AP improves for 171 development users, worsens for 101, and is unchanged for 988.

## Features and formulas

Each distinct qualifying broadcast has weight `2^(-age_days/56)`, measured from its clipped observation end. Effective history days integrate the same exponential kernel over the historical calendar. Repeat forecast = weighted personal count × forecast days / effective history days. This puts short-development and longer-production counts into comparable forecast units.

Habit models use only nonzero labelled historical broadcast opportunities, not unlabelled airtime or assumed actual exposure. Let H_c be weighted historical channel airings, U_uc personal qualifying watches, and q_c global per-user qualifying rate. Channel propensity is `(U_uc+20*q_c)/(H_c+20)`. A cell is channel × six-hour period × weekday/weekend. Cell propensity is `(U_cell+20*channel_propensity)/(H_cell+20)`. Sum these propensities over each programme's target-window scheduled airings. Cells have fixed arithmetic identifiers; no future metadata vocabulary or viewing outcomes are fitted.

Both final components use the same 56-day decay. The final score is `0.75*repeat_forecast + 0.25*habit_forecast`; last qualifying observation end then integer ID resolves score ties. Cold qualifying-history users receive the unchanged R1 fallback. Availability is an opportunity proxy; the model does not establish whether the user was exposed or available.

The rejected schedule alternatives use a two-airing-smoothed personal programme watch rate projected over future airings, and a past/future airing-rate multiplier capped to [0.25,4]. Both are repeat-only. Their losses are preserved in the ablation table and JSON.

## Immutable CSV and checks

**File:** `submissions/R2_recency_habit.csv`

**SHA-256:** `7fbefef69e439c7a027caf2147ed60ff92086068cc7e9babd3f9565b705afe38`

- Validated 1,260 rows, exact template schema/users/order, five distinct nonzero eligible integer IDs, no null/index, single-space separators.
- Different bytes from R0 and R1; previous files are unchanged and write-protected.
- 1,235 distinct lists, 479 distinct IDs, 98.38% qualifying-history repeat share. Five cold users receive all 25 fallback slots. Ordered lists change for 1,111 users versus R1, mostly through repeat ranking.
- Four focused tests check decay/future guard, effective history normalization, calendar cells, and hand-calculated forecast/shrinkage/cap/cold-user values.
- Independent dataframe sorting matches every production score/list/CSV row. Source hashes and R0/R1 CSV identities are checked. A cached full R2 reproduction exactly matches development predictions, AP and CSV bytes.
- Initial round runtime: 5.026 seconds after installed setup and reused completion cache; original timings are preserved separately from reproduction time.

Only numeric future availability is read. Ambiguous future schedule rows are excluded from opportunity calculations; the permitted candidate universe remains unchanged. Full production availability spans the supplied July 27–October 19 noon schedule (84.5 days), rather than guessing from week labels.

## Artifacts and reproduction

`metrics.json`, `ablations.csv`, `development_predictions.json`, `selected_strategy.json`, `paired_uncertainty.json`, `verification.json`, and `manifest.json` contain saved evidence. `formal_two_user_examples.json` retains the same two users across all variants; `five_production_examples.json` contains five lists without invented future outcomes. Code/config/contract hashes and the implementation commit are recorded in the manifest/log.

The fitting and target histories are unchanged and remain in `experiments/R1/formal_two_user_histories.csv`; no duplicate or redefined history is substituted for the R2 examples.

```sh
.venv/bin/python scripts/run_r2.py --reproduce
.venv/bin/python scripts/verify_r2.py
shasum -a 256 submissions/R2_recency_habit.csv
```

Reproduction writes evidence under ignored `cache/R2_reproduction/` and verifies the immutable CSV. The normal entry point stops after confirming an already frozen R2 identity.

## Uncertainty and next branch

The local gain is modest. Reused development data, target ties, single-stream/device ambiguity, unknown hidden target/population conventions, opportunity versus actual exposure, prior strengths and temporal transfer remain limitations. Coarse habits may promote frequently scheduled programmes without proving their personal usefulness. The R1–R2 comparison is a package comparison; the ablations separately show recency and habit contributions locally.

Submit this one file and return exact public/private scores, submission ID if available, and comments. R1 remains the scored incumbent until R2 is scored. If R2 transfers, the provisional R3 path is candidate recall/oracle diagnostics before any compact reranker; if it fails, diagnose transfer and retain R1 as reference. **STOP until feedback and instruction to proceed.**
