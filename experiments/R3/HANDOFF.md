# R3 — chronological boosted reranking

**Status: AWAITING_USER_SCORE.** R3 is the only active round; no R4 models, CSV or corrected late-audit targets have been generated.

## Returned feedback and hypothesis

The student reported R2 public **0.39904**, private **0.39378**, and authorized R3. R2 is the scored incumbent, improving on R1 by 0.00570 public/0.00898 private. These are exact user-reported numbers; the original screenshot/submission ID has not been inspected. This supports the R2 package without isolating causal ingredients.

R3 tests whether a past-only binary boosted model can combine personal repetition, recency, habits and reliability better than R2's fixed blend. The chosen model uses **23 personal, schedule/habit and reliability features**. ALS/content remain candidate sources; their numeric score/rank features were dropped after a local ablation. A public/private gain is not guaranteed.

## Candidate diagnostics

Each user receives 200 distinct eligible IDs. Sources: up to 100 personal-frequency candidates, 50 new R2 candidates, 30 new ALS, 15 new content and five new completion-popularity candidates. Remaining slots use R2 and the fixed fallback. All R2 Top-5 items are retained. Candidate construction accepts no target labels; positives are never injected.

On the unchanged June development task, the pool retrieves **3,281/3,844 labels**: micro recall **0.853538**, macro recall among nonempty users **0.841852**. Oracle MAP@5 is **0.599987**, averaging over all 1,260 users with empty labels as zero and retaining missing labels in the denominator. This is a ceiling, not achieved performance. Recall above the predeclared 0.8 criterion and the large gap from R2 justified reranking rather than adding another candidate model.

## Chronological supervised learning

| Role | Feature cutoff | Target window | History length |
|---|---|---|---:|
| Earlier training snapshot 1 | 23 March | 23 March–20 April | 2 weeks |
| Earlier training snapshot 2 | 6 April | 6 April–4 May | 4 weeks |
| Inner validation/stopping | 4 May | 4 May–1 June | 8 weeks |
| Development comparison | 1 June | 1–29 June | 12 weeks |
| Production scoring | 27 July | Supplied future schedule | 20 weeks |

All dates are 2020 and endpoints are half-open. Earlier fitting labels stop at 4 May before inner evaluation starts. The development refit adds the inner example, so fitting labels stop at 1 June before development starts. Earlier training target windows overlap one another; this is disclosed. Every candidate model, metadata vectorizer and feature snapshot uses only its own prefix.

The production refit additionally includes the completed June development example; its latest supervised label ends at 29 June, before production prediction starts on 27 July. Production features may use all pre-test history. The **corrected 29 June–27 July audit remains sealed**, and is not called pristine because old broad labels were previously inspected.

Counts/overlap hours become per-week rates or personal shares; schedule forecasts become per-forecast-week rates. Observed/calendar history lengths and missingness are features. The short-history training and longer-history inner/development comparisons are a transfer analogue: inner full-model MAP is 0.359956 versus R2 0.332586 on that same inner window. Different windows have different label populations, so their absolute scores are not interchangeable. Longer production history/horizon remains an extrapolation risk.

## Model and local comparisons

The installed scikit-learn 1.9.1 HistGradientBoostingClassifier supports explicit chronological X_val/y_val. It uses binary log loss for Top-5 membership, not an exact MAP objective. Two leaf/regularization settings were compared on the inner window. Both reached the **160-tree cap** despite configured log-loss early stopping; no early-stop trigger before the cap is claimed. The 31-leaf/L2=5 setting won inner MAP (0.359956 versus 0.356143).

Other fixed settings: learning rate 0.05, min samples per leaf 100, max bins 63, seed 42, one computational thread. No new package was installed. A physical-core detection warning occurred; explicit thread limits remained in effect and runs completed.

| Development variant | Primary MAP@5 |
|---|---:|
| R2 scored incumbent's strategy | 0.257532 |
| Full 29-feature reranker | 0.269543 |
| **Drop ALS/content score features — selected, 23 features** | **0.271132** |
| Drop schedule/habit features | 0.264500 |
| 25% reranker / 75% R2 rank blend | 0.265066 |
| 50% reranker / 50% R2 rank blend | 0.269370 |

The pool is held fixed for feature ablations: dropping ALS/content features does not remove their candidate contributions. The selected model's nonempty-user MAP is 0.380431 and five-label-user MAP is 0.399704. Primary population is unchanged: all 1,260 users, 898 nonempty, 642 with five labels, 362 empty as zero. R2 predictions and per-user AP replay exactly.

Selected primary delta is **+0.013600** against R2. A 2,000-replicate paired-user bootstrap gives a conditional 95% interval [0.008100,0.019160]; 293 users improve, 195 worsen and 772 are unchanged. This does not correct adaptive setting/family selection or temporal shift and is not independent confirmation.

## Immutable CSV, diagnostics and actual checks

**File:** `submissions/R3_boosted_reranker.csv`

**SHA-256:** `64e45fc3fc75a6bbd5481a441e37ba1fd139608863cd42862fed1d687df2935b`

- Validated 1,260 rows; exact template schema/order/users; five distinct nonzero eligible integer IDs; no null/index; single spaces.
- Different bytes from R0/R1/R2; prior files remain unchanged.
- Production has **1,180 distinct lists**, **138 distinct IDs**, five cold users receiving 25 fallback slots, and **89.08% qualifying-history repeat share**.
- Catalogue coverage is substantially narrower than R2's 479 IDs, while recommendations include more previously unqualified items. This is an observed transfer risk, not evidence of useful discovery or enjoyment.
- Three focused tests passed: target-independent distinct candidates, oracle denominators with missing/empty labels, and the sealed-audit guard.
- Independent dataframe sorting matches every CSV list. A saved-model predict_proba call reproduces probabilities exactly. Boundaries, pool eligibility/uniqueness, old CSV identities, full relevance denominators and R2 replay are checked.
- A full cached R3 reproduction matches development predictions, per-user AP and immutable CSV bytes exactly. Original runtime measurements are preserved.

The first R3 main run took **110.515 seconds**, with a prebuilt development candidate snapshot; production fit/scoring took 28.948 seconds. This excludes earlier setup/diagnostic work and is not a full clean-clone rerun. Snapshot features and trained models are cached outside git; exact code/config/input hashes and package versions are recorded.

## Reproduction and evidence

```sh
.venv/bin/python scripts/run_r3.py --reproduce
.venv/bin/python scripts/verify_r3.py
shasum -a 256 submissions/R3_boosted_reranker.csv
```

Reproduction writes under ignored `cache/R3_reproduction/`; existing round evidence and CSV identity are preserved. The normal entry point only confirms an already frozen R3 and stops. Implementation commit is in `manifest.json` and `SUBMISSION_LOG.md`.

`candidate_diagnostics.json`, `metrics.json`, `comparison.csv`, `development_predictions.json`, `paired_uncertainty.json`, `verification.json` and `manifest.json` preserve the comparisons and negative outcomes. The original two formal users are retained across variants with IDs/titles/categories/scores in `formal_two_user_examples.json`; their unchanged June histories remain under R1. `five_production_examples.json` contains no invented future labels.

## Next step

Submit this file once and return exact public/private scores, submission ID if available, and comments. R2 remains incumbent until R3 is scored. If R3 transfers, the provisional R4 branch is a small ensemble or supported missing signal; if it underperforms, use the best scored incumbent as reference and diagnose narrowing/transfer rather than stacking blindly. Freeze a shortlist/choice rule before the one later-audit inspection. These are plans only: **STOP before R4 pending feedback and instruction.**
