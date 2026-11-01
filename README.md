# SWEET.TV individual LLM4Rec/FKN exam

Vladislav Neklyudov, vsneklyudov@edu.hse.ru. Local branch `master`; no agent remote push or Kaggle upload.

**Final designation: R4, user-reported private MAP@5 0.42690, public 0.42890.** Exact file: `submissions/R4_count_distribution.csv`; identity in `submission/DESIGNATED_FINAL.json`. Experiments are finished. R3 private0.42671 was almost identical; no significance claim is made.

**The report remains a draft while required student/external details are pending.** Follow `HANDOFF.md`. The historical file `submission/final_submission.csv` remains R0 unchanged and is not the designated file.

## Setup

Verified: macOS26.6.2 arm64, Python3.15.0; packages pinned in requirements.txt. Other platforms were not run. Arial (macOS) or DejaVu Sans (Linux) is needed for the PDF.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Accept the competition rules in your own account, download from https://www.kaggle.com/competitions/sweettv-tv-program-recommender/data and extract into `data/`: dataset11-30.csv, export_arh_11-20-final.csv, export_arh_21-30-final.csv, export_arh_31-42-final.csv, submission.csv. Raw data, caches, credentials and private native logs are ignored by git. Input SHA-256s are saved in round manifests/data audits.

## Evidence and reproducibility

`experiments/EVALUATION_CONTRACT.md` fixes the >=80% continuous-session qualifying-broadcast count proxy and June1-June29 development split. All1,260 users remain, including empty targets scored0; repeats allowed; scheduled nonzero candidates; common qualifying distinct-user popularity fallback. `experiments/R1` independently evaluates popularity, content, ALS and personal frequency. R2 adds recency/habit; R3 a chronological boosted reranker; R4 a small count-distribution blend. No future viewing features are used. The later corrected audit is used for R4 selection, not claimed independent.

| Artifact | Purpose |
|---|---|
| SUBMISSION_LOG.md | All returned scores, immutable files and selection |
| experiments/R0-R4/ | Original configurations, comparisons, checks, times and source/code identities |
| experiments/final/two_user_examples.json | Same two users across all methods, ordered IDs/titles/categories/scores/AP |
| experiments/final/PERSONAL_REVIEW.md | Readable examples for student interpretation |
| experiments/R1/worked_completion_checks.md | Actual original-row joins and cutoff example |
| experiments/R1/real_user_ap.json | Real metric hand-check |
| report/student_details.json | Personal statements, GitHub, submission evidence |
| report/report.md, report/report.pdf | Consolidated report source/PDF |
| HANDOFF.md | Remaining course hand-in actions |

Round runners preserve original evidence by default. Cached reconstruction commands:

```sh
.venv/bin/python scripts/run_r1.py --reproduce
.venv/bin/python scripts/run_r2.py --reproduce
.venv/bin/python scripts/run_r3.py --reproduce
.venv/bin/python scripts/run_r4.py --reproduce
```

For a fresh checkout without caches, first bootstrap into ignored locations (avoids overwriting historical R0 outputs), then use the four commands above:

```sh
.venv/bin/python scripts/run.py --output-dir cache/bootstrap_R0_results --submission-dir cache/bootstrap_R0_submission
```

R4 reproduction replays the frozen procedure including its already-used audit; it is not a new untouched validation. Original pipeline and cached R1-R3 reproductions plus R4 production resampling ran; a second clean-clone full rerun did not. CPU timings exclude dependency installation unless explicitly stated.

For final read-only checks and PDF rebuild:

```sh
.venv/bin/python scripts/final_checks.py
.venv/bin/python scripts/build_report.py
.venv/bin/python scripts/inspect_pdf.py
```

Put the genuine R4 score screenshot at `report/assets/kaggle_private_score.png`, update student_details.json, then rebuild and inspect every page. No screenshot or personal verification is fabricated. Earlier ChatGPT planning is disclosed separately from local Codex. The original native session is JSONL; homework section4.1 accepts tool-native format, while its naming examples use .json. Preserve the original unchanged, not a converted transcript.
