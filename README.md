# Individual LLM4Rec/FKN exam project

Vladislav Neklyudov · vsneklyudov@edu.hse.ru · 1 November 2026.

The genuine local implementation compares popularity, sparse content filtering and implicit ALS on one fixed temporal hold-out. **Popularity wins: MAP@5 0.5957169312**, versus 0.1779153439 (content) and 0.1579761905 (ALS). No tuning or hybrid was run. The exact designated output is `submission/final_submission.csv`. The code never uploads or submits anything.

**The report is a draft** until the student supplies GitHub/Kaggle evidence, personally checks the worked examples, rewrites their interpretation, and resolves native JSONL audit acceptance. See `HANDOFF.md`.

## Setup and data

Verified environment: Python 3.15.0, macOS arm64; other platforms were not run. CPU implicit 0.7.3 built successfully with the installed Apple compiler. PDF rendering needs Arial on macOS or DejaVu Sans on Linux.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Sign in to the [competition data page](https://www.kaggle.com/competitions/sweettv-tv-program-recommender/data), accept its rules yourself, and download/extract these files into `data/`:

- `dataset11-30.csv`
- `export_arh_11-20-final.csv`
- `export_arh_21-30-final.csv`
- `export_arh_31-42-final.csv`
- `submission.csv` (official template, unchanged)

Raw files total approximately 890 MB. Data, cache and credentials are ignored by git. Exact input byte counts and SHA-256 hashes are in `results/data_audit.json`; actual package versions, seed and parameters are saved in `results/environment.json` and `config.json`.

## Run and reproduce

From the repository root:

```sh
.venv/bin/python scripts/run.py
.venv/bin/python scripts/verify.py
.venv/bin/python scripts/build_report.py
.venv/bin/python scripts/inspect_pdf.py
```

The entry point joins data, evaluates each method, selects the highest held-out MAP@5, refits that strategy on all eligible pre-test history, and writes/validates the CSV. Input/code/config fingerprints govern preprocessing-cache reuse. Model fitting runs again each time. To force a cache rebuild, use `.venv/bin/python scripts/run.py --rebuild-data`.

The first successful entry-point run here had no cache and rebuilt all data. A later small genuine-data probe reran each method twice; a second full clean rerun is **not** claimed. Saved fit/inference times are actual measurements, excluding installation and exploratory runs. The optional `--hybrid` flag exists but was not run or evaluated for this handoff.

## Construction and evaluation

- Explicitly parse naive wall-clock timestamps. Schedule duration is seconds; viewing `duraton` is never programme length. Use timestamp interval endpoints where viewing-duration values disagree.
- Remove duplicate views, union intervals within user/channel, and remove nonpositive or ambiguous overlapping schedule rows. A channel-wise sorted half-open interval join computes `max(0,min(view_end,airing_end)-max(view_start,airing_start))`, including multiple airings per view.
- Exclude `tv_show_id=0`. Require at least 300 watched seconds per user-airing after union and clipping, then sum retained seconds per user-programme.
- Fit before **2020-06-29 00:00:00**; validate on **[2020-06-29,2020-07-27)**. Clip crossing intervals before label construction. No test viewing is used.
- Fit vocabulary, IDF, profiles and ALS only on the prefix. Test schedule reading is restricted to programme-ID/channel/time/duration availability. Unknown items/users receive documented common popularity backfill.
- Every method uses the same target-window nonzero scheduled candidates, permits repeats and emits five distinct IDs. Popularity is the number of distinct fitting users with a watched label. Integer ID breaks ties.
- MAP@5 sums precision at relevant ranks and divides by `min(5,distinct relevant items)`. Average over all 1,260 template users, retaining 22 empty-relevance users as zero and six cold users. Current server metric convention is not independently verified.

## Saved evidence

| Artifact | Purpose |
|---|---|
| `results/metrics.json`, `comparison.csv`, `comparison.md` | Actual comparison and selection |
| `results/data_audit.json` | Fingerprints, parsing, dates, coverage, duplicate/join/threshold counts and stage times |
| `results/environment.json` | Actual environment, command, seed, source hashes and parameters |
| `results/validation_predictions.json` | All held-out Top-5 outputs and per-user AP |
| `results/two_user_examples.json`, `.md`, `two_user_histories.csv` | Same two users; IDs, titles, categories, scores and actual histories |
| `results/worked_checks.md`, `real_user_ap_example.json` | Readable joins and actual AP@5 example for student verification |
| `results/verification.json`, `focused_test_output.txt` | Checks actually run; small deterministic rerun scope |
| `results/submission_validation.json` | Production refit, CSV validation and identity |
| `results/experiment_notes.md`, `session_setup.md` | Real failure/fix, access/export facts; not an audit export |
| `report/report.md`, `report/report.pdf` | Generated report source and PDF draft |
| `report/student_details.json` | Student text, GitHub, Kaggle and original-audit details |
| `results/report_build.json`, `pdf_inspection.json`, `pdf_visual_review.json` | Remaining fields, automated PDF checks and actual visual review |

## Report and native audit

Edit `report/student_details.json`, save the genuine score screenshot at `report/assets/kaggle_private_score.png`, and run the build/inspection commands above. Previews go to `/tmp/llm4rec-report-preview/`. Inspect every final PDF page yourself; geometry checks cannot certify screenshot truth or legibility. The maximum is five content pages, excluding references.

`build_report.py` regenerates Markdown from evidence and the JSON, preserving personal text stored in that JSON. For direct Markdown edits, render with `.venv/bin/python scripts/render_report.py`; a subsequent build would overwrite direct Markdown changes.

Earlier ChatGPT planning is disclosed separately from local Codex work. The student chose to copy the original native file under `~/.codex` unchanged after work finishes. Its extension is **.jsonl**, not one-document **.json**; course acceptance remains to be confirmed. No conversion, reconstructed transcript or project note is an original export. The exact native path and final actions are in `HANDOFF.md`.

The local branch is `master`. The exact final CSV is in git; raw data/cache/credentials/private session files are excluded. No remote push or Kaggle upload was executed by the agent.
