# Student handoff

**Historical R0 handoff.** Current sequential research instructions, file identity and waiting status are in `SUBMISSION_LOG.md` and `experiments/R2/HANDOFF.md`. The old R0 upload instructions below are archival; do not consume another attempt on R0. The report is still a draft and will be consolidated after all returned round results.

**Deadline: 1 November 2026, 19:00 Europe/Moscow.** Code, recorded comparison, valid CSV and draft report are ready. Personal and external requirements remain below.

## What ran

The full `.venv/bin/python scripts/run.py` entry point completed without a prior cache, comparing all three individual methods and refitting the measured winner. Popularity MAP@5 is **0.595717**, content **0.177915**, ALS **0.157976**. Fit/inference times are 0.015/0.252 s, 0.186/0.467 s, and 3.962/0.303 s respectively. Exact saved numbers are in `results/metrics.json`; the comparison includes all 1,260 official-template users.

`.venv/bin/python scripts/verify.py` passed six focused tests, 10,000 scalar-reference interval comparisons, leakage-boundary checks, exact CSV validation and a small genuine-data deterministic rerun of every method. A second full clean rerun is not claimed. The actual pandas example-serialization failure, fix and confirming run are documented. There was no tuning, hybrid or score fishing.

## 1. Upload the one designated CSV

**File:** `submission/final_submission.csv`

**SHA-256:** `73d71295c19c61c3702a3c8e0fae951f27a307ce9b05cd39fd37e9b93e0996e0`

```sh
shasum -a 256 submission/final_submission.csv
```

The student authorized one smoke submission, then chose to upload it personally. They reported Kaggle showing **100 submissions**, versus five in historical instructions; remaining quota was not independently checked. The agent has not submitted anything, and local CLI authentication is absent. Use signed-in **Late Submission**, entering your full Russian name and Kaggle username in the description. A successful smoke submission can remain the designated final submission; no additional uploads are needed for this handoff.

Wait for **COMPLETE**, record the identifier/description and returned **private** score, and save a genuine submission-page screenshot at `report/assets/kaggle_private_score.png`. If late submission/private-score visibility is unavailable, record the actual behavior and contact the course channel immediately; do not invent a result.

## 2. Personally check and interpret

Read `results/worked_checks.md` and inspect the raw viewing/schedule rows. Recompute at least one join and the actual AP@5 example personally. User `8848657562275607589` has 22 relevant IDs and hits at ranks 2 and 4: `(1/2+2/4)/5=0.2`. The complete relevance set is in `results/real_user_ap_example.json`.

Review both example users' actual fitting/held-out histories and all three lists. Rewrite the interpretation in your own words, explaining the evidence and limitations. Update the personal text/check/review fields in `report/student_details.json` only after doing these checks. Agent verification is not student verification.

## 3. Push the prepared repository

The local branch is `master`; the student creates and pushes the remote. For example:

```sh
git remote add origin YOUR_ACCESSIBLE_GITHUB_REPOSITORY_URL
git push -u origin master
```

Keep raw data, cache, credentials and private native session files outside git. The exact final CSV belongs in GitHub. Insert the accessible repository URL into `report/student_details.json`.

## 4. Finalize and inspect the report

Identity is already Vladislav Neklyudov, vsneklyudov@edu.hse.ru. Fill all `TODO:` fields in `report/student_details.json`, add the genuine private-score screenshot, record the uploaded CSV hash and actual account behavior/quota, and complete personal review. Disclose earlier ChatGPT planning separately from local Codex work.

```sh
.venv/bin/python scripts/build_report.py
.venv/bin/python scripts/inspect_pdf.py
```

Open and inspect `report/report.pdf`. The inspected draft has **five content pages plus one references page**, readable Cyrillic and tables, and no observed clipping. Genuine screenshot legibility remains unchecked until inserted. The inspection script checks bounds/page count but does not visually review for you. If personal text or the screenshot adds a page, shorten the text. `results/report_build.json` lists unresolved requirements. Until they are resolved, the PDF is a **draft**.

For direct Markdown edits, render with `.venv/bin/python scripts/render_report.py`; rebuilding from JSON overwrites direct Markdown changes.

## 5. Copy the original native session unchanged

The student selected the native-file route. The genuine located original is:

```text
/Users/roark/.codex/sessions/2026/11/01/rollout-2026-11-01T14-39-42-01a196e8-8117-7b61-a05e-d4d86d8f1472.jsonl
```

It is **JSONL**, not a single JSON document. Confirm course acceptance of that format; changing a suffix or converting does not create an original export. The agent has not copied, reconstructed, edited, sanitized or simulated this original. Project results/notes are not audit substitutes. Earlier ChatGPT planning is not covered by this local file.

Enter the native filename and confirmed format status in the report before its final rebuild. After this chat's final response and all work/checks finish, copy the original unchanged outside git. Student-run example:

```sh
mkdir -p "$HOME/LLM4Rec-handin"
cp -p /Users/roark/.codex/sessions/2026/11/01/rollout-2026-11-01T14-39-42-01a196e8-8117-7b61-a05e-d4d86d8f1472.jsonl "$HOME/LLM4Rec-handin/"
cp report/report.pdf "$HOME/LLM4Rec-handin/report.pdf"
```

If Codex performs more work after the copy, take a fresh unchanged copy at the end so it includes that work.

## 6. Hand in exactly two course files

Submit **the finalized report PDF and original unmodified local AI-session export** through the course's announced channel before **19:00 Europe/Moscow**. The CSV goes to Kaggle and GitHub, not as a third course attachment.

## Still unresolved

- Kaggle COMPLETE/private score, identifier/description, screenshot and actual private-score behavior.
- GitHub remote push and accessible URL.
- Student personal join/AP checks, rewritten interpretation and final PDF review.
- Native JSONL course acceptance and final unchanged original copy.
- The three named assignment sources were absent; work followed the self-contained brief and actual data.
