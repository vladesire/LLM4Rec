**Final update:** GitHub URL and genuine screenshot are supplied and integrated. Screenshot confirms R0 public0.28078/private0.27750 and R1-R4 scores. Personal join arithmetic and confirmation of the corrected AP (2.75/5=0.55) remain incomplete; final student review and remote push remain student actions. Older pending-evidence statements below are superseded by this update.

# Final student handoff - 1 November 2026

**Experiments are finished. R4 is designated by the highest returned private score: 0.42690.** R3 was 0.42671; the difference is only 0.00019 and is not evidence of statistically reliable superiority. All scores are user-reported pending screenshot inspection.

## Finish these before 19:00 Europe/Moscow

1. Answer the personal verification/interpretation questions. Read `experiments/final/PERSONAL_REVIEW.md`, `experiments/R1/worked_completion_checks.md` and the two users' histories. Personally recompute one original-row join, one actual AP@5, and explain an assumption. Write the discussion in your own words. The agent must not claim these happened until you confirm them.
2. Save the genuine **R4** submission-page screenshot to `report/assets/kaggle_private_score.png`. It must identify the submission and show its **private 0.42690**. Supply ID, COMPLETE status and the actual description (full Russian name and Kaggle username). Confirm the uploaded CSV identity. Do not resubmit the CSV.
3. Create/push your GitHub remote on `master` and provide the accessible URL. Implementation and the exact R4 CSV must be included; raw data, cache, credentials and the private session must not be included. The agent does not push.
4. Fill the supplied personal/external fields in `report/student_details.json`, rebuild, and personally inspect all PDF pages. A report with unresolved required placeholders is a **draft**. Do not mark `student_review_complete` until you have read the final report.
5. After the final work, copy the **original native session unchanged** using your genuine ~/.codex mechanism. Located source: `/Users/roark/.codex/sessions/2026/11/01/rollout-2026-11-01T14-39-42-01a196e8-8117-7b61-a05e-d4d86d8f1472.jsonl`. Do not edit, convert, sanitize or recreate it. The guidelines section4.1 says tool-native format is accepted, though its example name is session.json. Ours is JSONL; preserve its native extension and flag that distinction if the course upload requires .json. Earlier ChatGPT planning is separate and not covered by the local export. Do not copy the still-growing log before the work it should cover finishes.
6. Submit **exactly two course files** using the announced channel: final report PDF and original native AI-session export. Suggested report name: `Vladislav_Neklyudov_FKN_exam_report.pdf`. Preserve the native log unchanged with its native filename/format, or use a permitted filename-only rename if required. CSVs go to Kaggle/GitHub, not as extra course attachments.

## Exact designated file

`submissions/R4_count_distribution.csv`

SHA-256 `e3e6e7a74d12c3ffb8bc15054b4a5373007d1f1e1eb0e019fbc89c4fa08da2d6`

`submission/DESIGNATED_FINAL.json` records selection and identity. **`submission/final_submission.csv` is the historical R0 file**, retained unchanged for reproducibility; it is NOT the designated final file. All immutable round CSVs and original metrics remain intact.

## Rebuild after inserting details/screenshot

From the repository root:

```sh
.venv/bin/python scripts/build_report.py
.venv/bin/python scripts/inspect_pdf.py
.venv/bin/python scripts/final_checks.py
```

PDF: `report/report.pdf`. Source: `report/report.md`. Preview images: `/tmp/llm4rec-report-preview/page-1.png` etc. Check five content pages maximum (references excluded), no clipping, correct lists/tables, and readable score screenshot. Long personal text may need concise editing to preserve the page cap. JSON edits preserve personal text across builds; direct Markdown edits are overwritten by build_report.py. Required open fields are machine-listed in `report/pending_requirements.json`.

## What actually ran

Original pipeline completed from raw files; independent content and genuine ALS plus popularity were evaluated. R1/R2/R3 cached reproductions passed; R4 full production resampling matched its immutable CSV. R4 distribution and expected-AP mathematical checks passed. Final read-only checks validate all five hashes, R4 schema/user order/eligible five distinct IDs and highest returned private score. Two R4 formal examples were reconstructed from the fixed seed and checked against saved predictions. No second full clean-clone rerun is claimed. PDF geometry/glyph/page checks and visual review are recorded separately and tied to the current PDF hash.

The corrected June development task differs from historical R0 broad five-minute labels. R4 opened the corrected later audit and used it for selection; four adaptive private-score queries are not an independent test. The report preserves negative results and count-model limitations. No further experiments or uploads are needed.

See `experiments/final/COMPLIANCE.md` for requirements versus current evidence. This file and the PDF are deliverables/evidence notes, not substitutes for the original audit log.
