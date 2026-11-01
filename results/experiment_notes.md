# Actual experiment notes

These notes are project evidence, not the original AI-session export.

- Initial inspection used the genuine five CSV files. The first schedule has 1,074,235 rows and nonzero-ID coverage 0.10762356467625799, agreeing with the brief's approximate 10.8% observation.
- The four-week hold-out was fixed at `[2020-06-29 00:00:00, 2020-07-27 00:00:00)` before evaluating any model. Viewing and airing intervals are clipped to each side before labels are constructed.
- A 300-second absolute threshold was selected after viewing-duration inspection: median timestamp-derived interval is about 397 seconds in the initial systematic sample. This threshold limits short zapping while retaining partial viewing. It is a design choice, not a tuned optimum.
- CPU implicit ALS dependency installation and a 2-user/3-item smoke fit succeeded. No alternative eligible algorithm was needed.
- **Actual runtime failure:** the first preparation run failed in example serialization because `sr.item` is a pandas Series method. It was replaced with `sr["item"]`; a small integration test now checks clipping, example serialization, thresholding, and schedule-conflict handling. Subsequent run outcomes are recorded in saved verification outputs. The failed run produced no model metrics or designated submission.
- All quoted metrics, populations, timing and submission identity must come from the JSON/CSV artifacts produced by the entry point, not these narrative notes.
- The measured comparison completed: popularity 0.5957169312169311, content 0.17791534391534392, ALS 0.1579761904761905. Only one individual configuration per method was evaluated. No hybrid or tuning was run. Popularity was selected and refitted for the designated CSV.
- All six focused tests and the full entry point passed after the pandas repair. A separate 230-user/22,850-pair probe produced exactly identical scores and Top-5 twice for each method, even after replacing 17,519 post-cutoff metadata rows. This is not a second full clean rerun.
- The student reported Kaggle showing 100 submissions, authorized one smoke upload, then chose to upload the designated file personally. No submission was executed by the agent. A Kaggle CLI help check failed on a missing `six` dependency; the student declined further installation and selected manual upload. This did not affect the working recommendation pipeline or CSV.
- Final report inspection found five content pages plus one references page, readable tables/Cyrillic and no observed clipping. A genuine score screenshot is still pending. See checksum-linked PDF inspection evidence.
