#!/usr/bin/env python3
"""Generate an evidence-backed report source and PDF; preserves personal edits in JSON."""
from __future__ import annotations

import json
from pathlib import Path

from data_pipeline import dump
from render_report import render


def read(path):
    return json.loads(Path(path).read_text())


def main():
    m, a = read('results/metrics.json'), read('results/data_audit.json')
    v, e = read('results/verification.json'), read('results/environment.json')
    examples = read('results/two_user_examples.json')
    ap = read('results/real_user_ap_example.json')
    final = read('results/submission_validation.json')
    s = read('report/student_details.json')
    history = __import__('pandas').read_csv('results/two_user_histories.csv', dtype={'user_id': str})
    required = ['github_url', 'personal_interpretation', 'personal_join_check', 'personal_ap_check',
                'kaggle_submission_id', 'kaggle_submission_description', 'kaggle_submission_status',
                'kaggle_private_score', 'kaggle_current_quota', 'kaggle_uploaded_csv_sha256',
                'kaggle_account_behavior', 'native_export_filename']
    pending = [key for key in required if not str(s.get(key, '')).strip() or str(s[key]).startswith('TODO:')]
    for flag in ['personal_checks_completed', 'student_review_complete', 'native_format_course_acceptance_confirmed']:
        if not s.get(flag):
            pending.append(flag)
    screenshot = Path('report') / s['score_screenshot']
    if not screenshot.exists():
        pending.append('genuine private-score screenshot')
    if s['kaggle_uploaded_csv_sha256'] != final['sha256']:
        pending.append('uploaded CSV identity confirmation')
    if s['kaggle_submission_status'] != 'COMPLETE':
        pending.append('Kaggle COMPLETE status')
    status = 'DRAFT — required student and Kaggle evidence pending' if pending else 'Final student-reviewed report'
    fmt = lambda x: f'{x:,}'
    p, c, als = (m['models'][name] for name in ['popularity', 'content', 'als'])
    full_label_fraction = (a['fit_join']['labelled_matched_seconds'] + a['validation_join']['labelled_matched_seconds']) / (a['fit_join']['view_seconds'] + a['validation_join']['view_seconds'])
    lines = [
        '# Temporal TV Programme Recommendation',
        f"**{s['student_name']}** · {s['student_email']}",
        '**Individual LLM4Rec/FKN assignment · 1 November 2026**',
        f"GitHub repository: {s['github_url']}",
        f'**{status}**',
        '', '## Abstract',
        f"This work predicts five television programme IDs per user from implicit viewing logs and broadcast schedules. I compare distinct-user popularity, sparse TF-IDF content filtering, and confidence-weighted implicit ALS on a chronological four-week hold-out. Half-open interval overlap, a five-minute per-airing threshold, and prefix-only learning produce {fmt(m['fitting_interactions'])} fitting user–programme pairs. Popularity achieves MAP@5 of {p['map_at_5']:.6f} across all {fmt(m['evaluation_users'])} official-template users, outperforming both personalized methods. It is therefore refitted on eligible pre-test history for the designated CSV. The result supports a simple recurring-programme baseline for these labels; it does not establish individual enjoyment. Private-score evidence and the student's personal review remain pending in this draft.",
        '', '## 1. Introduction and related work',
        'The task is to rank programmes that will actually air. For example, a viewer who repeatedly watches a news programme could receive its next scheduled occurrence. Programme IDs recur across broadcasts, so previously watched items remain eligible. Availability, channel changes, and unlabelled airtime make a direct viewing-log-to-item mapping unreliable.',
        'The supplied competition files [1] provide implicit feedback rather than ratings. Hu, Koren and Volinsky [2] motivate separating binary preference from confidence in an observation. Content filtering instead compares a profile of consumed attributes with programme features. I use sparse TF-IDF as documented by scikit-learn [3], and the CPU implicit implementation [4] for a genuine factor model. No language model or deep neural recommender is trained.',
        '', '## 2. Data and label construction',
        f"The viewing CSV contains {fmt(a['viewing']['raw_rows'])} rows and {fmt(a['viewing']['raw_users'])} users, from {a['viewing']['min_start']} to {a['viewing']['max_stop']}. All CSVs use comma delimiters and UTF-8-compatible text. User IDs are preserved as strings, including values above signed 64-bit range. Schedule dates use day.month.year; viewing dates use year-month-day. Both are parsed onto the same naive wall-clock axis; the files do not establish a timezone.",
        f"Schedule duration is in seconds (first-file median {a['schedules']['export_arh_11-20-final.csv']['duration_seconds_quantiles']['0.5']:.0f}); viewing duraton is an interval length, not programme length. {a['viewing']['duraton_timestamp_mismatch_rows']} values disagree with timestamp subtraction, so start/stop timestamps define the joins. Direct vsetv_id=channel_id matching leaves five observed channels unmapped: 149, 1174, 1508, 1509 and 1510. No invented remapping is used.",
        f"Nonzero-ID row coverage is {100*a['schedules']['export_arh_11-20-final.csv']['nonzero_fraction']:.2f}% in the first schedule and {100*a['schedules']['export_arh_21-30-final.csv']['nonzero_fraction']:.2f}% in the second. Zero IDs are excluded from matrices and recommendations. {a['viewing']['exact_duplicate_intervals_removed']} duplicate viewing intervals are removed; overlapping/adjacent intervals within each user/channel are unioned ({fmt(a['viewing']['overlapping_or_adjacent_intervals_merged'])} merges). The schedule has {a['canonical_schedule']['nonpositive_rows_removed']} nonpositive-duration row and {a['canonical_schedule']['overlapping_airings_removed']} conflicting airings; all are removed, including {a['canonical_schedule']['labelled_overlapping_airings_removed']} labelled airing. This conservative policy prevents duplicate coverage.",
        '', '<!-- pagebreak -->', '', '## 3. Method and leakage controls',
        'For viewing [v_start,v_end) and airing [a_start,a_end), overlap is max(0, min(v_end,a_end) − max(v_start,a_start)). A channel-wise sorted interval join advances each viewing through every intersected broadcast. It is not a Cartesian join, and a long viewing may yield several labels. Visits to the same airing are summed after interval union.',
        f"A watched label requires at least {a['fit_labels']['threshold_seconds']} seconds per user–airing. This absolute threshold was chosen after inspecting viewing lengths (full-data median {a['viewing']['timestamp_interval_seconds_quantiles']['0.5']:.0f} seconds), to reduce brief zapping without requiring completion. It is not tuned. Retained airings are aggregated into user–programme seconds. Unlabelled airtime is missing information, not a dislike.",
        'Fitting uses observations before 29 June 2020 00:00; validation is [29 June,27 July 2020 00:00). Viewing is clipped at the cutoff before overlap and thresholding, so crossing intervals cannot leak future seconds. Four late weeks provide forward prediction with substantial history. This is shorter than the future schedule [27 July,19 October 2020 12:00), which limits horizon comparability. These dates come from parsed rows, not filename week numbers.',
        'All methods rank the same nonzero IDs scheduled in the target window. Repeats are allowed. Positive, finite method scores are sorted descending; integer programme ID breaks ties. Missing or insufficient method scores use the same prefix-popularity order, without duplicates. Unseen programmes with zero popularity are deterministic ID-tie backfill candidates, rather than inferred dislikes.',
        'Popularity scores each programme by the number of distinct fitting users with a retained label. This definition also supplies every fallback. It uses neither held-out viewing nor future metadata.',
        'Content filtering uses one earliest available metadata row per nonzero programme before the fitting cutoff. Features are title word tokens, category and three genres (field tokens repeated twice), year, director, and up to 20 actor names. Prefix-only vocabulary and smoothed IDF produce L2-normalized sparse item vectors. A user profile is the L2-normalized weighted mean with weight log1p(retained seconds/300). Cosine similarity ranks candidates with prefix metadata; items without that metadata receive no content score. No held-out or test metadata enters vocabulary, IDF, or profiles.',
        'ALS minimizes sum_ui c_ui(p_ui − x_u·y_i)^2 + λ(sum_u ||x_u||² + sum_i ||y_i||²). Observed pairs have p_ui=1 and c_ui=1+10 log1p(retained seconds/300); unobserved pairs have p_ui=0,c_ui=1. Settings are 32 factors, 15 exact ALS iterations, λ=0.1, seed 42, float32 and one CPU thread; implicit alpha=1 avoids a second confidence scaling. Unknown items and cold users use the common fallback. It is matrix factorization, not nearest-neighbour similarity.',
        f"The installed environment is {e['machine']} {e['platform'].split('-')[0]}, Python {e['python'].split()[0]}, NumPy {e['packages']['numpy']}, pandas {e['packages']['pandas']}, SciPy {e['packages']['scipy']}, scikit-learn {e['packages']['scikit-learn']} and implicit {e['packages']['implicit']}. Sparse matrices and a fingerprinted local preprocessing cache keep the pipeline small. Exact package versions, input SHA-256 values, parameters and stage timings are saved.",
        '', '<!-- pagebreak -->', '', '## 4. Experiments, checks and discussion',
        f"Fitting contains {fmt(m['fitting_users'])} labelled users, {fmt(m['fitting_interactions'])} distinct user–programme pairs and {fmt(p['fitting_interaction_items'])} interacted items. Validation has {fmt(m['held_out_interactions_all_log_users'])} labelled pairs across log users; {fmt(m['held_out_interactions_template_users'])} belong to the {fmt(m['evaluation_users'])} official-template users used for primary evaluation. There are {fmt(m['candidate_items'])} available candidate items.",
        'AP@5 sums precision at each relevant recommendation rank and divides by min(5, number of distinct relevant items). The primary MAP averages over every template user. Empty relevance contributes zero; relevance sets and outputs are distinct, and duplicate predictions are rejected. This follows the supplied brief. Current authenticated Kaggle metric/account behavior was unavailable, so exact server convention remains to be confirmed rather than assumed identical.',
        f"The population includes {p['empty_relevance_users_included_as_zero']} users without held-out labels and {p['cold_users']} users without fitting labels. No unreachable relevant user–item labels are removed ({m['unreachable_distinct_user_item_labels']} observed). Secondary nonempty-user MAP is {p['map_at_5_nonempty_secondary']:.6f} for popularity; warm/cold MAP is {p['warm_map_at_5']:.6f}/{p['cold_map_at_5']:.6f}. All comparisons below use the identical split and policy.",
        '', '| Method | MAP@5 | Fit (s) | Infer (s) | Scored-slot coverage | Distinct Top-5 IDs |', '|---|---:|---:|---:|---:|---:|'
    ]
    for name, row in m['models'].items():
        lines.append(f"| {name.title()} | {row['map_at_5']:.6f} | {row['fit_seconds']:.3f} | {row['inference_seconds']:.3f} | {100*row['method_slots_fraction']:.2f}% | {row['distinct_recommended_items']} |")
    lines += [
        '', f"Content has metadata for {c['method_known_candidates']}/{m['candidate_items']} candidates; ALS has fitted factors for {als['method_known_candidates']}/{m['candidate_items']}. Both use full popularity fallback for the six cold users. Preparation took {a['stage_seconds']['total_preparation_including_fingerprints']:.3f} s and the successful entry point {e['elapsed_entry_point_seconds']:.3f} s in this installed environment, excluding setup and exploratory runs. These are observations, not runtime guarantees.",
        f"Only one configuration of each individual method was compared; there was no hyperparameter search, hybrid or Kaggle score tuning. The same hold-out selected popularity, so results are selection-biased rather than an independent final evaluation. The production refit uses {fmt(a['production_labels']['unique_user_item_interactions'])} pre-test pairs and {fmt(a['candidate_policy']['test_items'])} future candidates. Its validation score is not recomputed on refitted history.",
        f"The join assigns {100*a['fit_join']['labelled_seconds_fraction']:.2f}% of fitting and {100*a['validation_join']['labelled_seconds_fraction']:.2f}% of held-out viewing seconds to labelled airings ({100*full_label_fraction:.2f}% overall). It drops {fmt(a['fit_labels']['user_airings_dropped_below_threshold'])}/{fmt(a['validation_labels']['user_airings_dropped_below_threshold'])} below-threshold fitting/held-out user–airings. Missing schedules and unlabelled broadcasts limit observed relevance. Background viewing, unequal exposure, and the shorter validation horizon further limit interpretation.",
        'Popularity wins under these broad recurring-TV labels but recommends just five distinct IDs to the whole population. Personalized content and ALS spread exposure more widely and score worse; their similarity/confidence objectives do not directly optimize this AP@5 target. Retaining the measured winner is a practical mitigation, not a claim that diversity or personal preference has been solved. No live A/B, enjoyment, retention or revenue result is claimed.',
        f"An actual preparation failure occurred when pandas Series.item resolved to a method during example serialization. Explicit column indexing fixed it; the added integration check and completed full entry point confirmed the repair. {v['focused_test_count']} focused tests passed. A scalar reference checked {fmt(v['random_interval_pairs_checked_against_scalar_formula'])} interval pairs. A {v['small_deterministic_rerun']['fitting_users']}-user, {fmt(v['small_deterministic_rerun']['fitting_interactions'])}-pair probe produced identical scores and Top-5 twice for all three methods, including when {fmt(v['small_deterministic_rerun']['future_metadata_rows_mutated'])} post-cutoff metadata rows were replaced. This is a small rerun, not a second full clean run.",
        '', '<!-- pagebreak -->', '', '## 5. The same two users across all methods',
        'Examples use observed validation viewing; true test-period viewing is unavailable. Histories and all ordered Top-5 IDs, titles, categories, method scores and fallback flags are saved in results/two_user_examples.json and results/two_user_histories.csv. The users are chosen near the 10th and 90th percentiles of fitting history length among evaluable template users, rather than selected for model wins.'
    ]
    for index, ex in enumerate(examples, 1):
        lines += ['', f"### User {index}: {ex['user_id']}",
                  f"Fitting history: {ex['training_distinct_items']} distinct programmes; held-out history: {ex['held_out_distinct_items']}.",
                  '', '| Method | Ordered Top-5 programme IDs | AP@5 |', '|---|---|---:|']
        for name, result in ex['methods'].items():
            ids = ' → '.join(str(r['item']) for r in result['recommendations'])
            lines.append(f"| {name.title()} | {ids} | {result['ap_at_5']:.6f} |")
        summaries = []
        for phase, title in [('fitting', 'Fitting'), ('held_out', 'Held-out')]:
            subset = history.loc[(history.user_id == ex['user_id']) & (history.phase == phase)].head(2)
            summaries.append(title + ': ' + '; '.join(f"{r.title} (ID {r.item}, {int(r.watched_seconds):,} s)" for r in subset.itertuples(index=False)))
        lines += ['', ' '.join(summaries)]
    lines += ['', '### Baseline titles, scores and actual relevance',
              'The ordered popularity list is shared by the two users. Scores are distinct fitting-user counts; Y/N indicates held-out relevance for user 1/user 2.',
              '', '| ID | Programme title | Category | Score | Relevant 1/2 |', '|---:|---|---|---:|---|']
    for one, two in zip(examples[0]['methods']['popularity']['recommendations'], examples[1]['methods']['popularity']['recommendations']):
        lines.append(f"| {one['item']} | {one['title'].replace('|', '/')} | {one['category']} | {one['method_score']:.0f} | {'Y' if one['held_out_relevant'] else 'N'}/{'Y' if two['held_out_relevant'] else 'N'} |")
    lines += ['', f"For user {ap['user_id']}, the saved real AP@5 example has {ap['number_distinct_relevant']} distinct relevant IDs and hits at ranks 2 and 4: (1/2+2/4)/5={ap['ap_at_5']:.6f}. This matches the recorded result. Student personal verification remains pending unless completed below.",
              'Agent-observed interpretation: ALS slightly improves on popularity for user 1; both personalized lists score zero for user 2. High fitting history volume alone therefore does not guarantee a useful forward ranking. The measured evidence supports the aggregate baseline choice while retaining individual failure examples.',
              f"**Student interpretation:** {s['personal_interpretation']}",
              f"**Personal join check:** {s['personal_join_check']}", f"**Personal AP@5 check:** {s['personal_ap_check']}",
              '', '<!-- pagebreak -->', '', '## 6. External validation and AI disclosure',
              f"**Kaggle submission:** {s['kaggle_submission_id']}. Status: {s['kaggle_submission_status']}. Private score: {s['kaggle_private_score']}.",
              f"Description: {s['kaggle_submission_description']}. Observed quota: {s['kaggle_current_quota']}. Current account behavior: {s['kaggle_account_behavior']}.",
              f"The designated file is submission/final_submission.csv ({final['rows']} rows, exact template order and schema). SHA-256: {final['sha256']}. Uploaded identity: {s['kaggle_uploaded_csv_sha256']}.",
              '', f"![Genuine designated final Kaggle submission, including PRIVATE score and submission identity]({s['score_screenshot']})", '',
              'ChatGPT assisted with planning before the local implementation session. That upstream activity is disclosed separately and is not represented as part of the local Codex audit. In this genuine local session, Codex inspected data, wrote and debugged code, ran joins/models/checks, selected the recorded baseline, and drafted documentation and this PDF. Web browsing consulted the sources below; Python libraries generated numerical outputs and the PDF. No subagents or other model APIs were used.',
              f"The agent ran the checks recorded in results/verification.json. Student personal checks completed: {s['personal_checks_completed']}; full report review completed: {s['student_review_complete']}. Library implementations and source descriptions were trusted within the documented checks; the competition server metric, account quota and private-score behavior were not verified by the agent. Numerical results come from saved outputs, not generated prose.",
              f"Original local audit reference: {s['native_export_filename']}. Native format: {s['native_export_format']}. {s['audit_export_note']} The located native session is rollout-2026-11-01T14-39-42-01a196e8-8117-7b61-a05e-d4d86d8f1472.jsonl. This is JSONL, not a single JSON document; course acceptance confirmed: {s['native_format_course_acceptance_confirmed']}. The agent has not copied, reconstructed, edited, sanitized, simulated or converted that original. Saved chat history is distinct from earlier external planning [5].",
              '', '<!-- pagebreak -->', '', '## References',
              '[1] sweet.tv, “sweet.tv – TV program recommender,” Kaggle, competition data. [Competition data page](https://www.kaggle.com/competitions/sweettv-tv-program-recommender/data), accessed Nov. 1, 2026. Local downloaded CSVs were inspected; the public page exposed no readable metric details.',
              '[2] Y. Hu, Y. Koren, and C. Volinsky, “Collaborative Filtering for Implicit Feedback Datasets,” in Proc. IEEE ICDM, 2008, pp. 263–272, doi: 10.1109/ICDM.2008.22. [Author-hosted paper](https://yifanhu.net/PUB/cf.pdf), consulted Nov. 1, 2026.',
              '[3] scikit-learn developers, “Feature extraction,” scikit-learn documentation. [Official feature-extraction documentation](https://scikit-learn.org/stable/modules/feature_extraction.html), consulted Nov. 1, 2026.',
              '[4] B. Frederickson and implicit contributors, “AlternatingLeastSquares,” Implicit 0.7.2 API documentation. [CPU ALS documentation](https://benfred.github.io/implicit/api/models/cpu/als.html), consulted Nov. 1, 2026. Installed library version: 0.7.3.',
              '[5] OpenAI, “Projects and chats,” ChatGPT/Codex documentation. [Official chat documentation](https://learn.chatgpt.com/docs/projects), consulted Nov. 1, 2026. It describes saved transcripts; it does not establish a JSON export mechanism for this session.', '']
    Path('report/report.md').write_text('\n\n'.join(lines).replace('\n\n|', '\n|').replace('|\n\n', '|\n'))
    render('report/report.md', 'report/report.pdf')
    dump('results/report_build.json', {'status': status, 'pending_required_fields': list(dict.fromkeys(pending)),
                                       'genuine_score_screenshot_present': screenshot.exists(),
                                       'source': 'report/report.md', 'pdf': 'report/report.pdf',
                                       'student_name': s['student_name'], 'student_email': s['student_email']})
    print('Built report/report.md and report/report.pdf;', status)


if __name__ == '__main__':
    main()
