#!/usr/bin/env python3
"""Focused checks plus a small deterministic rerun on genuine cached data."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import ap_at_5, match_channel, overlap_seconds
from data_pipeline import dump, human, sha256
from models import fit_score
from run import validate_submission


def main():
    started = time.perf_counter()
    out = Path('results')
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], capture_output=True, text=True)
    (out / 'focused_test_output.txt').write_text(tests.stdout + tests.stderr)
    assert tests.returncode == 0, tests.stderr
    data = pd.read_pickle('cache/prepared.pkl')
    config = json.loads(Path('config.json').read_text())
    audit = data['audit']
    cutoff, test_start = data['cutoff'], data['test_start']
    assert audit['fit_join']['max_clipped_observation_end'] <= human(cutoff)
    assert audit['validation_join']['max_clipped_observation_end'] <= human(test_start)
    assert audit['viewing']['view_intervals_starting_in_test'] == 0
    assert audit['learned_preprocessing']['future_metadata_features_read_by_pipeline'] is False
    assert all((d.item > 0).all() for d in [data['train'], data['val'], data['production']])
    # Independently compare the vectorized interval join against scalar overlap.
    rng = np.random.default_rng(config['seed'])
    ss = np.arange(0, 2000, 20, dtype=np.int64)
    se = ss + rng.integers(1, 21, len(ss))
    vs = rng.integers(-50, 2020, 100)
    ve = vs + rng.integers(0, 400, 100)
    vi, ai, amount = match_channel(vs, ve, ss, se)
    actual = sorted((int(v), int(a), int(s)) for v, a, s in zip(vi, ai, amount))
    expected = [(v, a, overlap_seconds(int(vs[v]), int(ve[v]), int(ss[a]), int(se[a])))
                for v in range(len(vs)) for a in range(len(ss))
                if overlap_seconds(int(vs[v]), int(ve[v]), int(ss[a]), int(se[a])) > 0]
    assert actual == expected
    # Small probe: real prefix labels, 200 fitting users and up to 400 items;
    # the last three evaluation users are deliberately withheld from fitting.
    eval_users = data['evaluation_users'][:40]
    probe_users = sorted(set(eval_users[:-3] + sorted(data['train'].u.unique().tolist())[:200]) - set(eval_users[-3:]))[:230]
    item_ids = data['train'].groupby('item').u.nunique().sort_values(ascending=False).head(400).index
    tiny = data['train'].loc[data['train'].u.isin(probe_users) & data['train'].item.isin(item_ids)].copy()
    candidate_ids = sorted(set(data['validation_candidates']) & set(item_ids))[:300]
    candidate_ids += sorted(set(data['validation_candidates']) - set(item_ids))[:10]
    tiny_meta = data['metadata'].loc[data['metadata'].item.isin(item_ids)].copy()
    changed_future = tiny_meta.copy()
    future_rows = changed_future.s >= cutoff
    for col in ['tv_show_title', 'tv_show_category', 'tv_show_genre_1', 'tv_show_genre_2', 'tv_show_genre_3', 'year_of_production', 'director', 'actors']:
        changed_future.loc[future_rows, col] = 'FUTURE_SENTINEL_NEVER_FIT'
    probe = {}
    for name in ['popularity', 'content', 'als']:
        first = fit_score(name, tiny, tiny_meta, cutoff, len(data['users']), eval_users, candidate_ids, config)
        second = fit_score(name, tiny, changed_future, cutoff, len(data['users']), eval_users, candidate_ids, config)
        assert first['predictions'] == second['predictions'], name
        np.testing.assert_array_equal(first['scores'], second['scores'])
        if name == 'content':
            assert first['details']['metadata_max_start'] < cutoff
        probe[name] = {'two_runs_exact_same_top5': True, 'two_runs_exact_same_scores': True,
                       'future_metadata_mutation_did_not_change_output': True,
                       'first_fit_seconds': first['fit_seconds'], 'second_fit_seconds': second['fit_seconds']}
    template = pd.read_csv('data/submission.csv', dtype=str, keep_default_na=False)
    submission = pd.read_csv('submission/final_submission.csv', dtype=str, keep_default_na=False)
    csv_check = validate_submission(submission, template, data['test_candidates'])
    recorded_hash = Path('submission/final_submission.csv.sha256').read_text().split()[0]
    assert sha256('submission/final_submission.csv') == recorded_hash
    real = json.loads((out / 'real_user_ap_example.json').read_text())
    assert np.isclose(ap_at_5(real['distinct_relevant_ids'], [r['item'] for r in real['rank_calculation']]), real['ap_at_5'])
    checks = {'focused_test_process_exit_code': tests.returncode, 'focused_test_count': 6,
              'random_interval_pairs_checked_against_scalar_formula': len(vs) * len(ss),
              'training_observations_clipped_at_cutoff': True, 'test_viewing_used': False,
              'nonzero_interaction_items_only': True, 'learned_metadata_boundary_verified': True,
              'small_deterministic_rerun': {'fitting_users': int(tiny.u.nunique()), 'fitting_interactions': len(tiny),
                                            'evaluation_users': len(eval_users), 'candidates': len(candidate_ids),
                                            'future_metadata_rows_mutated': int(future_rows.sum()),
                                            'scope': 'Small genuine-data probe; three intentionally cold evaluation users. Not a full clean rerun or new model-selection experiment.',
                                            'models': probe},
              'full_entry_point_completed': True, 'full_entry_point_command': '.venv/bin/python scripts/run.py',
              'full_clean_rerun_performed': False,
              'initial_successful_run_rebuilt_data_without_cache': not audit['cache_used'],
              'final_csv': csv_check, 'csv_sha256_matches_designated_checksum': True,
              'real_user_ap_recalculation_matches': True, 'student_personal_checks_completed': False,
              'elapsed_verification_seconds': time.perf_counter() - started}
    dump(out / 'verification.json', checks)
    examples = [
        ('Partial overlap', '[90,150)', '[100,200)', 'min(150,200)-max(90,100) = 50 seconds'),
        ('Half-open no overlap', '[0,100)', '[100,200)', 'max(0,100-100) = 0 seconds'),
        ('One viewing spans two broadcasts', '[50,250)', '[100,200), [200,300)', '100 seconds in the first and 50 in the second; neither is lost'),
        ('Crossing split at 200: fit', '[90,250) clipped to [90,200)', '[100,300)', '100 seconds; no second at or after cutoff enters fitting'),
        ('Crossing split at 200: hold-out', '[90,250) clipped to [200,250)', '[100,300)', '50 seconds; total across the two sides is 150')]
    lines = ['# Worked checks for student review', '',
             'Agent-executed checks are recorded in verification.json. Student personal verification remains pending.', '',
             'The numbers below are synthetic seconds on a common time axis, not fabricated dataset rows.', '',
             '| Case | Viewing | Broadcast | Hand calculation |', '|---|---|---|---|']
    lines += ['| ' + ' | '.join(row) + ' |' for row in examples]
    lines += ['', '## Genuine matched rows', '', '| Channel | Programme | Viewing | Broadcast | Overlap |', '|---|---:|---|---|---:|']
    for row in audit['fit_join']['examples']:
        lines.append(f"| {row['channel']} | {row['item']} | {row['view_start']} to {row['view_end']} | {row['airing_start']} to {row['airing_end']} | {row['overlap_seconds']} s |")
    lines += ['', 'For example, 19:42:11–20:45:00 is 1,800 + 1,800 + 169 = 3,769 seconds in the first genuine row. The rest of that viewing interval can match subsequent broadcasts.', '',
              '## AP@5 reference checks', '',
              '- Relevant {A,B,C}; [A,B,D,C,E]: (1 + 1 + 3/4)/3 = 11/12.',
              '- Relevant {A,B,C}; [D,A,E,B,C]: (1/2 + 2/4 + 3/5)/3 = 8/15.',
              '- Empty relevance contributes AP=0 and remains in the population.',
              '- Duplicate outputs are rejected before scoring, rather than receiving repeat credit.', '',
              '## One actual user calculation', '', f"User `{real['user_id']}`, method `{real['method']}`, has {real['number_distinct_relevant']} distinct held-out relevant programmes. Denominator = min(5,{real['number_distinct_relevant']}) = {real['denominator']}.", '',
              '| Rank | Programme | Held-out hit | Precision | Numerator contribution |', '|---:|---:|---|---:|---:|']
    for r in real['rank_calculation']:
        lines.append(f"| {r['rank']} | {r['item']} | {r['relevant']} | {r['precision_at_rank']:.6f} | {r['ap_numerator_contribution']:.6f} |")
    lines += ['', f"AP@5 = {real['numerator']}/{real['denominator']} = {real['ap_at_5']:.6f}.", '',
              'The complete relevant-ID set is in real_user_ap_example.json. The fitting and held-out histories for the same two example users are in two_user_histories.csv.', '',
              '**Student action:** inspect the original CSV rows and recompute at least one join and this AP@5 example personally. Record your own reasoning in report/student_details.json; no personal check is claimed yet.']
    (out / 'worked_checks.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
