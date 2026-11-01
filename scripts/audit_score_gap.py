#!/usr/bin/env python3
"""Diagnose the submitted baseline without modifying its CSV or original results.

Alternative relevance definitions below are sensitivity checks, not claims
about the competition's hidden labels and not new model-selection results.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import ap_at_5
from data_pipeline import dump, prepare, sha256
from models import fit_score


def summarize_relevance(relevant, users, predictions):
    return {name: float(np.mean([ap_at_5(relevant.get(u, set()), row)
                                for u, row in zip(users, values['predictions'])]))
            for name, values in predictions.items()}


def main():
    data = pd.read_pickle('cache/prepared.pkl')
    users = data['evaluation_users']
    val = data['val']
    predictions = json.loads(Path('results/validation_predictions.json').read_text())
    submission = pd.read_csv('submission/final_submission.csv', dtype=str)
    metadata = data['metadata'].drop_duplicates('item').set_index('item')
    actual_rows = submission.tv_show_id.unique().tolist()
    submitted_ids = [int(x) for x in actual_rows[0].split()]
    reports = {}
    for k in [None, 20, 10, 5]:
        observed = val if k is None else val.sort_values(['u', 'seconds', 'item'], ascending=[True, False, True]).groupby('u', sort=False).head(k)
        rel = {int(u): set(g.item) for u, g in observed.groupby('u')}
        reports['all_watched_labels' if k is None else f'top_{k}_by_retained_watched_seconds'] = summarize_relevance(rel, users, predictions)
    lengths = val.groupby('u').item.nunique().reindex(users, fill_value=0)
    popularity_fit = data['train'].groupby('item').u.nunique()
    popularity_prod = data['production'].groupby('item').u.nunique()
    top_items = []
    for item in submitted_ids:
        row = metadata.loc[item]
        reach = int(val.loc[(val.item == item) & val.u.isin(users), 'u'].nunique())
        top_items.append({'item': item, 'title': str(row.tv_show_title), 'category': str(row.tv_show_category),
                          'fit_distinct_users': int(popularity_fit.get(item, 0)),
                          'production_distinct_users': int(popularity_prod.get(item, 0)),
                          'validation_template_user_reach': reach,
                          'validation_template_user_reach_fraction': reach / len(users)})
    # Compare the exact submitted order against the existing held-out labels.
    frozen = {'submitted_final_order': {'predictions': [submitted_ids] * len(users)}}
    rel = {int(u): set(g.item) for u, g in val.groupby('u')}
    final_order_map = summarize_relevance(rel, users, frozen)
    # Reach seen in each model's five predictions is an alternative diagnostic,
    # without changing labels or fitting any additional model.
    repeated = {}
    for name, result in predictions.items():
        seen = {int(u): set(g.item) for u, g in data['train'].groupby('u')}
        repeated[name] = float(np.mean([sum(i in seen.get(u, set()) for i in row) / 5 for u, row in zip(users, result['predictions'])]))
    result = {'reported_private_score': 0.27,
              'private_score_provenance': 'Reported by student; exact score screenshot/submission identifier not yet inspected.',
              'submission_sha256': sha256('submission/final_submission.csv'),
              'identical_recommendation_lists_in_csv': len(actual_rows),
              'submitted_programmes': top_items,
              'held_out_distinct_relevance_quantiles_template_users': {str(q): float(v) for q, v in lengths.quantile([0,.1,.25,.5,.75,.9,1]).items()},
              'template_users_with_at_least_five_labels': int((lengths >= 5).sum()),
              'mean_held_out_distinct_labels': float(lengths.mean()),
              'sensitivity_to_unverified_relevance_definitions': reports,
              'warning': 'Top-N-by-seconds definitions are diagnostic hypotheses, not verified competition rules. Original reported results are unchanged.',
              'submitted_order_on_original_holdout': final_order_map,
              'recommendation_slot_fraction_previously_watched_in_fitting': repeated}
    dump('results/score_gap_audit.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    # Match the roughly twelve-week forecast horizon using the same label rule.
    # This changes only the historical cutoff for a baseline sensitivity check.
    config = json.loads(Path('config.json').read_text())
    config['validation_days'] = 84
    horizon_data = prepare('data', 'cache/diagnostic_horizon84', 'results/diagnostics/horizon84', config)
    baseline = fit_score('popularity', horizon_data['train'], horizon_data['metadata'], horizon_data['cutoff'],
                         len(horizon_data['users']), horizon_data['evaluation_users'], horizon_data['validation_candidates'], config)
    horizon_rel = {int(u): set(g.item) for u, g in horizon_data['val'].groupby('u')}
    horizon_predictions = {'popularity': {'predictions': baseline['predictions']}}
    result['twelve_week_historical_horizon_probe'] = {
        'map_at_5': summarize_relevance(horizon_rel, horizon_data['evaluation_users'], horizon_predictions)['popularity'],
        'cutoff': horizon_data['audit']['split']['fit_cutoff_exclusive'],
        'end': horizon_data['audit']['split']['validation_end_exclusive'],
        'fitting_interactions': len(horizon_data['train']), 'candidate_items': len(horizon_data['validation_candidates']),
        'same_watch_threshold_and_population_policy': True,
        'purpose': 'Historical horizon sensitivity diagnosis, not future-score tuning or an unbiased new test.'}
    dump('results/score_gap_audit.json', result)
    print('HORIZON PROBE', json.dumps(result['twelve_week_historical_horizon_probe']), flush=True)


if __name__ == '__main__':
    main()
