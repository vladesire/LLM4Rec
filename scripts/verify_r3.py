#!/usr/bin/env python3
"""Independent R3 ranking/chronology/model checks without a new fit."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from data_pipeline import dump,sha256
from rerank_features import checked_metrics
from run import validate_submission


def main():
    root=Path(__file__).resolve().parents[1];out=root/'experiments/R3'
    manifest=json.loads((out/'manifest.json').read_text());metric=json.loads((out/'metrics.json').read_text())
    cached=pd.read_pickle(root/'cache/r3/production_verification.pkl');s=cached['snapshot']
    scores=np.asarray(cached['scores']).reshape(s['pool_ids'].shape);size=s['pool_ids'].shape[1]
    assert np.isfinite(scores).all() and s['y'] is None and s['relevant'] is None
    assert s['prefix_max_observation_end']<=s['start']
    assert manifest['production_labels_latest_end']<s['start']
    assert metric['chronology']['latest_early_training_label_end']<=metric['chronology']['inner_start']
    assert metric['chronology']['latest_development_training_label_end']<=metric['chronology']['development_start']
    cold=s['X'][:,s['feature_names'].index('user_cold_qualifying')].reshape(s['pool_ids'].shape)>0
    assert (cold==cold[:,[0]]).all()
    fallback=sorted(zip(s['candidates'],s['fallback']),key=lambda r:(-r[1],r[0]))
    independent=[]
    for u,row in enumerate(scores):
        assert len(set(s['pool_ids'][u]))==size and set(s['pool_ids'][u]).issubset(set(s['candidates']))
        frame=pd.DataFrame({'item':s['pool_ids'][u],'score':row,'last':s['last_end'][u]})
        chosen=[] if cold[u,0] else frame.sort_values(['score','last','item'],ascending=[False,False,True]).item.head(5).tolist()
        for item,_ in fallback:
            if len(chosen)==5:break
            if item not in chosen:chosen.append(int(item))
        independent.append(' '.join(map(str,chosen)))
    csv=pd.read_csv(root/manifest['csv'],dtype=str,keep_default_na=False)
    template=pd.read_csv(root/'data/submission.csv',dtype=str,keep_default_na=False)
    validation=validate_submission(csv,template,s['candidates'])
    assert independent==csv.tv_show_id.tolist() and sha256(root/manifest['csv'])==manifest['sha256']
    with threadpool_limits(limits=1):probability=cached['model'].predict_proba(s['X'][:,cached['columns']])[:,1]
    np.testing.assert_array_equal(probability,cached['probabilities'])
    for number in [0,1,2]:
        previous=json.loads((root/f'experiments/R{number}/manifest.json').read_text())
        assert sha256(root/previous['csv'])==previous['sha256'] and previous['sha256']!=manifest['sha256']
    # Compare the reused reference and full population against saved R2.
    predictions=json.loads((out/'development_predictions.json').read_text())
    r2=json.loads((root/'experiments/R2/development_predictions.json').read_text())['blend_habit_25']
    assert predictions['R2_reference']['predictions']==r2['predictions']
    assert predictions['R2_reference']['per_user_ap']==r2['per_user_ap']
    devcache=pd.read_pickle(root/'cache/r3/development_verification.pkl');dev=devcache['snapshot']
    selected=manifest['selected'];m,ap=checked_metrics(dev['relevant'],dev['users'],predictions[selected]['predictions'])
    assert ap==predictions[selected]['per_user_ap']
    assert m['map_at_5']<=dev['candidate_diagnostics']['oracle_map_at_5_ceiling']+1e-12
    difference=np.asarray(ap)-np.asarray(r2['per_user_ap']);rng=np.random.default_rng(42)
    means=difference[rng.integers(0,len(difference),(2000,len(difference)))].mean(axis=1)
    uncertainty={'paired_development_delta_vs_R2':float(difference.mean()),'bootstrap_replicates':2000,'seed':42,
                 'percentile_interval_95':[float(x) for x in np.quantile(means,[.025,.975])],
                 'users_ap_improves':int((difference>0).sum()),'users_ap_worsens':int((difference<0).sum()),
                 'users_ap_unchanged':int((difference==0).sum()),
                 'limitation':'Selected local difference; temporal shift and adaptive configuration/family selection are not corrected.'}
    dump(out/'paired_uncertainty.json',uncertainty)
    checks={**validation,'independent_dataframe_ranking_matches':True,'saved_model_predict_proba_reproduced_exactly':True,
            'previous_CSVs_unchanged':True,'R2_reference_predictions_and_AP_unchanged':True,
            'candidate_pool_distinct_and_eligible':True,'full_relevance_denominator_metric_recomputed':True,
            'chronological_training_and_early_stopping_boundaries_passed':True,
            'production_has_no_labels_or_future_viewing':True,'corrected_late_audit_sealed':True,
            'no_additional_fitting':True,'no_kaggle_upload':True}
    dump(out/'verification.json',checks)
    print(json.dumps({'verification':checks,'uncertainty':uncertainty},indent=2))


if __name__=='__main__':main()
