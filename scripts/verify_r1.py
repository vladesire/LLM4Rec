#!/usr/bin/env python3
"""Independent frozen-R1 verification, with no new fitting or target selection."""
import json
from pathlib import Path

import pandas as pd

from completion_pipeline import epoch, qualifying_interactions
from data_pipeline import dump, sha256
from run import validate_submission


def main():
    root=Path(__file__).resolve().parents[1]
    out=root/'experiments/R1';manifest=json.loads((out/'manifest.json').read_text())
    config=json.loads((out/'config.json').read_text());state=pd.read_pickle(root/'cache/completion/prepared.pkl')
    events=pd.read_pickle(root/'cache/completion/events.pkl')
    train,_=qualifying_interactions(events,int(events.overlap_start.min()),epoch(config['production_cutoff']))
    eligible=set(state['test_candidates'])
    pop=train.groupby('item').u.nunique().to_dict()
    fallback=sorted(eligible,key=lambda i:(-pop.get(i,0),i))
    table=pd.read_csv(root/manifest['csv'],dtype=str,keep_default_na=False)
    checks=validate_submission(table,state['template'],state['test_candidates'])
    profiles={int(u):g for u,g in train.loc[train.item.isin(eligible)].groupby('u')}
    independent=[]
    for u in state['evaluation_users']:
        profile=profiles.get(u)
        ranked=[] if profile is None else profile.sort_values(['count','last_end','item'],ascending=[False,False,True]).item.head(5).tolist()
        for item in fallback:
            if len(ranked)==5:break
            if item not in ranked:ranked.append(item)
        independent.append(' '.join(map(str,ranked)))
    assert independent==table.tv_show_id.tolist(),'Independent dataframe ranking differs from immutable file'
    assert sha256(root/manifest['csv'])==manifest['sha256']
    assert sha256(root/'submission/final_submission.csv')==sha256(root/'submissions/R0_global_popularity.csv')
    for name,digest in json.loads((root/'results/environment.json').read_text())['script_sha256'].items():
        assert sha256(root/'scripts'/name)==digest,'Original R0 executed source changed'
    examples=json.loads((out/'original_row_worked_examples.json').read_text())
    for example in examples:
        rows=example['original_view_csv_records']
        assert min(epoch(r['record']['start_time']) for r in rows)==epoch(example['session_start'])
        assert max(epoch(r['record']['stop_time']) for r in rows)==epoch(example['session_end'])
        original=example['original_schedule_record'];start=epoch(pd.to_datetime(original['start_time'],format='%d.%m.%Y %H:%M:%S'))
        end=start+int(original['duration'])
        overlap=max(0,min(epoch(example['session_end']),end)-max(epoch(example['session_start']),start))
        assert overlap==example['overlap_seconds']
        assert int(original['duration'])==example['original_broadcast_duration_seconds']
    ap=json.loads((out/'real_user_ap.json').read_text())
    contribution=sum(r['contribution'] for r in ap['rank_calculation'])
    assert abs(contribution/ap['denominator']-ap['ap_at_5'])<1e-12
    assert manifest['late_corrected_audit_inspected'] is False
    checks.update({'independent_dataframe_score_to_list_parity':True,'sha256_matches':True,
                   'r0_original_csv_and_executed_sources_unchanged':True,'raw_timestamp_and_schedule_examples_checked':len(examples),
                   'real_user_ap_arithmetic_checked':True,'late_corrected_audit_remains_sealed':True,
                   'no_additional_model_fitting':True,'kaggle_upload_executed':False})
    dump(out/'verification.json',checks)
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
