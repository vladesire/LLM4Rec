#!/usr/bin/env python3
"""Check frozen R2 output independently; no fitting or late-audit evaluation."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline import dump,sha256
from run import validate_submission


def main():
    root=Path(__file__).resolve().parents[1];out=root/'experiments/R2'
    manifest=json.loads((out/'manifest.json').read_text())
    cached=pd.read_pickle(root/'cache/completion/r2_production_verification.pkl')
    ctx,scores=cached['context'],cached['scores']
    frame=pd.read_csv(root/manifest['csv'],dtype=str,keep_default_na=False)
    template=pd.read_csv(root/'data/submission.csv',dtype=str,keep_default_na=False)
    checks=validate_submission(frame,template,ctx['candidates'])
    fallback=sorted(zip(ctx['candidates'],ctx['fallback']),key=lambda x:(-x[1],x[0]))
    independent=[]
    for pos in range(len(ctx['users'])):
        row=pd.DataFrame({'item':ctx['candidates'],'score':scores[pos],'last':ctx['last_end'][pos]})
        row=row.loc[np.isfinite(row.score)&(row.score>0)].sort_values(['score','last','item'],ascending=[False,False,True])
        chosen=row.item.head(5).tolist()
        for item,_ in fallback:
            if len(chosen)==5:break
            if item not in chosen:chosen.append(int(item))
        independent.append(' '.join(map(str,chosen)))
    assert independent==frame.tv_show_id.tolist()
    assert sha256(root/manifest['csv'])==manifest['sha256']
    assert (ctx['watch'].overlap_end<=ctx['cutoff']).all()
    assert (ctx['historical'].s<ctx['cutoff']).all()
    assert (ctx['target'].e>ctx['cutoff']).all() and (ctx['target'].s<ctx['end']).all()
    assert manifest['late_corrected_audit_inspected'] is False
    for number in [0,1]:
        old=json.loads((root/f'experiments/R{number}/manifest.json').read_text())
        assert sha256(root/old['csv'])==old['sha256']
        assert manifest['sha256']!=old['sha256']
    r1=json.loads((root/'experiments/R1/manifest.json').read_text())
    for name,digest in r1['source_sha256'].items():assert sha256(root/'scripts'/name)==digest
    assert len(ctx['train'])==r1['production_model']['fitting_interactions']
    # Paired user bootstrap describes this selected local difference only.
    predictions=json.loads((out/'development_predictions.json').read_text())
    selected=manifest['selected_strategy'];a=np.asarray(predictions[selected]['per_user_ap']);b=np.asarray(predictions['repeat_none']['per_user_ap'])
    difference=a-b;rng=np.random.default_rng(42)
    means=difference[rng.integers(0,len(difference),size=(2000,len(difference)))].mean(axis=1)
    previous=pd.read_csv(root/r1['csv'],dtype=str)
    uncertainty={'selected':selected,'paired_primary_map_delta':float(difference.mean()),
                 'user_bootstrap_replicates':2000,'seed':42,
                 'percentile_interval_95':[float(x) for x in np.quantile(means,[.025,.975])],
                 'development_users_ap_improves':int((difference>0).sum()),'development_users_ap_worsens':int((difference<0).sum()),
                 'development_users_ap_unchanged':int((difference==0).sum()),
                 'production_users_with_changed_ordered_list_vs_R1':int((previous.tv_show_id!=frame.tv_show_id).sum()),
                 'limitation':'Conditional description of a selected difference; ignores temporal shift and adaptive/multiple development selection. Not an independent significance claim.'}
    dump(out/'paired_uncertainty.json',uncertainty)
    checks.update({'independent_dataframe_score_to_list_parity':True,'fitting_viewing_before_forecast_cutoff':True,
                   'future_schedule_only_availability':True,'R0_R1_csvs_and_R1_source_unchanged':True,
                   'R2_differs_from_previous_csvs':True,'late_corrected_audit_sealed':True,
                   'no_additional_fitting':True,'no_kaggle_upload':True})
    dump(out/'verification.json',checks)
    print(json.dumps({'verification':checks,'paired_uncertainty':uncertainty},indent=2))


if __name__=='__main__':main()
