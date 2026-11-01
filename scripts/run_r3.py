#!/usr/bin/env python3
"""Chronological binary boosted reranking, R3 only."""
from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from completion_pipeline import epoch,qualifying_interactions,target_sets
from data_pipeline import dump,read_schedule,sha256
from rerank_features import make_snapshot,checked_metrics
from run import validate_submission
from run_r1 import immutable_bytes,describe

ROOT=Path(__file__).resolve().parents[1]


def rank_pool(snapshot,scores):
    n,size=snapshot['pool_ids'].shape;values=np.asarray(scores).reshape(n,size)
    assert np.isfinite(values).all(),'Nonfinite reranker score'
    predicted=[];fallback=[]
    cold_feature=snapshot['feature_names'].index('user_cold_qualifying')
    cold=snapshot['X'][:,cold_feature].reshape(n,size)[:,0]>0
    for i,row in enumerate(values):
        order=np.lexsort((snapshot['pool_ids'][i],-snapshot['last_end'][i],-row))
        chosen=[] if cold[i] else snapshot['pool_ids'][i,order[:5]].tolist()
        flags=[False]*len(chosen)
        for ix in np.lexsort((snapshot['candidates'],-snapshot['fallback'])):
            if len(chosen)==5:break
            item=int(snapshot['candidates'][ix])
            if item not in chosen:chosen.append(item);flags.append(True)
        predicted.append(chosen);fallback.append(flags)
    return predicted,np.asarray(fallback,dtype=bool)


def reciprocal_ranks(snapshot,scores):
    values=np.asarray(scores).reshape(snapshot['pool_ids'].shape);output=np.empty_like(values)
    for i,row in enumerate(values):
        order=np.lexsort((snapshot['pool_ids'][i],-snapshot['last_end'][i],-row))
        output[i,order]=1/(60+np.arange(1,len(order)+1))
    return output


def estimator(config,setting,iterations=None,early=False):
    return HistGradientBoostingClassifier(loss='log_loss',learning_rate=config['learning_rate'],
             max_iter=iterations or config['max_iter'],max_leaf_nodes=setting['max_leaf_nodes'],
             min_samples_leaf=config['min_samples_leaf'],l2_regularization=setting['l2_regularization'],
             max_bins=config['max_bins'],categorical_features=None,early_stopping=early,
             scoring='loss',validation_fraction=None,n_iter_no_change=config['n_iter_no_change'],
             random_state=config['seed'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reproduce',action='store_true');args=parser.parse_args()
    os.chdir(ROOT);start_time=time.perf_counter();canonical=ROOT/'experiments/R3'
    out=ROOT/'cache/R3_reproduction' if args.reproduce else canonical;out.mkdir(parents=True,exist_ok=True)
    if not args.reproduce and (out/'manifest.json').exists():
        old=json.loads((out/'manifest.json').read_text());assert sha256(old['csv'])==old['sha256']
        print('R3 already frozen',ROOT/old['csv']);print('STOP before R4.');return
    config=json.loads((canonical/'config.json').read_text());assert not config['inspect_corrected_late_audit']
    r2cfg=json.loads(Path('experiments/R2/config.json').read_text());r1cfg=json.loads(Path('experiments/R1/config.json').read_text())
    r2=json.loads(Path('experiments/R2/manifest.json').read_text());assert r2['private_score']==.39378
    for number in [0,1,2]:
        previous=json.loads(Path(f'experiments/R{number}/manifest.json').read_text());assert sha256(previous['csv'])==previous['sha256']
    for name,digest in r2['source_sha256'].items():assert sha256(Path('scripts')/name)==digest,'R2 source changed'
    data=pd.read_pickle('cache/completion/prepared.pkl');events=pd.read_pickle('cache/completion/events.pkl')
    def snapshot(window,label=True):return make_snapshot(data,events,epoch(window[0]),epoch(window[1]),config,r2cfg,r1cfg,label)
    dev=snapshot(config['development_window'])
    if dev['candidate_diagnostics']['micro_recall']<config['minimum_candidate_micro_recall_for_reranking']:
        dump(out/'decision.json',{'status':'CANDIDATE_RECALL_TOO_LOW','diagnostics':dev['candidate_diagnostics']})
        print('STOP: candidate source diagnosis needed before reranking.');return
    training=[snapshot(window) for window in config['historical_training_windows']]
    inner=snapshot(config['inner_window'])
    assert max(s['end'] for s in training)<=inner['start']
    assert inner['end']<=dev['start']
    for s in training+[inner,dev]:
        assert s['prefix_max_observation_end']<=s['start']
        assert s['feature_names']==dev['feature_names']
    X=np.concatenate([s['X'] for s in training]);y=np.concatenate([s['y'] for s in training])
    inner_records={};inner_models={}
    for setting in config['settings']:
        tick=time.perf_counter();model=estimator(config,setting,early=True)
        with threadpool_limits(limits=1):
            model.fit(X,y,X_val=inner['X'],y_val=inner['y'])
            probability=model.predict_proba(inner['X'])[:,1]
        predicted,_=rank_pool(inner,probability);m,_=checked_metrics(inner['relevant'],inner['users'],predicted)
        inner_records[setting['name']]={**m,'stopped_iterations':int(model.n_iter_),
              'early_stopping_triggered_before_cap':bool(model.n_iter_<config['max_iter']),
              'last_validation_negative_log_loss':float(model.validation_score_[-1]),'elapsed_seconds':time.perf_counter()-tick,
              'parameters':model.get_params()}
        inner_models[setting['name']]=model
        print('R3 INNER',setting['name'],'MAP@5',m['map_at_5'],'iterations',model.n_iter_,flush=True)
    selected_setting=min(config['settings'],key=lambda s:(-inner_records[s['name']]['map_at_5'],s['max_leaf_nodes']))
    iterations=inner_records[selected_setting['name']]['stopped_iterations']
    fit_X=np.concatenate([s['X'] for s in training+[inner]]);fit_y=np.concatenate([s['y'] for s in training+[inner]])
    all_columns=np.arange(len(dev['feature_names']));column_sets={'full':all_columns}
    column_sets['drop_collaborative_content']=np.asarray([i for i,n in enumerate(dev['feature_names']) if dev['feature_families'][n]!='collaborative'])
    column_sets['drop_schedule_habit']=np.asarray([i for i,n in enumerate(dev['feature_names']) if dev['feature_families'][n]!='schedule'])
    metrics={};predictions={};probabilities={};fitted={};aps={};columns_for={};score_bank={'R2_reference':dev['R2_pool_scores']}
    baseline_metric,baseline_ap=checked_metrics(dev['relevant'],dev['users'],dev['R2_predictions'])
    metrics['R2_reference']=baseline_metric;predictions['R2_reference']=dev['R2_predictions'];aps['R2_reference']=baseline_ap
    for name,columns in column_sets.items():
        tick=time.perf_counter();model=estimator(config,selected_setting,iterations=iterations,early=False)
        with threadpool_limits(limits=1):
            model.fit(fit_X[:,columns],fit_y)
            probability=model.predict_proba(dev['X'][:,columns])[:,1]
        predicted,_=rank_pool(dev,probability);m,ap=checked_metrics(dev['relevant'],dev['users'],predicted)
        metrics[name]={**m,'fit_and_predict_seconds':time.perf_counter()-tick,'features':len(columns)}
        predictions[name]=predicted;aps[name]=ap;probabilities[name]=probability;fitted[name]=model;columns_for[name]=columns
        score_bank[name]=probability.reshape(dev['pool_ids'].shape)
        print('R3 DEVELOPMENT',name,'MAP@5',m['map_at_5'],flush=True)
    best_model=min(column_sets,key=lambda n:(-metrics[n]['map_at_5'],len(column_sets[n])))
    for weight in config['simple_fusion_weights']:
        name='blend_model_'+str(int(weight*100))
        scores=weight*reciprocal_ranks(dev,probabilities[best_model])+(1-weight)*reciprocal_ranks(dev,dev['R2_pool_scores'])
        predicted,_=rank_pool(dev,scores);m,ap=checked_metrics(dev['relevant'],dev['users'],predicted)
        metrics[name]={**m,'model_weight':weight,'base_model':best_model};predictions[name]=predicted;aps[name]=ap
        score_bank[name]=scores
        print('R3 DEVELOPMENT',name,'MAP@5',m['map_at_5'],flush=True)
    contenders=[n for n in metrics if n!='R2_reference']
    selected=min(contenders,key=lambda n:(-metrics[n]['map_at_5'],0 if n in column_sets else 1,n))
    # Preserve all model outcomes even if none supports a new file.
    diagnostic={'development':dev['candidate_diagnostics'],'inner':inner['candidate_diagnostics'],
                'training':[s['candidate_diagnostics'] for s in training],
                'source_provider_development_metrics':dev['provider_metrics'],'candidate_labels_injected':False}
    dump(out/'candidate_diagnostics.json',diagnostic)
    dump(out/'metrics.json',{'inner_settings':inner_records,'selected_setting':selected_setting,'stopped_iterations':iterations,
         'development_models':metrics,'selected':selected,'feature_names':dev['feature_names'],'feature_families':dev['feature_families'],
         'history_length_analogue':{'training_history_weeks':[s['history_weeks'] for s in training],
              'inner_history_weeks':inner['history_weeks'],'development_history_weeks':dev['history_weeks'],
              'inner_R2_map':inner['provider_metrics']['R2']['map_at_5'],
              'inner_best_model_map':inner_records[selected_setting['name']]['map_at_5'],
              'counts_normalized_to_rates_and_shares':True,'remaining_issue':'Production has longer history/horizon; this is an analogue, not proof of extrapolation.'},
         'chronology':{'latest_early_training_label_end':max(s['end'] for s in training),'inner_start':inner['start'],
              'latest_development_training_label_end':inner['end'],'development_start':dev['start'],
              'overlapping_earlier_target_windows':True},'corrected_late_audit_inspected':False,
         'selection_note':'Inner selects stopping/configuration. Reused June development selects family/mixture. No independent final-test claim.'})
    dump(out/'development_predictions.json',{n:{'predictions':p,'per_user_ap':aps[n]} for n,p in predictions.items()})
    pd.to_pickle({'snapshot':dev,'scores':score_bank,'selected':selected},'cache/r3/development_verification.pkl')
    pd.DataFrame([{'variant':n,**m} for n,m in metrics.items()]).to_csv(out/'comparison.csv',index=False)
    if metrics[selected]['map_at_5']<=metrics['R2_reference']['map_at_5']:
        dump(out/'decision.json',{'status':'RERANKER_NOT_SUPPORTED','selected_challenger':selected,'reference':'R2','reason':'No tested reranker or blend beats R2 locally. No duplicate or known inferior CSV generated.'})
        print('STOP: reranker/fusion does not improve R2; inspect a simpler diagnostic challenger.');return
    selected_base=selected if selected in column_sets else metrics[selected]['base_model']
    weight=None if selected in column_sets else metrics[selected]['model_weight']
    columns=column_sets[selected_base]
    # Explicit production refit adds the completed June development example.
    production_X=np.concatenate([s['X'] for s in training+[inner,dev]])[:,columns]
    production_y=np.concatenate([s['y'] for s in training+[inner,dev]])
    assert max(s['end'] for s in training+[inner,dev])<=epoch('2020-06-29')<epoch(config['production_cutoff'])
    tick=time.perf_counter();model=estimator(config,selected_setting,iterations=iterations,early=False)
    with threadpool_limits(limits=1):model.fit(production_X,production_y)
    future,_,_=read_schedule('data/export_arh_31-42-final.csv',metadata=False)
    production=make_snapshot(data,events,epoch(config['production_cutoff']),int(future.e.max()),config,r2cfg,r1cfg,label=False)
    with threadpool_limits(limits=1):probability=model.predict_proba(production['X'][:,columns])[:,1]
    score=probability.reshape(production['pool_ids'].shape)
    if weight is not None:score=weight*reciprocal_ranks(production,score)+(1-weight)*reciprocal_ranks(production,production['R2_pool_scores'])
    prod_predictions,fallback=rank_pool(production,score)
    frame=data['template'].copy();frame['tv_show_id']=[' '.join(map(str,row)) for row in prod_predictions]
    validation=validate_submission(frame,data['template'],data['test_candidates'])
    slug='boosted_reranker' if weight is None else 'reranker_blend';path=ROOT/f'submissions/R3_{slug}.csv'
    content=frame.to_csv(index=False,lineterminator='\n').encode()
    import hashlib
    digest=hashlib.sha256(content).hexdigest()
    for previous in (ROOT/'submissions').glob('R[012]*.csv'):assert sha256(previous)!=digest,'Duplicate prior CSV'
    immutable_bytes(path,content);immutable_bytes(path.with_suffix('.csv.sha256'),(digest+'  '+path.name+'\n').encode())
    reread=pd.read_csv(path,dtype=str,keep_default_na=False);validate_submission(reread,data['template'],data['test_candidates'])
    assert frame.tv_show_id.tolist()==reread.tv_show_id.tolist()
    pd.to_pickle({'snapshot':production,'scores':score,'model':model,'columns':columns,'selected':selected,'selected_base':selected_base,
                  'model_weight':weight,'probabilities':probability},'cache/r3/production_verification.pkl')
    meta=data['metadata'].loc[data['metadata'].s<dev['start']].drop_duplicates('item').set_index('item')
    formal=[]
    for uid in config['formal_example_users']:
        u=data['users'].index(uid);i=dev['users'].index(u)
        pool_lookup={int(item):j for j,item in enumerate(dev['pool_ids'][i])}
        full_lookup={int(item):j for j,item in enumerate(dev['candidates'])}
        formal.append({'user_id':uid,'relevant_ids':sorted(dev['relevant'].get(u,set())),
           'methods':{n:{'ap_at_5':aps[n][i],
               'score_definition':'R2 forecast count' if n=='R2_reference' else 'weighted reciprocal rank' if n.startswith('blend_') else 'binary preference probability',
               'recommendations':[{'rank':rank,**describe(meta,item),'relevant':item in dev['relevant'].get(u,set()),
                                    'method_score':float(score_bank[n][i,pool_lookup[item]]),
                                    'popularity_backfill':bool(dev['X'][i*config['candidate_size'],dev['feature_names'].index('user_cold_qualifying')]>0),
                                    'fallback_popularity':float(dev['fallback'][full_lookup[item]])}
                                   for rank,item in enumerate(p[i],1)]} for n,p in predictions.items()}})
    dump(out/'formal_two_user_examples.json',formal)
    all_meta=data['metadata'].drop_duplicates('item').set_index('item')
    active_col=production['feature_names'].index('user_qualifying_count_per_calendar_week')
    active=production['X'][:,active_col].reshape(production['pool_ids'].shape)[:,0]
    ordered=np.argsort(active,kind='stable');chosen=[data['users'].index(uid) for uid in config['formal_example_users']]
    for u in [production['users'][int(ordered[0])],production['users'][int(ordered[len(ordered)//2])],production['users'][int(ordered[-1])]]+production['users']:
        if len(chosen)==5:break
        if u not in chosen:chosen.append(u)
    repeat_col=production['feature_names'].index('has_qualifying_personal_history')
    personal=production['X'][:,repeat_col].reshape(production['pool_ids'].shape)
    repeat_hits=0;examples=[]
    for i,u in enumerate(production['users']):
        lookup={int(item):j for j,item in enumerate(production['pool_ids'][i])}
        assert all(item in lookup for item in prod_predictions[i])
        repeat_hits+=sum(personal[i,lookup[item]]>0 for item in prod_predictions[i])
        if u in chosen:
            examples.append({'user_id':data['users'][u],'true_future_labels_available':False,
              'recommendations':[{'rank':rank,**describe(all_meta,item),'method_score':float(score[i,lookup[item]]),
                                   'popularity_backfill':bool(fallback[i,rank-1]),
                                   'previously_qualified':bool(personal[i,lookup[item]]>0)} for rank,item in enumerate(prod_predictions[i],1)]})
    dump(out/'five_production_examples.json',examples)
    tests=subprocess.run([sys.executable,'-m','unittest','tests/test_rerank_r3.py','-v'],capture_output=True,text=True)
    (out/'focused_test_output.txt').write_text(tests.stdout+tests.stderr);assert tests.returncode==0
    manifest={'round':'R3','status':'AWAITING_USER_SCORE','csv':str(path.relative_to(ROOT)),'sha256':digest,
          'public_score':None,'private_score':None,'submission_id':None,'comments':None,'incumbent':'R2','incumbent_private_score':.39378,
          'selected':selected,'selected_base':selected_base,'model_weight':weight,'model':'sklearn HistGradientBoostingClassifier',
          'model_parameters':model.get_params(),'feature_columns':columns.tolist(),'feature_names':production['feature_names'],
          'command':'.venv/bin/python scripts/run_r3.py --reproduce','configuration_sha256':sha256(canonical/'config.json'),
          'contract_sha256':sha256('experiments/EVALUATION_CONTRACT.md'),
          'source_sha256':{n:sha256(Path('scripts')/n) for n in ['run_r3.py','rerank_features.py','behavioral_r2.py','round_models.py','completion_pipeline.py','core.py']},
          'python':sys.version,'platform':platform.platform(),'packages':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','scipy','implicit']},
          'installed_fit_signature':str(inspect.signature(HistGradientBoostingClassifier.fit)),
          'elapsed_round_seconds':time.perf_counter()-start_time,'production_fit_and_predict_seconds':time.perf_counter()-tick,
          'candidate_diagnostics':dev['candidate_diagnostics'],'csv_checks':validation,
          'production_distinct_lists':len({tuple(r) for r in prod_predictions}),'production_distinct_items':len({i for row in prod_predictions for i in row}),
          'production_fallback_users':int(fallback.any(axis=1).sum()),'production_fallback_slots':int(fallback.sum()),
          'production_qualifying_history_repeat_share':repeat_hits/(5*len(production['users'])),
          'production_labels_latest_end':max(s['end'] for s in training+[inner,dev]),
          'production_feature_cutoff':production['start'],'late_corrected_audit_inspected':False,'future_positives_injected':False,
          'hypothesis':'Past-only boosted combination of personal, habit, collaborative and reliability signals transfers better than the R2 global blend.',
          'next':'STOP for R3 scores/comments before R4.'}
    dump(out/'manifest.json',manifest)
    print('R3 READY',path,'SHA256',digest,'selected',selected,flush=True)
    print('STOP before R4.',flush=True)


if __name__=='__main__':main()
