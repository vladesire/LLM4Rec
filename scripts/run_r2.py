#!/usr/bin/env python3
"""Execute R2 development ablations and freeze one CSV; stop before R3."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from behavioral_r2 import context,features,predictions
from completion_pipeline import epoch,qualifying_interactions,target_sets
from core import ap_at_5
from data_pipeline import canonical_schedule,dump,human,read_schedule,sha256
from run import validate_submission
from run_r1 import describe,immutable_bytes

ROOT=Path(__file__).resolve().parents[1]


def metric(predicted,relevant,users):
    values=[ap_at_5(relevant.get(u,set()),row) for u,row in zip(users,predicted)]
    sizes=np.asarray([len(relevant.get(u,set())) for u in users]);array=np.asarray(values)
    return {'map_at_5':float(array.mean()),'nonempty_map_at_5':float(array[sizes>0].mean()),
            'five_label_map_at_5':float(array[sizes>=5].mean()),'population':len(users),
            'nonempty_users':int((sizes>0).sum()),'five_label_users':int((sizes>=5).sum()),
            'empty_users_included_as_zero':int((sizes==0).sum())},values


def apply_strategy(bundle,strategy):
    return sum(weight*bundle[component] for component,weight in strategy['components'].items())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reproduce',action='store_true');args=parser.parse_args()
    os.chdir(ROOT);start=time.perf_counter();canonical=ROOT/'experiments/R2'
    out=ROOT/'cache/R2_reproduction' if args.reproduce else canonical
    out.mkdir(parents=True,exist_ok=True)
    if not args.reproduce and (out/'manifest.json').exists():
        old=json.loads((out/'manifest.json').read_text());assert sha256(old['csv'])==old['sha256']
        print('R2 already frozen',ROOT/old['csv']);print('STOP: await feedback before R3.');return
    config=json.loads((canonical/'config.json').read_text());assert not config['inspect_corrected_late_audit']
    for round_name in ['R0','R1']:
        previous=json.loads(Path(f'experiments/{round_name}/manifest.json').read_text())
        assert sha256(previous['csv'])==previous['sha256']
    r1=json.loads(Path('experiments/R1/manifest.json').read_text());assert r1['private_score']==.38480
    for name,digest in r1['source_sha256'].items():assert sha256(Path('scripts')/name)==digest,'R1 implementation changed'
    data=pd.read_pickle('cache/completion/prepared.pkl');events=pd.read_pickle('cache/completion/events.pkl')
    dev=pd.read_pickle('cache/completion/r1_development.pkl');users=data['evaluation_users']
    cutoff,end=epoch(config['development_start']),epoch(config['development_end'])
    relevant=dev['relevance'];candidates=dev['candidates']
    ctx=context(data,events,cutoff,end,candidates,data['schedule'],config)
    pd.testing.assert_frame_equal(ctx['train'].sort_values(['u','item']).reset_index(drop=True),
                                  dev['train'][['u','item','count','last_end']].sort_values(['u','item']).reset_index(drop=True),check_dtype=False)
    bundles={};strategies={};results={};records={};per_ap={};score_bank={}
    def add(name,bundle,components,complexity,stage):
        scores=sum(weight*bundle[key] for key,weight in components.items())
        strategy={'half_life_days':bundle['details']['half_life_days'],'components':components,'complexity':complexity,'stage':stage}
        result=predictions(ctx,scores,name);m,aps=metric(result['predictions'],relevant,users)
        record={**m,**result['details'],'half_life_days':strategy['half_life_days'],'components':components,
                'complexity':complexity,'stage':stage,'feature_seconds':bundle['details']['fit_feature_seconds']}
        results[name]=result;records[name]=record;per_ap[name]=aps;strategies[name]=strategy;score_bank[name]=scores
        print(f"R2 DEVELOPMENT {name}: MAP@5={m['map_at_5']:.8f}",flush=True)
    for half_life in config['half_life_days']:
        bundle=features(ctx,half_life);bundles[half_life]=bundle
        tag='none' if half_life is None else str(half_life)
        add('repeat_'+tag,bundle,{'repeat':1.0},0 if half_life is None else 1,'recency')
    def choice_key(name):
        r=records[name];h=r['half_life_days'];simplicity=0 if h is None else 1 if h==56 else 2
        return (-r['map_at_5'],r['complexity'],simplicity,name)
    best_repeat=min(results,key=choice_key);half_life=strategies[best_repeat]['half_life_days'];bundle=bundles[half_life]
    baseline=json.loads(Path('experiments/R1/development_predictions.json').read_text())['personal_frequency']
    assert results['repeat_none']['predictions']==baseline['predictions'],'No-decay repeat no longer matches R1'
    assert per_ap['repeat_none']==baseline['per_user_ap']
    for name,component in [('schedule_rate','opportunity'),('schedule_capped','capped_opportunity'),('channel_habit','channel_habit'),('channel_time_habit','time_habit')]:
        add(name,bundle,{component:1.0},2,'single_addition')
    for weight in config['schedule_blend_weights']:
        add('blend_schedule_'+str(int(weight*100)),bundle,{'repeat':1-weight,'capped_opportunity':weight},3,'global_blend')
    for weight in config['habit_blend_weights']:
        add('blend_habit_'+str(int(weight*100)),bundle,{'repeat':1-weight,'time_habit':weight},3,'global_blend')
    repeat_score=records[best_repeat]['map_at_5']
    schedule_gain=max(records[n]['map_at_5'] for n in ['blend_schedule_25','blend_schedule_50'])>repeat_score
    habit_gain=max(records[n]['map_at_5'] for n in ['blend_habit_10','blend_habit_25'])>repeat_score
    if schedule_gain and habit_gain:
        a,b,c=config['conditional_three_way_weights']
        add('blend_repeat_schedule_habit',bundle,{'repeat':a,'capped_opportunity':b,'time_habit':c},4,'conditional_three_way')
    selected=min(records,key=choice_key)
    # Supplementary labels are the same permitted development window, not late audit.
    aggregate,_=qualifying_interactions(events,cutoff,end,aggregate=True);aggregate_rel,_=target_sets(aggregate)
    reverse_rel,_=target_sets(dev['target'],descending_id=True)
    for name in records:
        records[name]['aggregate_target_map_at_5']=metric(results[name]['predictions'],aggregate_rel,users)[0]['map_at_5']
        records[name]['descending_target_tie_map_at_5']=metric(results[name]['predictions'],reverse_rel,users)[0]['map_at_5']
        records[name]['delta_vs_R1']=records[name]['map_at_5']-records['repeat_none']['map_at_5']
    evidence={'task':'unchanged completion/frequency proxy v1','development':{'fit_end_exclusive':cutoff,'target_end_exclusive':end},
              'population':1260,'candidate_items':len(candidates),'models':records,'selected':selected,'best_repeat':best_repeat,
              'selection_rule':config['selection_rule'],'features':[b['details'] for b in bundles.values()],
              'r1_no_decay_replay_exact':True,'corrected_late_audit_inspected':False,
              'r1_scores_user_reported':{'public':.39334,'private':.38480},
              'adaptive_selection_note':'R1 Kaggle feedback informed this branch. The June development window is reused; sensitivities do not make selection independent.'}
    dump(out/'metrics.json',evidence);pd.DataFrame([{'variant':name,**{k:v for k,v in m.items() if k!='components'},'components':json.dumps(m['components'])} for name,m in records.items()]).to_csv(out/'ablations.csv',index=False)
    dump(out/'development_predictions.json',{name:{'predictions':r['predictions'],'per_user_ap':per_ap[name]} for name,r in results.items()})
    if selected=='repeat_none':
        dump(out/'decision.json',{'status':'NO_SUPPORTED_CHALLENGER','reason':'R1 remains best on the predeclared development rule; no duplicate CSV generated.'})
        print('STOP: no supported R2 challenger; R1 remains best.');return
    strategy=strategies[selected];dump(out/'selected_strategy.json',{'name':selected,**strategy})
    # Production uses only past viewing plus numeric future availability.
    future,_,future_audit=read_schedule('data/export_arh_31-42-final.csv',metadata=False)
    assert sha256('data/export_arh_31-42-final.csv')==data['audit']['input_sha256']['export_arh_31-42-final.csv']
    future,ambiguity=canonical_schedule(future)
    production_cutoff=epoch(config['production_cutoff']);forecast_end=int(future.e.max())
    pctx=context(data,events,production_cutoff,forecast_end,data['test_candidates'],future,config)
    pbundle=features(pctx,strategy['half_life_days']);scores=apply_strategy(pbundle,strategy)
    production=predictions(pctx,scores,selected)
    frame=data['template'].copy();frame['tv_show_id']=[' '.join(map(str,row)) for row in production['predictions']]
    validation=validate_submission(frame,data['template'],data['test_candidates'])
    slug={'repeat_28':'recency28','repeat_56':'recency56','schedule_rate':'schedule_rate','schedule_capped':'schedule_capped',
          'channel_habit':'channel_habit','channel_time_habit':'channel_time_habit','blend_schedule_25':'recency_schedule',
          'blend_schedule_50':'recency_schedule','blend_habit_10':'recency_habit','blend_habit_25':'recency_habit',
          'blend_repeat_schedule_habit':'repeat_schedule_habit'}[selected]
    path=ROOT/f'submissions/R2_{slug}.csv';content=frame.to_csv(index=False,lineterminator='\n').encode()
    import hashlib
    digest=hashlib.sha256(content).hexdigest()
    for other in (ROOT/'submissions').glob('R[01]*.csv'):assert digest!=sha256(other),'Duplicate prior CSV'
    immutable_bytes(path,content);immutable_bytes(path.with_suffix('.csv.sha256'),(digest+'  '+path.name+'\n').encode())
    reread=pd.read_csv(path,dtype=str,keep_default_na=False);validate_submission(reread,data['template'],data['test_candidates'])
    replay=predictions(pctx,scores,'parity')['predictions'];assert replay==production['predictions']
    assert reread.tv_show_id.tolist()==frame.tv_show_id.tolist()
    # Cache scores for independent verification; no candidate variants are uploaded.
    pd.to_pickle({'context':pctx,'scores':scores,'selected':strategy,'bundle_details':pbundle['details']},'cache/completion/r2_production_verification.pkl')
    meta=data['metadata'].loc[data['metadata'].s<cutoff].drop_duplicates('item').set_index('item')
    formal=[]
    for uid in config['formal_example_users']:
        u=data['users'].index(uid);pos=users.index(u);entry={'user_id':uid,'relevant_ids':sorted(relevant.get(u,set())),'variants':{}}
        for name,result in results.items():
            positions={int(item):i for i,item in enumerate(candidates)}
            entry['variants'][name]={'ap_at_5':per_ap[name][pos],'recommendations':[
                {'rank':rank,**describe(meta,item),'score':float(score_bank[name][pos,positions[item]]),
                 'fallback':bool(result['fallback'][pos,rank-1]),'target_relevant':item in relevant.get(u,set())}
                for rank,item in enumerate(result['predictions'][pos],1)]}
        formal.append(entry)
    dump(out/'formal_two_user_examples.json',formal)
    all_meta=data['metadata'].drop_duplicates('item').set_index('item');chosen=[data['users'].index(uid) for uid in config['formal_example_users']]
    n=pctx['train'].groupby('u')['count'].sum().reindex(users,fill_value=0).sort_values()
    for u in [int(n.index[0]),int(n.index[len(n)//2]),int(n.index[-1])]+users:
        if len(chosen)==5:break
        if u not in chosen:chosen.append(u)
    idx={int(item):i for i,item in enumerate(pctx['candidates'])}
    dump(out/'five_production_examples.json',[{'user_id':data['users'][u],'true_future_labels_available':False,
          'recommendations':[{'rank':rank,**describe(all_meta,item),'score':float(scores[users.index(u),idx[item]]),
                              'fallback':bool(production['fallback'][users.index(u),rank-1])}
                             for rank,item in enumerate(production['predictions'][users.index(u)],1)]} for u in chosen])
    tests=subprocess.run([sys.executable,'-m','unittest','tests/test_behavioral_r2.py','-v'],capture_output=True,text=True)
    (out/'focused_test_output.txt').write_text(tests.stdout+tests.stderr);assert tests.returncode==0
    manifest={'round':'R2','status':'AWAITING_USER_SCORE','csv':str(path.relative_to(ROOT)),'sha256':digest,
              'public_score':None,'private_score':None,'submission_id':None,'comments':None,
              'incumbent':'R1','incumbent_private_score':.38480,'incumbent_score_provenance':'exact user report; screenshot not inspected',
              'selected_strategy':selected,'strategy':strategy,'command':'.venv/bin/python scripts/run_r2.py --reproduce',
              'configuration_sha256':sha256(canonical/'config.json'),'contract_sha256':sha256('experiments/EVALUATION_CONTRACT.md'),
              'source_sha256':{n:sha256(Path('scripts')/n) for n in ['run_r2.py','behavioral_r2.py','completion_pipeline.py','round_models.py','core.py']},
              'python':sys.version,'platform':platform.platform(),'seed':config['seed'],
              'packages':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy']},
              'prepared_input_sha256':data['audit']['input_sha256'],
              'elapsed_round_seconds':time.perf_counter()-start,'production_details':production['details'],
              'production_feature_details':pbundle['details'],'future_ambiguity_handling':ambiguity,
              'future_metadata_used':False,'csv_checks':validation,'score_to_list_and_csv_parity':True,
              'late_corrected_audit_inspected':False,'focused_test_exit_code':tests.returncode,
              'hypothesis':'Modest recency/opportunity/habit information improves the completion-aware personal repeat baseline.',
              'next':'STOP. Wait for R2 feedback and instruction before R3.'}
    dump(out/'manifest.json',manifest)
    print('R2 READY',path,'SHA256',digest,'selected',selected,flush=True)
    print('STOP: no R3 training or CSV generated.',flush=True)


if __name__=='__main__':main()
