#!/usr/bin/env python3
"""Run R1 only. Never upload, inspect the late corrected audit, or start R2."""
from __future__ import annotations

import importlib.metadata
import argparse
import io
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from completion_pipeline import epoch, legacy_broad_prefix, prepare_completion, qualifying_interactions, target_sets
from core import ap_at_5
from data_pipeline import FILES, dump, human, sha256
from round_models import ranked_lists, score_model
from run import validate_submission

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/R1'


def immutable_bytes(path, content):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        assert path.read_bytes()==content,f'Immutable artifact differs: {path}'
    else:
        with path.open('xb') as f:f.write(content)
        path.chmod(0o444)


def preserve_r0():
    original=ROOT/'submission/final_submission.csv'
    expected='73d71295c19c61c3702a3c8e0fae951f27a307ce9b05cd39fd37e9b93e0996e0'
    assert sha256(original)==expected
    source_commit=subprocess.check_output(['git','rev-parse','6a26a69'],text=True).strip()
    for name in ['config.json','results/metrics.json','results/environment.json']:
        recorded=subprocess.check_output(['git','show',f'{source_commit}:{name}'])
        assert (ROOT/name).read_bytes()==recorded,f'R0 artifact changed: {name}'
    target=ROOT/'submissions/R0_global_popularity.csv'
    immutable_bytes(target,original.read_bytes())
    for source,name in [('config.json','config.json'),('results/metrics.json','metrics_historical_broad_proxy.json')]:
        immutable_bytes(ROOT/'experiments/R0'/name,(ROOT/source).read_bytes())
    manifest={'round':'R0','csv':'submissions/R0_global_popularity.csv','sha256':expected,'code_commit':source_commit,
              'public_score':None,'private_score_approximate':0.27,'private_score_exact':None,
              'score_provenance':'Student report; no original score evidence inspected',
              'historical_local_target':'All programmes with a >=300-second user-airing label; old June29–July27 split',
              'historical_proxy_map_at_5':0.5957169312169311,
              'warning':'Historical broad-label proxy result; not comparable numerically with corrected-target metrics.'}
    saved=ROOT/'experiments/R0/manifest.json'
    if saved.exists():
        existing=json.loads(saved.read_text());assert existing['sha256']==expected
        return existing
    dump(saved,manifest)
    return manifest


def evaluate(result, relevant, users):
    values=np.asarray([ap_at_5(relevant.get(u,set()),row) for u,row in zip(users,result['predictions'])])
    sizes=np.asarray([len(relevant.get(u,set())) for u in users])
    return {'map_at_5':float(values.mean()),'population':len(users),'empty_relevance_users':int((sizes==0).sum()),
            'nonempty_users':int((sizes>0).sum()),'at_least_five_label_users':int((sizes>=5).sum()),
            'map_at_5_nonempty':float(values[sizes>0].mean()) if (sizes>0).any() else None,
            'map_at_5_at_least_five_labels':float(values[sizes>=5].mean()) if (sizes>=5).any() else None,
            'fit_seconds':result['fit_seconds'],'inference_seconds':result['inference_seconds'],
            **result['details']},values.tolist()


def describe(meta,item):
    if item not in meta.index:return {'item':int(item),'title':'[No permitted historical metadata]','category':'[Unknown]'}
    row=meta.loc[item]
    return {'item':int(item),'title':str(row.tv_show_title),'category':str(row.tv_show_category)}


def raw_worked_examples(data,dev_start,dev_end):
    """Read selected original CSV records again, not just cached example tables."""
    ev=data['events'];safe=ev.loc[ev.overlap_end<=dev_end]
    sessions=data['sessions'].set_index('session');schedule=data['schedule'].set_index('airing')
    qualified=safe.loc[(safe.completion_fraction>=.8)&safe.session.isin(sessions.loc[sessions.raw_fragment_count==1].index)]
    partial=safe.loc[(safe.completion_fraction>0)&(safe.completion_fraction<.8)]
    spans=safe.groupby('session').airing.nunique();multi=safe.loc[safe.session.isin(spans.loc[spans>1].index)]
    crossing=safe.loc[(safe.overlap_start<dev_start)&(safe.overlap_end>dev_start)]
    chosen=[]
    for label,group in [('qualifying single fragment',qualified),('partial overlap',partial),('session spans broadcasts',multi),('development-cutoff crossing',crossing)]:
        if len(group):chosen.append((label,group.iloc[0]))
    raw_ids=set();source_records={}
    for _,event in chosen:
        members=data['raw'].loc[data['raw'].session==event.session]
        raw_ids.update(members.raw_id.tolist())
        sc=schedule.loc[int(event.airing)]
        key=int(sc.source_file_index);source_records.setdefault(key,set()).add(int(sc.source_record))
    original_views={};offset=0
    for chunk in pd.read_csv('data/dataset11-30.csv',dtype=str,keep_default_na=False,chunksize=300000):
        for rid in raw_ids:
            if offset<=rid<offset+len(chunk):original_views[int(rid)]=chunk.iloc[int(rid-offset)].to_dict()
        offset+=len(chunk)
        if len(original_views)==len(raw_ids):break
    original_airings={}
    for key,indices in source_records.items():
        offset=0
        for chunk in pd.read_csv(Path('data')/FILES[key+1],dtype=str,keep_default_na=False,chunksize=100000):
            for rid in indices:
                if offset<=rid<offset+len(chunk):original_airings[(key,rid)]=chunk.iloc[rid-offset].to_dict()
            offset+=len(chunk)
            if all((key,rid) in original_airings for rid in indices):break
    output=[]
    for kind,event in chosen:
        sv=sessions.loc[int(event.session)];sc=schedule.loc[int(event.airing)]
        orig=original_airings[(int(sc.source_file_index),int(sc.source_record))]
        assert int(orig['tv_show_id'])==int(event['item']) and int(orig['channel_id'])==int(event.ch)
        assert epoch(pd.to_datetime(orig['start_time'],format='%d.%m.%Y %H:%M:%S'))==int(sc.s)
        assert int(orig['duration'])==int(event.original_duration)==int(sc.e-sc.s)
        members=data['raw'].loc[data['raw'].session==event.session]
        originals=[{'zero_based_record':int(row.raw_id),'record':original_views[int(row.raw_id)]} for row in members.itertuples()]
        for raw in originals:
            record=raw['record'];assert record['user_id']==data['users'][int(sv.u)] and int(record['vsetv_id'])==int(sv.ch)
        overlap=max(0,min(int(sv.e),int(sc.e))-max(int(sv.s),int(sc.s)))
        assert overlap==int(event.overlap_seconds)
        split_fit=max(0,min(int(sv.e),int(sc.e),dev_start)-max(int(sv.s),int(sc.s)))
        split_target=max(0,min(int(sv.e),int(sc.e),dev_end)-max(int(sv.s),int(sc.s),dev_start))
        entry={'case':kind,'user_id':data['users'][int(sv.u)],'channel':int(sv.ch),'item':int(event['item']),
               'session_id':int(event.session),'airing_id':int(event.airing),
               'session_start':human(sv.s),'session_end':human(sv.e),
               'original_broadcast_duration_seconds':int(event.original_duration),'overlap_seconds':overlap,
               'completion_fraction':overlap/int(event.original_duration),
               'fitting_side_seconds_at_june1':split_fit,'target_side_seconds_at_june1':split_target,
               'fitting_completion_full_denominator':split_fit/int(event.original_duration),
               'target_completion_full_denominator':split_target/int(event.original_duration),
               'original_view_csv_records':originals,'original_schedule_file':FILES[int(sc.source_file_index)+1],
               'original_schedule_zero_based_record':int(sc.source_record),'original_schedule_record':orig}
        if kind=='session spans broadcasts':
            entry['all_intersections_for_session']=[{'airing':int(r.airing),'item':int(r.item),'seconds':int(r.overlap_seconds)}
                                                    for r in safe.loc[safe.session==event.session].itertuples()]
        output.append(entry)
    dump(OUT/'original_row_worked_examples.json',output)
    lines=['# Original-row completion checks','','The agent re-read these records from the original CSVs. Record numbers are zero-based CSV records, not physical line numbers. Student personal verification remains pending.','']
    for row in output:
        lines += [f"## {row['case']}",f"User {row['user_id']}; channel {row['channel']}; programme {row['item']}; session {row['session_id']}; airing {row['airing_id']}.",
                  f"Session [{row['session_start']},{row['session_end']}); schedule record {row['original_schedule_file']} #{row['original_schedule_zero_based_record']}.",
                  f"Overlap {row['overlap_seconds']} / original duration {row['original_broadcast_duration_seconds']} = {row['completion_fraction']:.6f}.",
                  f"At 1 June: fitting {row['fitting_side_seconds_at_june1']} s, target {row['target_side_seconds_at_june1']} s; full denominator is retained on both sides.",
                  'Original viewing records: '+json.dumps(row['original_view_csv_records'],ensure_ascii=False),
                  'Original schedule record: '+json.dumps(row['original_schedule_record'],ensure_ascii=False),'']
    (OUT/'worked_completion_checks.md').write_text('\n\n'.join(lines))
    return len(output)


def save_examples(data,train,target,rel,results,metrics,config,prod,prod_train):
    users=data['evaluation_users'];lookup={uid:i for i,uid in enumerate(data['users'])}
    meta=data['metadata'].loc[data['metadata'].s<epoch(config['development_start'])].drop_duplicates('item').set_index('item')
    candidate_positions={item:i for i,item in enumerate(results['personal_frequency']['candidates'])}
    formal=[];histories=[]
    for uid in config['formal_example_users']:
        u=lookup[uid];pos=users.index(u);entry={'user_id':uid,'methods':{},'target_ids':sorted(rel.get(u,set()))}
        for phase,frame in [('development_fitting',train),('development_target',target)]:
            for r in frame.loc[frame.u==u].sort_values(['count','item'],ascending=[False,True]).itertuples(index=False):
                histories.append({'user_id':uid,'phase':phase,**describe(meta,int(r.item)),
                                  'qualifying_broadcast_count':int(r.count),'qualified_seconds':int(r.qualified_seconds),'last_end':human(r.last_end)})
        for name,r in results.items():
            rows=[]
            for rank,item in enumerate(r['predictions'][pos],1):
                ix=candidate_positions[item];value=float(r['scores'][pos,ix])
                rows.append({'rank':rank,**describe(meta,item),'method_score':value if np.isfinite(value) else None,
                             'fallback':bool(r['fallback'][pos,rank-1]),'completion_popularity':float(r['popularity'][ix]),
                             'target_relevant':item in rel.get(u,set())})
            entry['methods'][name]={'ap_at_5':metrics[name]['per_user_ap'][pos],'recommendations':rows}
        formal.append(entry)
    dump(OUT/'formal_two_user_examples.json',formal)
    pd.DataFrame(histories).to_csv(OUT/'formal_two_user_histories.csv',index=False)
    # Five production examples have no future relevance annotation.
    n=prod_train.groupby('u')['count'].sum().reindex(users,fill_value=0).sort_values()
    chosen=[lookup[uid] for uid in config['formal_example_users']]
    for u in [int(n.index[0]),int(n.index[len(n)//2]),int(n.index[-1])]+users:
        if len(chosen)==5:break
        if u not in chosen:chosen.append(u)
    meta_prod=data['metadata'].drop_duplicates('item').set_index('item')
    known={(int(r.u),int(r.item)):(int(r.count),int(r.last_end)) for r in prod_train.itertuples(index=False) if r.u in chosen}
    examples=[]
    for u in chosen:
        pos=users.index(u);rows=[]
        for rank,item in enumerate(prod['predictions'][pos],1):
            count,last=known.get((u,item),(0,None))
            rows.append({'rank':rank,**describe(meta_prod,item),'qualifying_historical_broadcast_count':count,
                         'last_qualifying_end':human(last) if last else None,'fallback':bool(prod['fallback'][pos,rank-1])})
        examples.append({'user_id':data['users'][u],'recommendations':rows,'true_future_preferences_available':False})
    dump(OUT/'five_production_examples.json',examples)
    # One real corrected-development AP calculation, with the full target set.
    values=metrics['personal_frequency']['per_user_ap']
    pos=next((i for i,v in enumerate(values) if 0<v<1),next((i for i,v in enumerate(values) if v>0),0))
    u=users[pos];hits=0;calc=[];total=0.0
    for rank,item in enumerate(results['personal_frequency']['predictions'][pos],1):
        hit=item in rel.get(u,set());hits+=int(hit);amount=hits/rank if hit else 0;total+=amount
        calc.append({'rank':rank,'item':item,'hit':hit,'precision':hits/rank,'contribution':amount})
    dump(OUT/'real_user_ap.json',{'user_id':data['users'][u],'relevant_ids':sorted(rel.get(u,set())),
                                 'denominator':min(5,len(rel.get(u,set()))),'rank_calculation':calc,
                                 'numerator':total,'ap_at_5':values[pos],'student_personally_verified':False})


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--reproduce',action='store_true',help='Recompute R1 evidence under cache/R1_reproduction; keep the original round artifacts unchanged')
    args=parser.parse_args()
    os.chdir(ROOT);start=time.perf_counter()
    canonical=ROOT/'experiments/R1'
    if args.reproduce:
        OUT=ROOT/'cache/R1_reproduction'
        immutable_bytes(OUT/'config.json',(canonical/'config.json').read_bytes())
    elif (OUT/'manifest.json').exists():
        existing=json.loads((OUT/'manifest.json').read_text())
        assert sha256(ROOT/existing['csv'])==existing['sha256']
        print('R1 already frozen:',ROOT/existing['csv'],existing['sha256'])
        print('STOP. Use --reproduce for an explicit local reproduction; wait for user feedback before another round.')
        return
    OUT.mkdir(parents=True,exist_ok=True)
    config=json.loads((OUT/'config.json').read_text())
    assert config['round']=='R1' and not config['inspect_corrected_late_audit']
    r0=preserve_r0();data=prepare_completion()
    dev_start,dev_end,prod_cutoff=[epoch(config[k]) for k in ['development_start','development_end','production_cutoff']]
    assert dev_end==epoch(config['reserved_audit_start']) and prod_cutoff==epoch(config['reserved_audit_end'])
    lower=int(data['events'].overlap_start.min())
    train,fit_audit=qualifying_interactions(data['events'],lower,dev_start)
    target,target_audit=qualifying_interactions(data['events'],dev_start,dev_end)
    rel,target_top=target_sets(target)
    assert fit_audit['max_observation_end']<=dev_start and target_audit['max_observation_end']<=dev_end
    schedule=data['availability'];candidates=sorted(schedule.loc[(schedule.s<dev_end)&(schedule.e>dev_start)&(schedule.item>0),'item'].unique().tolist())
    users=data['evaluation_users'];candidate_set=set(candidates)
    unreachable=sum(len(rel.get(u,set())-candidate_set) for u in users)
    # Retain rather than filter any unreachable IDs in rel and the metric denominator.
    legacy,legacy_audit=legacy_broad_prefix(data['raw'],data['schedule'],dev_start)
    results,metrics={},{}
    for name in ['r0_strategy','completion_popularity','content','als','personal_frequency']:
        r=score_model(name,train,data['metadata'],dev_start,len(data['users']),users,candidates,config,legacy)
        metric,ap=evaluate(r,rel,users);metric['per_user_ap']=ap
        r['candidates']=candidates;results[name]=r;metrics[name]=metric
        print(f"R1 DEVELOPMENT {name}: MAP@5={metric['map_at_5']:.8f}; any/full fallback={metric['fallback_users_any']}/{metric['fallback_users_full']}",flush=True)
    aggregate_target,aggregate_audit=qualifying_interactions(data['events'],dev_start,dev_end,aggregate=True)
    aggregate_rel,_=target_sets(aggregate_target);reverse_rel,_=target_sets(target,descending_id=True)
    sensitivities={}
    for label,alternative in [('aggregate_across_sittings_target',aggregate_rel),('descending_id_target_ties',reverse_rel)]:
        sensitivities[label]={name:evaluate(result,alternative,users)[0] for name,result in results.items()}
    target_top.loc[target_top.u.isin(users)].assign(user_id=lambda d:d.u.map(dict(enumerate(data['users'])))).to_csv(OUT/'development_targets.csv',index=False)
    pd.to_pickle({'train':train,'target':target,'relevance':rel,'legacy_broad':legacy,'candidates':candidates},'cache/completion/r1_development.pkl')
    compact={name:{k:v for k,v in metric.items() if k!='per_user_ap'} for name,metric in metrics.items()}
    dump(OUT/'metrics.json',{'task':'completion/frequency proxy v1','split':{'fit_end_exclusive':config['development_start'],'target_end_exclusive':config['development_end']},
                           'population':'all official 1260 template users; empty relevance AP=0', 'models':compact,
                           'unreachable_target_user_item_pairs_retained':unreachable,'candidate_items':len(candidates),
                           'fitting_labels':fit_audit,'development_target_labels':target_audit,
                           'sensitivity_results':sensitivities,'aggregate_target_audit':aggregate_audit,
                           'corrected_late_audit_inspected':False,
                           'r0_original_map_warning':'0.595717 used a different historical broad-label task and is not a comparator here.'})
    dump(OUT/'development_predictions.json',{name:{'predictions':r['predictions'],'per_user_ap':metrics[name]['per_user_ap']} for name,r in results.items()})
    pd.DataFrame([{'method':name,'MAP@5':m['map_at_5'],'nonempty_MAP@5':m['map_at_5_nonempty'],
                   'five_label_MAP@5':m['map_at_5_at_least_five_labels'],'fit_seconds':m['fit_seconds'],'inference_seconds':m['inference_seconds'],
                   'any_fallback_users':m['fallback_users_any'],'full_fallback_users':m['fallback_users_full'],
                   'repeat_share':m['qualifying_history_repeat_share'],'distinct_lists':m['distinct_recommendation_lists']} for name,m in compact.items()]).to_csv(OUT/'comparison.csv',index=False)
    # Production refit may use all history; no corrected late-audit targets/metrics are constructed.
    production_train,production_audit=qualifying_interactions(data['events'],lower,prod_cutoff)
    production=score_model('personal_frequency',production_train,data['metadata'],prod_cutoff,len(data['users']),users,data['test_candidates'],config)
    frame=data['template'].copy();frame['tv_show_id']=[' '.join(map(str,row)) for row in production['predictions']]
    validation=validate_submission(frame,data['template'],data['test_candidates'])
    content=frame.to_csv(index=False,lineterminator='\n').encode()
    path=ROOT/'submissions/R1_personal_completion.csv'
    # Repeat ranking independently, and check string/file decoding parity.
    replay,flags=ranked_lists(production['scores'],data['test_candidates'],production['popularity'],production['secondary'])
    assert replay==production['predictions'] and np.array_equal(flags,production['fallback'])
    import hashlib
    digest=hashlib.sha256(content).hexdigest();assert digest!=r0['sha256'],'R1 would duplicate R0'
    immutable_bytes(path,content)
    reread=pd.read_csv(path,dtype=str,keep_default_na=False)
    validate_submission(reread,data['template'],data['test_candidates'])
    assert reread.tv_show_id.tolist()==frame.tv_show_id.tolist()
    immutable_bytes(path.with_suffix('.csv.sha256'),(digest+'  '+path.name+'\n').encode())
    save_examples(data,train,target,rel,results,metrics,config,production,production_train)
    checked_examples=raw_worked_examples(data,dev_start,dev_end)
    tests=subprocess.run([sys.executable,'-m','unittest','tests/test_completion.py','tests.test_core.FocusedChecks.test_ap','-v'],capture_output=True,text=True)
    (OUT/'focused_test_output.txt').write_text(tests.stdout+tests.stderr);assert tests.returncode==0,tests.stderr
    dump(OUT/'data_audit.json',{'preparation':data['audit'],'fitting':fit_audit,'development_target':target_audit,
                              'production_refit':production_audit,'r0_strategy_prefix':legacy_audit,
                              'cache_used':data['preparation_cache_used'],'completion_rule':'continuous-session >=80% original full broadcast duration',
                              'late_corrected_audit_target_never_constructed':True})
    source_hashes={p.name:sha256(p) for p in [Path(__file__),Path('scripts/completion_pipeline.py'),Path('scripts/round_models.py'),Path('scripts/core.py'),Path('scripts/models.py'),Path('scripts/data_pipeline.py')]}
    manifest={'round':'R1','status':'AWAITING_USER_SCORE','csv':'submissions/R1_personal_completion.csv','sha256':digest,
              'public_score':None,'private_score':None,'submission_id':None,'comments':None,
              'r0_private_score_approximate':0.27,'incumbent':'R0 (only scored submission so far; approximate user-reported score)',
              'command':'.venv/bin/python scripts/run_r1.py --reproduce','configuration_sha256':sha256(OUT/'config.json'),
              'contract_sha256':sha256('experiments/EVALUATION_CONTRACT.md'),'source_sha256':source_hashes,
              'seed':config['seed'],'python':sys.version,'platform':platform.platform(),
              'packages':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','implicit']},
              'elapsed_round_seconds':time.perf_counter()-start,'production_model':production['details'],
              'csv_checks':validation,'score_to_list_and_csv_parity_passed':True,'different_from_R0':True,
              'original_row_examples_checked':checked_examples,'focused_test_process_exit':tests.returncode,
              'late_corrected_audit_inspected':False,'quota':'Not accessible locally; student must confirm four additional attempts remain before R1 upload.',
              'hypothesis':'Completion-aware personal repetition transfers better than an identical global-reach list; this package does not isolate the two changes.',
              'main_alternatives':'Preference-proxy error, session/device ambiguity, target ties, sparse usable history, availability and temporal transfer.',
              'next_round':'STOP. R2 branch is conditional on returned R1 scores and student instruction.'}
    dump(OUT/'manifest.json',manifest)
    print('R1 READY',path,'SHA256',digest,flush=True)
    print('STOP: awaiting student score/comments. No R2 models or CSV generated.',flush=True)


if __name__=='__main__':main()
