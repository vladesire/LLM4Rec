"""Small count-distribution R4 experiment. No uploads; late audit opened once."""
import json,time,sys,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from count_r4 import posterior,sample_decisions
from completion_pipeline import epoch,qualifying_interactions,target_sets
from data_pipeline import dump,read_schedule,canonical_schedule,sha256
from rerank_features import checked_metrics,make_snapshot,candidate_diagnostics
from run_r1 import immutable_bytes
from run import validate_submission
from run_r3 import rank_pool,reciprocal_ranks
from round_models import count_matrix

OUT=Path('cache/R4_reproduction') if '--reproduce' in sys.argv else Path('experiments/R4');OUT.mkdir(parents=True,exist_ok=True)
start=time.perf_counter()
if (OUT/'manifest.json').exists():
 m=json.loads((OUT/'manifest.json').read_text());assert sha256(m['csv'])==m['sha256'];print('R4 frozen; STOP for feedback');sys.exit()
cfg={'kappa_grid':[5,20,100],'draws':512,'seed':42,'model_pool_top_mean':500,'tail_expansion':1000,
     'recommendation_pool':200,'blend_weights':[.1,.25,.5],
     'selection':'Choose kappa on inner membership MAP. Freeze development winner of B/C/D and its three R3 blends. Later audit compares that challenger with R3 once; ship only if audit delta nonnegative and development delta positive.',
     'prior':'user-channel qualifying rate shrunk by 20 available broadcasts toward population channel rate; target-channel mixture; clipped to [1e-6,1-1e-6]',
     'observation_window':'each user first historical view through cutoff; count all intersected labelled broadcast opportunities; no decay',
     'sampling':'independent programme Beta/Binomial; separate shared user seed streams, items-first array; B/C/D use identical draws',
     'late_audit_previously_exposed_corrected':False}
dump(OUT/'config.json',cfg)
data=pd.read_pickle('cache/completion/prepared.pkl');events=pd.read_pickle('cache/completion/events.pkl')
r2cfg=json.loads(Path('experiments/R2/config.json').read_text());r1cfg=json.loads(Path('experiments/R1/config.json').read_text());r3cfg=json.loads(Path('experiments/R3/config.json').read_text())
original=json.loads(Path('experiments/R3/manifest.json').read_text());assert sha256(original['csv'])==original['sha256']
devcache=pd.read_pickle('cache/r3/development_verification.pkl');dev=devcache['snapshot'];dev['R3_pool_scores']=devcache['scores'][original['selected']]
prod_cache=pd.read_pickle('cache/r3/production_verification.pkl')
def observed(s):
 target,_=qualifying_interactions(events,s['start'],s['end']);s['observed_counts']=count_matrix(target,len(data['users']),s['candidates'])[s['users']].toarray().astype(int)
 return s
def apply_policy(s,model,res):
 fallback=s['candidates'][np.lexsort((s['candidates'],-s['fallback']))].tolist()
 for label in ['B_mean','C_membership','D_expected_AP']:
  for u,row in enumerate(res[label]):
   idx={int(v):j for j,v in enumerate(s['pool_ids'][u])};full={int(v):j for j,v in enumerate(s['candidates'])}
   chosen=[]
   if not model['cold'][u]:
    for i in row:
     score=model['mean'][u,full[i]] if label=='B_mean' else res['membership_scores'][u,idx[i]]
     if score>0:chosen.append(i)
   for i in fallback:
    if len(chosen)==5:break
    if i not in chosen:chosen.append(i)
   res[label][u]=chosen
 return res
def execute(s,k,limit=None,draws=512,stability=False):
 model=posterior(data,events,s,r2cfg,k)
 with threadpool_limits(limits=1):res=sample_decisions(model,s,draws,42,users_limit=limit,stability=stability)
 if limit is None:res=apply_policy(s,model,res)
 return model,res
inner=observed(make_snapshot(data,events,epoch('2020-05-04'),epoch('2020-06-01'),r3cfg,r2cfg,r1cfg))
inner_results={}
for k in cfg['kappa_grid']:
 model,res=execute(inner,k)
 result={name:checked_metrics(inner['relevant'],inner['users'],res[name])[0] for name in ['B_mean','C_membership','D_expected_AP']}
 result['diagnostics']=res['diagnostics'];inner_results[k]=result
 dump(OUT/'inner_grid.json',inner_results)
 print('INNER',k,{n:r['map_at_5'] for n,r in result.items() if n!='diagnostics'},flush=True)
k=min(cfg['kappa_grid'],key=lambda v:(-inner_results[v]['C_membership']['map_at_5'],v))
dev=observed(dev);model,res=execute(dev,k)
r3pred=json.loads(Path('experiments/R3/development_predictions.json').read_text())[original['selected']]['predictions']
pred={'A_R3':r3pred,**{n:res[n] for n in ['B_mean','C_membership','D_expected_AP']}}
metrics={n:checked_metrics(dev['relevant'],dev['users'],p)[0] for n,p in pred.items()}
best=max(['B_mean','C_membership','D_expected_AP'],key=lambda n:metrics[n]['map_at_5'])
def blend(s,r3_scores,distribution_list,p,w):
 rank=np.zeros_like(p)
 for u,row in enumerate(distribution_list):
  loc={int(item):i for i,item in enumerate(s['pool_ids'][u])}
  order=[loc[i] for i in row];order += [int(i) for i in np.argsort(-p[u],kind='stable') if int(i) not in order]
  rank[u,order]=1/(60+np.arange(1,len(order)+1))
 return rank_pool(s,(1-w)*reciprocal_ranks(s,r3_scores)+w*rank)[0]
for w in cfg['blend_weights']:
 name='E_blend_'+str(w);pred[name]=blend(dev,dev['R3_pool_scores'],res[best],res['membership_scores'],w);metrics[name]=checked_metrics(dev['relevant'],dev['users'],pred[name])[0]
winner=max([n for n in pred if n!='A_R3'],key=lambda n:metrics[n]['map_at_5'])
freeze={'kappa':k,'distribution_rule':best,'challenger':winner,'development_metrics':metrics,'selection_rule':cfg['selection'],'frozen_before_late_audit':True}
dump(OUT/'frozen_shortlist.json',freeze);dump(OUT/'development_count_diagnostics.json',res['diagnostics'])
dump(OUT/'development_predictions.json',pred)
print('DEVELOPMENT', {n:m['map_at_5'] for n,m in metrics.items()},'FROZEN',winner,flush=True)
# Compare the SAME R3 predictions to the full candidate oracle and explain conventions.
dump(OUT/'candidate_headroom.json',{'candidate_diagnostics':candidate_diagnostics(dev['pool_ids'],dev['relevant'],dev['users']),
 'R3_map':metrics['A_R3']['map_at_5'],'remaining_oracle_headroom':dev['candidate_diagnostics']['oracle_map_at_5_ceiling']-metrics['A_R3']['map_at_5'],
 'definition':'Micro recall pools labels; oracle MAP averages per-user retrieved/full-relevance fractions and includes empty users as zero.'})
# Shared-draw tail and Monte Carlo checks on 30 users, not another target window.
_,expanded=execute(dev,k,limit=30,stability=True)
_,larger=execute(dev,k,limit=30,draws=2048)
dump(OUT/'simulation_stability.json',{'expanded_pool':expanded['diagnostics'],
 'C_top5_same_512_vs_2048_fraction':float(np.mean([a==b for a,b in zip(res['C_membership'][:30],larger['C_membership'])])),
 'D_top5_same_512_vs_2048_fraction':float(np.mean([a==b for a,b in zip(res['D_expected_AP'][:30],larger['D_expected_AP'])])),
 'limitation':'Finite Monte Carlo and a truncated competitor pool; independent item rates do not learn all behavioral correlations.'})
if metrics[winner]['map_at_5']<=metrics['A_R3']['map_at_5']:
 dump(OUT/'decision.json',{'status':'PAUSED_NO_SUPPORTED_CHALLENGER','reason':'All tested count/blend contenders lose on development; no audit consumed and no CSV generated.','freeze':freeze});print('PAUSE: no supported development challenger.');sys.exit()
# Open the planned corrected late audit once, AFTER saving the shortlist.
audit=make_snapshot(data,events,epoch('2020-06-29'),epoch('2020-07-27'),r3cfg,r2cfg,r1cfg,label=False)
audit=observed(audit);target,_=qualifying_interactions(events,audit['start'],audit['end']);audit['relevant'],_=target_sets(target)
with threadpool_limits(limits=1):a_score=prod_cache['model'].predict_proba(audit['X'][:,prod_cache['columns']])[:,1]
audit['R3_pool_scores']=a_score.reshape(audit['pool_ids'].shape);a_pred=rank_pool(audit,a_score)[0]
am,ar=execute(audit,k)
candidate=ar[best] if not winner.startswith('E_') else blend(audit,a_score,ar[best],ar['membership_scores'],float(winner.split('_')[-1]))
audit_metrics={'A_R3':checked_metrics(audit['relevant'],audit['users'],a_pred)[0],winner:checked_metrics(audit['relevant'],audit['users'],candidate)[0]}
dump(OUT/'late_audit_once.json',{'metrics':audit_metrics,'shortlist':freeze,'previous_corrected_exposure':False,'previous_old_broad_exposure':True,'used_for_selection':True})
print('LATE AUDIT',audit_metrics,flush=True)
if audit_metrics[winner]['map_at_5']<audit_metrics['A_R3']['map_at_5']:
 dump(OUT/'decision.json',{'status':'PAUSED_AUDIT_LOSS','reason':'Frozen challenger loses the late audit; no last-slot artifact generated.'});sys.exit()
prod=prod_cache['snapshot'];prod['R3_pool_scores']=np.asarray(prod_cache['scores']).reshape(prod['pool_ids'].shape)
future,_,_=read_schedule('data/export_arh_31-42-final.csv',metadata=False);data['future'],_=canonical_schedule(future)
pm,pr=execute(prod,k)
chosen=pr[best] if not winner.startswith('E_') else blend(prod,prod['R3_pool_scores'],pr[best],pr['membership_scores'],float(winner.split('_')[-1]))
frame=data['template'].copy();frame['tv_show_id']=[' '.join(map(str,row)) for row in chosen]
checks=validate_submission(frame,data['template'],data['test_candidates']);content=frame.to_csv(index=False,lineterminator='\n').encode()
import hashlib
digest=hashlib.sha256(content).hexdigest()
for old in Path('submissions').glob('R[0123]*.csv'):assert sha256(old)!=digest
path=Path('submissions/R4_count_distribution.csv');immutable_bytes(path,content);immutable_bytes(path.with_suffix('.csv.sha256'),(digest+'  '+path.name+'\n').encode())
pd.to_pickle({'model':pm,'result':pr,'snapshot':prod,'selected':chosen,'config':freeze},'cache/r3/r4_verification.pkl')
dump(OUT/'manifest.json',{'round':'R4','status':'AWAITING_USER_SCORE','csv':str(path),'sha256':digest,'public_score':None,'private_score':None,
 'incumbent':'R3','incumbent_private_score':.42671,'selected':freeze,'csv_checks':checks,'runtime_seconds':time.perf_counter()-start,
 'source_sha256':{n:sha256('scripts/'+n) for n in ['run_r4.py','count_r4.py']},'command':'.venv/bin/python scripts/run_r4.py',
 'late_audit_used_for_selection':True,'distribution_assumptions':'Beta-binomial, exchangeable independent programme rates; availability is not exposure; pool truncated',
 'next':'STOP for user R4 scores; only then designate best scored file.'})
print('R4 READY',path.resolve(),digest,flush=True)
