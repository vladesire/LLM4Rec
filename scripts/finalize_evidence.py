"""Consolidate saved round evidence; reconstruct only the two formal R4 examples."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from count_r4 import posterior,sample_decisions
from run_r3 import reciprocal_ranks,rank_pool
from core import ap_at_5

def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
x=pd.read_pickle('cache/completion/prepared.pkl');ev=pd.read_pickle('cache/completion/events.pkl')
dev=pd.read_pickle('cache/r3/development_verification.pkl');s=dev['snapshot'];s['R3_pool_scores']=dev['scores']['drop_collaborative_content']
m=posterior(x,ev,s,read('experiments/R2/config.json'),100)
r1=read('experiments/R1/formal_two_user_examples.json');r2=read('experiments/R2/formal_two_user_examples.json');r3=read('experiments/R3/formal_two_user_examples.json');pred=read('experiments/R4/development_predictions.json')['E_blend_0.1']
hist=pd.read_csv('experiments/R1/formal_two_user_histories.csv',dtype={'user_id':str});examples=[]
for a,b,c in zip(r1,r2,r3):
 uid=a['user_id'];u=x['users'].index(uid);pos=s['users'].index(u)
 sub=dict(s)
 for k in ['pool_indices','pool_ids','R2_pool_scores','R3_pool_scores','last_end']:sub[k]=s[k][[pos]]
 sub['users']=[u]
 sub['X']=s['X'][pos*200:(pos+1)*200]
 mm=dict(m)
 for k in ['a','b','mean','n','A','cold']:mm[k]=m[k][[pos]]
 with threadpool_limits(limits=1):res=sample_decisions(mm,sub,512,42)
 ids=sub['pool_ids'][0];lookup={int(i):j for j,i in enumerate(ids)};order=[lookup[i] for i in res['D_expected_AP'][0]]
 order += [int(i) for i in np.argsort(-res['membership_scores'][0],kind='stable') if int(i) not in order]
 rr=np.empty((1,200));rr[0,order]=1/(60+np.arange(1,201));score=.9*reciprocal_ranks(sub,sub['R3_pool_scores'])+.1*rr
 got,_=rank_pool(sub,score);assert got[0]==pred[pos]
 methods={k:a['methods'][k] for k in ['r0_strategy','completion_popularity','content','als','personal_frequency']}
 methods['R2']=b['variants']['blend_habit_25'];methods['R3']=c['methods']['drop_collaborative_content']
 # Metadata is for descriptive reporting, not a model feature. Prefer historical labels.
 metadata={int(r['item']):r for z in [a['methods'],b['variants'],c['methods']] for v in z.values() for r in v['recommendations']}
 rec=[]
 for rank,i in enumerate(got[0],1):
  r=metadata.get(i,{'title':'[No permitted historical metadata]','category':'[Unknown]'})
  rec.append(dict(rank=rank,item=i,title=r['title'],category=r['category'],method_score=float(score[0,lookup[i]]),target_relevant=i in set(a['target_ids'])))
 methods['R4']={'ap_at_5':ap_at_5(set(a['target_ids']),got[0]),'recommendations':rec}
 h=hist.loc[hist.user_id==uid]
 examples.append(dict(user_id=uid,relevant_ids=a['target_ids'],training_items=int(h.loc[h.phase=='development_fitting'].item.nunique()),training_broadcasts=int(h.loc[h.phase=='development_fitting'].qualifying_broadcast_count.sum()),methods=methods))
save('experiments/final/two_user_examples.json',examples)
lines=['# Personal review packet','Read the actual histories in experiments/R1/formal_two_user_histories.csv and original rows in experiments/R1/worked_completion_checks.md. No personal verification is claimed here.']
for e in examples:
 lines += ['## User '+e['user_id'],f"History: {e['training_items']} qualifying programmes, {e['training_broadcasts']} qualifying broadcasts. Target: {e['relevant_ids']}"]
 for name,v in e['methods'].items():
  lines += [f"### {name}; AP@5={v['ap_at_5']:.6f}"]+[f"{r['rank']}. {r['item']}: {r['title']} ({r['category']})" for r in v['recommendations']]
Path('experiments/final/PERSONAL_REVIEW.md').write_text('\n\n'.join(lines)+'\n')
print([(e['user_id'],e['training_items'],{n:v['ap_at_5'] for n,v in e['methods'].items()}) for e in examples])
