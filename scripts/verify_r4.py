"""Reconstruct the frozen R4 production list from fixed count samples, no fitting/audit."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from count_r4 import sample_decisions
from data_pipeline import dump,sha256
from run import validate_submission
from run_r3 import reciprocal_ranks,rank_pool
from core import ap_at_5

cache=pd.read_pickle('cache/r3/r4_verification.pkl');s=cache['snapshot'];m=cache['model'];out=Path('experiments/R4')
with threadpool_limits(limits=1):res=sample_decisions(m,s,512,42)
fallback=s['candidates'][np.lexsort((s['candidates'],-s['fallback']))].tolist()
for u,row in enumerate(res['D_expected_AP']):
 lookup={int(i):j for j,i in enumerate(s['pool_ids'][u])}
 chosen=[] if m['cold'][u] else [i for i in row if res['membership_scores'][u,lookup[i]]>0]
 for i in fallback:
  if len(chosen)==5:break
  if i not in chosen:chosen.append(i)
 res['D_expected_AP'][u]=chosen
rank=np.empty_like(res['membership_scores'])
for u,row in enumerate(res['D_expected_AP']):
 lookup={int(item):j for j,item in enumerate(s['pool_ids'][u])};order=[lookup[i] for i in row]
 order += [int(i) for i in np.argsort(-res['membership_scores'][u],kind='stable') if int(i) not in order]
 rank[u,order]=1/(60+np.arange(1,201))
score=.9*reciprocal_ranks(s,s['R3_pool_scores'])+.1*rank
assert np.isfinite(score).all()
pred,flags=rank_pool(s,score);assert pred==cache['selected']
manifest=json.loads((out/'manifest.json').read_text());frame=pd.read_csv(manifest['csv'],dtype=str)
assert frame.tv_show_id.tolist()==[' '.join(map(str,r)) for r in pred]
checks=validate_submission(frame,pd.read_csv('data/submission.csv',dtype=str),s['candidates'])
assert sha256(manifest['csv'])==manifest['sha256']
for number in range(4):
 old=json.loads(Path(f'experiments/R{number}/manifest.json').read_text());assert sha256(old['csv'])==old['sha256']
dev=pd.read_pickle('cache/r3/development_verification.pkl')['snapshot'];p=json.loads((out/'development_predictions.json').read_text());name=manifest['selected']['challenger']
d=np.array([ap_at_5(dev['relevant'].get(u,set()),b)-ap_at_5(dev['relevant'].get(u,set()),a) for u,a,b in zip(dev['users'],p['A_R3'],p[name])]);rng=np.random.default_rng(42)
ci=np.quantile(d[rng.integers(0,len(d),(2000,len(d)))].mean(axis=1),[.025,.975])
unc={'paired_development_delta':float(d.mean()),'bootstrap_interval_95':ci.tolist(),'replicates':2000,
 'users_improve':int((d>0).sum()),'users_worsen':int((d<0).sum()),'users_unchanged':int((d==0).sum()),
 'production_users_changed_vs_R3':int((frame.tv_show_id!=pd.read_csv('submissions/R3_boosted_reranker.csv',dtype=str).tv_show_id).sum()),
 'caution':'Tiny adaptively selected gain; no temporal-shift or selection correction; exploratory complement, not established superiority.'}
dump(out/'paired_uncertainty.json',unc)
checks.update({'deterministic_full_production_resampling_and_list_match':True,'finite_scores':True,'previous_R0_R3_files_unchanged':True,
 'no_new_audit_inspection':True,'no_fitting_or_upload':True,'historical_n_le_A':bool((m['n']<=m['A']).all()),
 'distinct_lists':len({tuple(r) for r in pred}),'fallback_users':int(flags.any(axis=1).sum())})
dump(out/'verification.json',checks)
print(checks,unc)
