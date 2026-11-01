import sys,json,itertools
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import betabinom
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parent))
from count_r4 import posterior,sample_decisions,moments,greedy,expected_ap
from data_pipeline import dump
from run_r3 import rank_pool,reciprocal_ranks
from core import ap_at_5
x=pd.read_pickle('cache/completion/prepared.pkl');events=pd.read_pickle('cache/completion/events.pkl');dev=pd.read_pickle('cache/r3/development_verification.pkl');s=dev['snapshot'];s['R3_pool_scores']=dev['scores']['drop_collaborative_content']
cfg=json.loads(Path('experiments/R2/config.json').read_text());k=100;m=posterior(x,events,s,cfg,k)
with threadpool_limits(limits=1):
 a=sample_decisions(m,s,512,42,users_limit=30);b=sample_decisions(m,s,2048,42,users_limit=30)
def lists(res):
 out=[]
 for u in range(30):
  ids=s['pool_ids'][u];lookup={int(i):j for j,i in enumerate(ids)};order=[lookup[i] for i in res['D_expected_AP'][u]]
  order += [int(i) for i in np.argsort(-res['membership_scores'][u],kind='stable') if int(i) not in order]
  rr=np.empty(200);rr[order]=1/(60+np.arange(1,201));r3=s['R3_pool_scores'][u];o=np.lexsort((ids,-s['last_end'][u],-r3));r=np.empty(200);r[o]=1/(60+np.arange(1,201));v=.9*r+.1*rr
  chosen=ids[np.lexsort((ids,-s['last_end'][u],-v))[:5]].tolist();out.append(chosen)
 return out
stability=float(np.mean([p==q for p,q in zip(lists(a),lists(b))]))
# Diagnostic count likelihood/coverage on fixed random pairs across all scheduled items.
from completion_pipeline import qualifying_interactions
from round_models import count_matrix
actual,_=qualifying_interactions(events,s['start'],s['end']);counts=count_matrix(actual,len(x['users']),s['candidates'])[s['users']].toarray()
rng=np.random.default_rng(42);u=rng.integers(0,1260,10000);i=rng.integers(0,len(s['candidates']),10000)
y=counts[u,i];M=m['M'][i];aa=m['a'][u,i];bb=m['b'][u,i]
lo=betabinom.ppf(.05,M,aa,bb);hi=betabinom.ppf(.95,M,aa,bb);nll=-betabinom.logpmf(y,M,aa,bb)
activity=m['n'].sum(axis=1);median=np.median(activity[activity>0]);groups={}
for name,mask in {'known':m['n'][u,i]>0,'new':m['n'][u,i]==0,'high_activity':activity[u]>=median,'low_activity':activity[u]<median}.items():
 groups[name]={'pairs':int(mask.sum()),'mean_prediction':float(m['mean'][u,i][mask].mean()),'mean_observed':float(y[mask].mean()),'90pct_interval_coverage':float(((y>=lo)&(y<=hi))[mask].mean()),'mean_negative_log_likelihood':float(nll[mask].mean())}
T=np.array([[1,2,-1,-1,-1],[-1]*5,[2,3,4,-1,-1],[1,3,5,6,-1]]);ids=np.arange(1,7);_,aa,bb=moments(T,ids);g=greedy(aa,bb,ids);best=max(expected_ap(o,aa,bb) for o in itertools.permutations(range(6),5))
dump('experiments/R4/additional_checks.json',{'final_10pct_blend_same_list_512_vs_2048_first30':stability,'sample_groups':groups,'diagnostic_pairs':10000,'seed':42,'greedy_tiny_expected_AP':expected_ap(g,aa,bb),'exhaustive_tiny_optimum':best,'future_opportunity_violations':m['opportunity_violations'],'caution':'Uniform pair diagnostics are zero-dominated; neither likelihood nor small gains establish private-score improvement.'})
print('FINAL BLEND STABILITY',stability,'GROUPS',groups)
