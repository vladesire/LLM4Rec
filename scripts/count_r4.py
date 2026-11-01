"""Empirical-Bayes count forecasts and sampled expected-AP decisions for R4."""
import itertools
import numpy as np
import pandas as pd
from scipy.special import betaln
from scipy.stats import betabinom
from behavioral_r2 import context,dense_user_counts
from round_models import count_matrix

def opportunity_counts(schedule,keys,first,key):
    out=np.zeros((len(first),len(keys)),dtype=np.int32)
    groups={int(k):np.sort(g.e.to_numpy()) for k,g in schedule.groupby(key)}
    for j,k in enumerate(keys):
        ends=groups.get(int(k),np.empty(0));out[:,j]=len(ends)-np.searchsorted(ends,first,side='right')
    return out

def posterior(data,events,snapshot,r2config,kappa):
    ctx=context(data,events,snapshot['start'],snapshot['end'],snapshot['candidates'],data['schedule'] if snapshot['start']<1595808000 else data['future'],r2config)
    items=snapshot['candidates'];users=ctx['users'];cutoff=ctx['cutoff'];h=ctx['historical'];t=ctx['target'];w=ctx['watch']
    first=data['raw'].loc[data['raw'].s<cutoff].groupby('u').s.min().reindex(range(len(data['users'])),fill_value=cutoff).to_numpy()
    A=opportunity_counts(h,items,first[users],'item');A[first[users]>=cutoff]=0
    n=count_matrix(ctx['train'],len(data['users']),items)[users].toarray().astype(np.int32)
    violations=int((n>A).sum());assert violations==0,f'Historical count exceeds aligned opportunities: {violations}'
    channels=np.sort(np.unique(np.r_[h.ch,t.ch]));Ach=opportunity_counts(h,channels,first,'ch');Ach[first>=cutoff]=0
    global_n=w.groupby('ch').size().reindex(channels,fill_value=0).to_numpy()
    global_rate=np.divide(global_n,Ach.sum(axis=0),out=np.zeros(len(channels)),where=Ach.sum(axis=0)>0)
    Uch=dense_user_counts(w,np.ones(len(w)),users,w.ch,channels)
    prior_channel=(Uch+20*global_rate)/(Ach[users]+20)
    target=pd.crosstab(t.item,t.ch).reindex(index=items,columns=channels,fill_value=0).to_numpy()
    M=target.sum(axis=1).astype(np.int32)
    mu=(prior_channel@target.T)/np.maximum(M,1)
    mu=np.clip(mu,1e-6,1-1e-6)
    a=kappa*mu+n;b=kappa*(1-mu)+A-n
    mean=M*a/(a+b)
    return {'a':a,'b':b,'mean':mean,'n':n,'A':A,'M':M,'first':first[users],
            'cold':~np.isin(users,ctx['train'].u.unique()),'opportunity_violations':violations,
            'fallback':ctx['fallback'],'prior_mean':mu,'context':ctx}

def sampled_targets(a,b,M,ids,draws,seed,user):
    q=np.random.default_rng(np.random.SeedSequence([seed,int(user),0])).beta(a[:,None],b[:,None],size=(len(ids),draws))
    N=np.random.default_rng(np.random.SeedSequence([seed,int(user),1])).binomial(M[:,None],q)
    assert (N<=M[:,None]).all()
    # Integer key gives exact count-descending, ID-ascending ties.
    key=N.T.astype(np.int64)*(int(np.max(ids))+1)-ids[None,:]
    top=np.argpartition(-key,4,axis=1)[:,:5]
    k=np.take_along_axis(key,top,axis=1);top=np.take_along_axis(top,np.argsort(-k,axis=1),axis=1)
    target=ids[top].copy();target[np.take_along_axis(N.T,top,axis=1)<=0]=-1
    return target,N

def moments(targets,items):
    I=(targets[:,:,None]==np.asarray(items)[None,None,:]).any(axis=1).astype(float)
    D=(targets>=0).sum(axis=1);weight=np.divide(1.,D,out=np.zeros(len(D)),where=D>0)
    return I.mean(axis=0),(I*weight[:,None]).mean(axis=0),I.T@(I*weight[:,None])/len(D)

def expected_ap(order,a,b):
    return sum((a[i]+sum(b[j,i] for j in order[:r]))/(r+1) for r,i in enumerate(order))

def greedy(a,b,ids,length=5):
    chosen=[]
    for r in range(min(length,len(ids))):
        remaining=[i for i in range(len(ids)) if i not in chosen]
        chosen.append(min(remaining,key=lambda i:(-(a[i]+sum(b[j,i] for j in chosen))/(r+1),int(ids[i]))))
    return chosen

def sample_decisions(model,snapshot,draws=512,seed=42,limit=500,users_limit=None,stability=False):
    ids=snapshot['candidates'];pool=snapshot['pool_indices'];nusers=len(snapshot['users']) if users_limit is None else users_limit
    B=[];C=[];D=[];empty=[];coverage=[];model_sizes=[];tail_changes=[]
    sampled_scores=[];greedy_objectives=[]
    for u in range(nusers):
        recommendation=pool[u];mean=model['mean'][u]
        base=np.unique(np.r_[np.flatnonzero(model['n'][u]>0),np.argsort(-mean,kind='stable')[:limit],recommendation])
        extra=np.setdiff1d(np.argsort(-mean,kind='stable')[:1000],base)
        chosen=np.r_[base,extra] if stability else base
        targets,N=sampled_targets(model['a'][u,chosen],model['b'][u,chosen],model['M'][chosen],ids[chosen],draws,seed,snapshot['users'][u])
        p,_,_=moments(targets,ids[recommendation]);order=np.lexsort((ids[recommendation],-p))
        shortlist=np.unique(np.r_[order[:25],np.argsort(-snapshot.get('R3_pool_scores',snapshot['R2_pool_scores'])[u],kind='stable')[:5]])
        ps,a,b=moments(targets,ids[recommendation[shortlist]])
        decision=shortlist[greedy(a,b,ids[recommendation[shortlist]])]
        B.append(ids[recommendation[np.lexsort((ids[recommendation],-mean[recommendation]))[:5]]].tolist())
        C.append(ids[recommendation[order[:5]]].tolist());D.append(ids[recommendation[decision]].tolist())
        empty.append(float((targets<0).all(axis=1).mean()));model_sizes.append(len(chosen));sampled_scores.append(p)
        greedy_objectives.append(expected_ap([list(shortlist).index(i) for i in decision],a,b))
        if stability:
            original,_=sampled_targets(model['a'][u,base],model['b'][u,base],model['M'][base],ids[base],draws,seed,snapshot['users'][u])
            p0,_,_=moments(original,ids[recommendation]);tail_changes.append(float(np.abs(p-p0).max()))
        # Distributional interval checks use the same samples, without changing labels.
        if snapshot.get('observed_counts') is not None:
            actual=snapshot['observed_counts'][u,chosen];lo,hi=np.quantile(N,[.05,.95],axis=1)
            coverage.append({'covered':int(((actual>=lo)&(actual<=hi)).sum()),'pairs':len(chosen),
                             'predicted_total':float(mean.sum()),'observed_total':float(snapshot['observed_counts'][u].sum()),
                             'zero_mass':float(np.exp(betaln(model['a'][u,chosen],model['b'][u,chosen]+model['M'][chosen])-betaln(model['a'][u,chosen],model['b'][u,chosen])).mean())})
    return {'B_mean':B,'C_membership':C,'D_expected_AP':D,'membership_scores':np.asarray(sampled_scores),
            'diagnostics':{'draws':draws,'seed':seed,'modeling_pool_size_mean':float(np.mean(model_sizes)),
                          'empty_draw_fraction':float(np.mean(empty)),'count_diagnostics':coverage,
                          'maximum_membership_change_after_tail_expansion':max(tail_changes) if tail_changes else None,
                          'mean_model_expected_AP_of_greedy':float(np.mean(greedy_objectives))}}
