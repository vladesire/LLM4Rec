"""Past-only snapshot features and target-independent candidate pools for R3."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from behavioral_r2 import context,features,decay_weights,dense_user_counts,predictions
from completion_pipeline import epoch,clip_events,qualifying_interactions,target_sets
from data_pipeline import canonical_schedule,read_schedule,sha256
from round_models import score_model


def full_order(values,items,last=None):
    valid=np.flatnonzero(np.isfinite(values)&(values>0))
    return valid[np.lexsort((items[valid],-values[valid]))] if last is None else valid[np.lexsort((items[valid],-last[valid],-values[valid]))]


def candidate_pool(source_scores,items,last,popularity,quotas,size):
    """API intentionally takes no target labels or future viewing."""
    n=source_scores['R2'].shape[0];output=np.empty((n,size),dtype=np.int32)
    pop_order=np.lexsort((items,-popularity))
    for u in range(n):
        selected=[];used=set();orders={}
        for name,quota in quotas.items():
            order=pop_order if name=='popularity' else full_order(source_scores[name][u],items,last[u] if name in ['personal_frequency','R2'] else None)
            orders[name]=order;taken=0
            for ix in order:
                ix=int(ix)
                if ix not in used:selected.append(ix);used.add(ix);taken+=1
                if taken==quota:break
        for order in [orders['R2'],pop_order]:
            for ix in order:
                if len(selected)>=size:break
                ix=int(ix)
                if ix not in used:selected.append(ix);used.add(ix)
        assert len(selected)==size and len(set(selected))==size
        output[u]=selected
    return output


def checked_metrics(relevant,users,predicted):
    from core import ap_at_5
    ap=[ap_at_5(relevant.get(u,set()),row) for u,row in zip(users,predicted)]
    sizes=np.asarray([len(relevant.get(u,set())) for u in users]);array=np.asarray(ap)
    return {'map_at_5':float(array.mean()),'nonempty_map_at_5':float(array[sizes>0].mean()) if (sizes>0).any() else None,
            'five_label_map_at_5':float(array[sizes>=5].mean()) if (sizes>=5).any() else None,
            'users':len(users),'empty_users':int((sizes==0).sum()),'nonempty_users':int((sizes>0).sum()),
            'five_label_users':int((sizes>=5).sum())},ap


def candidate_diagnostics(pool_ids,relevant,users):
    found=[];total=[];ceil=[]
    for u,pool in zip(users,pool_ids):
        rel=relevant.get(u,set());hit=len(rel&set(pool));found.append(hit);total.append(len(rel))
        ceil.append(hit/min(5,len(rel)) if rel else 0.)
    sizes=np.asarray(total);found=np.asarray(found)
    macro=np.divide(found,sizes,out=np.zeros(len(users)),where=sizes>0)
    return {'candidate_size':pool_ids.shape[1],'micro_recall':float(found.sum()/max(1,sizes.sum())),
            'macro_recall_all_users_empty_zero':float(macro.mean()),
            'macro_recall_nonempty':float(macro[sizes>0].mean()) if (sizes>0).any() else None,
            'oracle_map_at_5_ceiling':float(np.mean(ceil)),'retrieved_labels':int(found.sum()),'total_labels':int(sizes.sum()),
            'nonempty_users_with_all_labels_retrieved':int(((found==sizes)&(sizes>0)).sum()),
            'warning':'Oracle ceiling, not achieved model performance. Complete relevance sets remain in denominators.'}


def make_snapshot(data,events,start,end,config,r2config,r1config,label=True):
    if label:
        assert end<=epoch('2020-06-29') and start<epoch('2020-06-29'),'Corrected late audit is sealed until R4'
    cache=Path('cache/r3');cache.mkdir(parents=True,exist_ok=True)
    signature={'start':start,'end':end,'label':label,'config':config,'feature_source':sha256(__file__),
               'r2_source':sha256('scripts/behavioral_r2.py'),'r1_source':sha256('scripts/round_models.py'),
               'completion_source':sha256('scripts/completion_pipeline.py'),'input_hashes':data['audit']['input_sha256']}
    key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:18];path=cache/(key+'.pkl')
    if path.exists():return pd.read_pickle(path)
    tick=time.perf_counter()
    if start==epoch('2020-07-27'):
        target_schedule,_,_=read_schedule('data/export_arh_31-42-final.csv',metadata=False)
        assert sha256('data/export_arh_31-42-final.csv')==data['audit']['input_sha256']['export_arh_31-42-final.csv']
        target_schedule,_=canonical_schedule(target_schedule);items=np.asarray(data['test_candidates'],dtype=np.int64)
    else:
        target_schedule=data['schedule'];available=data['availability']
        items=np.sort(available.loc[(available.s<end)&(available.e>start)&(available.item>0),'item'].unique())
    ctx=context(data,events,start,end,items,target_schedule,r2config);bundle=features(ctx,56)
    train=ctx['train'];users=ctx['users'];n_users=len(data['users'])
    personal=score_model('personal_frequency',train,data['metadata'],start,n_users,users,items,r1config)
    als=score_model('als',train,data['metadata'],start,n_users,users,items,r1config)
    content=score_model('content',train,data['metadata'],start,n_users,users,items,r1config)
    r2=.75*bundle['repeat']+.25*bundle['time_habit']
    source_scores={'personal_frequency':personal['scores'],'R2':r2,'als':als['scores'],'content':content['scores']}
    pool=candidate_pool(source_scores,items,ctx['last_end'],ctx['fallback'],config['candidate_source_quotas'],config['candidate_size'])
    pool_ids=items[pool];rows=np.arange(len(users))[:,None];last=ctx['last_end'][rows,pool]
    r2_lists=predictions(ctx,r2,'R2')['predictions']
    assert all(set(top).issubset(set(ids)) for top,ids in zip(r2_lists,pool_ids))
    columns=[];names=[];families={}
    def add(name,values,family):
        values=np.asarray(values)
        if values.ndim==1:values=np.broadcast_to(values[:,None],pool.shape)
        assert values.shape==pool.shape
        names.append(name);families[name]=family;columns.append(values.ravel().astype(np.float32))
    weeks=max(ctx['history_days']/7,1/7);forecast_weeks=ctx['forecast_days']/7
    counts=personal['scores'];total_user=ctx['train'].groupby('u')['count'].sum().reindex(users,fill_value=0).to_numpy(dtype=float)
    add('qualifying_count_per_history_week',counts[rows,pool]/weeks,'personal')
    add('personal_frequency_share',counts[rows,pool]/np.maximum(total_user[:,None],1),'personal')
    add('repeat56_per_forecast_week',bundle['repeat'][rows,pool]/forecast_weeks,'personal')
    for source,family in [('personal_frequency','personal'),('R2','schedule'),('als','collaborative'),('content','collaborative')]:
        values=source_scores[source];rank=np.full(values.shape,len(items)+1,dtype=float)
        for u,row in enumerate(values):
            order=full_order(row,items,ctx['last_end'][u] if source in ['personal_frequency','R2'] else None)
            rank[u,order]=np.arange(1,len(order)+1)
        add(source+'_log_rank',np.log1p(rank[rows,pool]),family)
    add('last_qualifying_age_days',np.where(last>0,(start-last)/86400,400),'personal')
    add('has_qualifying_personal_history',(counts[rows,pool]>0).astype(float),'personal')
    # Completion summaries contain only this prefix's session/airing overlaps.
    clipped=clip_events(events,ctx['lower'],start)
    partial=clipped.groupby(['u','item'],as_index=False).agg(mean_completion=('completion_fraction','mean'),
            overlap_seconds=('overlap_seconds','sum'),observed_airings=('airing','nunique'))
    index=pd.MultiIndex.from_arrays([np.repeat(users,pool.shape[1]),pool_ids.ravel()])
    partial=partial.set_index(['u','item']).reindex(index,fill_value=0)
    add('mean_observed_completion',partial.mean_completion.to_numpy().reshape(pool.shape),'personal')
    add('overlap_hours_per_history_week',partial.overlap_seconds.to_numpy().reshape(pool.shape)/3600/weeks,'personal')
    add('observed_airings_per_history_week',partial.observed_airings.to_numpy().reshape(pool.shape)/weeks,'personal')
    for component in ['opportunity','capped_opportunity','channel_habit','time_habit']:
        add(component+'_per_forecast_week',bundle[component][rows,pool]/forecast_weeks,'schedule')
    add('R2_score_per_forecast_week',r2[rows,pool]/forecast_weeks,'schedule')
    future_air=ctx['target'].groupby('item').size().reindex(items,fill_value=0).to_numpy()/forecast_weeks
    hist_weights=decay_weights(np.minimum(ctx['historical'].e,start),start,56)
    past_air=pd.Series(hist_weights,index=ctx['historical'].item).groupby(level=0).sum().reindex(items,fill_value=0).to_numpy()/(bundle['details']['effective_history_days']/7)
    add('future_airings_per_week',np.broadcast_to(future_air,(len(users),len(items)))[rows,pool],'schedule')
    add('past56_airings_per_effective_week',np.broadcast_to(past_air,(len(users),len(items)))[rows,pool],'schedule')
    for source in ['als','content']:
        values=source_scores[source][rows,pool]
        add(source+'_score',np.where(np.isfinite(values),values,np.nan),'collaborative')
        add(source+'_known',np.isfinite(values).astype(float),'collaborative')
    prefix_raw=data['raw'].loc[data['raw'].s<start]
    first=prefix_raw.groupby('u').s.min().reindex(users)
    observed_weeks=((start-first)/86400/7).fillna(0).to_numpy()
    unique=ctx['train'].groupby('u').item.nunique().reindex(users,fill_value=0).to_numpy(dtype=float)
    add('calendar_history_weeks',np.full(len(users),weeks),'reliability')
    add('observed_user_history_weeks',observed_weeks,'reliability')
    add('user_qualifying_count_per_calendar_week',total_user/weeks,'reliability')
    add('user_distinct_programmes_per_calendar_week',unique/weeks,'reliability')
    add('user_cold_qualifying',(total_user==0).astype(float),'reliability')
    add('personal_count_per_observed_week',counts[rows,pool]/np.maximum(observed_weeks[:,None],1/7),'personal')
    X=np.column_stack(columns);assert not np.isinf(X).any()
    if label:
        target,_=qualifying_interactions(events,start,end);relevant,_=target_sets(target)
        y=np.asarray([[item in relevant.get(u,set()) for item in ids] for u,ids in zip(users,pool_ids)],dtype=np.int8).ravel()
        diagnostics=candidate_diagnostics(pool_ids,relevant,users)
        provider={name:checked_metrics(relevant,users,model['predictions'])[0] for name,model in [('personal',personal),('als',als),('content',content)]}
        provider['R2']=checked_metrics(relevant,users,r2_lists)[0]
    else:relevant=None;y=None;diagnostics=None;provider=None
    snapshot={'start':start,'end':end,'users':users,'candidates':items,'pool_indices':pool,'pool_ids':pool_ids,
              'X':X,'y':y,'feature_names':names,'feature_families':families,'relevant':relevant,
              'candidate_diagnostics':diagnostics,'provider_metrics':provider,'R2_predictions':r2_lists,
              'R2_pool_scores':r2[rows,pool],'last_end':last,'fallback':ctx['fallback'],
              'prefix_max_observation_end':int(ctx['watch'].overlap_end.max()),
              'prefix_metadata_cutoff':start,'history_weeks':weeks,'forecast_weeks':forecast_weeks,
              'signature':signature,'snapshot_seconds':time.perf_counter()-tick,
              'candidate_policy':'source quotas then R2/popularity; no target labels accepted by candidate function',
              'grouping':'contiguous (cutoff,user) blocks; every block contains candidate_size distinct IDs'}
    pd.to_pickle(snapshot,path)
    print('R3 SNAPSHOT',pd.to_datetime(start,unit='s').date(),'rows',len(X),'oracle',diagnostics,flush=True)
    return snapshot
