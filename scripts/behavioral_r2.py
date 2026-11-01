"""R2 recency, schedule opportunity and pooled channel/time habit forecasts."""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from completion_pipeline import clip_events
from round_models import ranked_lists


def decay_weights(ends,cutoff,half_life):
    age=(cutoff-np.asarray(ends,dtype=np.int64))/86400
    assert (age>=0).all(),'Future observation in recency features'
    return np.ones(len(age)) if half_life is None else np.exp2(-age/half_life)


def effective_days(history_days,half_life):
    if half_life is None:return history_days
    scale=half_life/np.log(2)
    return scale*(-np.expm1(-history_days/scale))


def qualifying_watches(events,lower,cutoff,schedule):
    e=clip_events(events,lower,cutoff)
    e=e.loc[e.overlap_seconds>=.8*e.original_duration].copy()
    e=e.sort_values(['u','airing','overlap_seconds','overlap_end'],ascending=[True,True,False,False]).drop_duplicates(['u','airing'])
    starts=schedule.set_index('airing').s
    e['broadcast_start']=e.airing.map(starts)
    assert e.broadcast_start.notna().all() and (e.overlap_end<=cutoff).all()
    return e


def habit_cell(channels,starts):
    start=pd.to_datetime(np.asarray(starts,dtype=np.int64),unit='s')
    return np.asarray(channels,dtype=np.int64)*8+(start.day_of_week>=5).astype(np.int64)*4+start.hour.to_numpy()//6


def dense_user_counts(watches,values,users,key_values,keys):
    row_map={int(u):i for i,u in enumerate(users)}
    keep=watches.u.isin(users).to_numpy()
    rows=watches.u.loc[keep].map(row_map).to_numpy()
    columns=pd.Index(keys).get_indexer(np.asarray(key_values)[keep])
    selected=columns>=0
    return csr_matrix((np.asarray(values)[keep][selected],(rows[selected],columns[selected])),shape=(len(users),len(keys))).toarray()


def target_matrix(target,candidates,key_column,keys):
    items=pd.Index(candidates).get_indexer(target.item)
    cells=pd.Index(keys).get_indexer(target[key_column])
    keep=(items>=0)&(cells>=0)
    return csr_matrix((np.ones(int(keep.sum())),(items[keep],cells[keep])),shape=(len(candidates),len(keys)))


def context(data,events,cutoff,end,candidates,target_schedule,config):
    started=time.perf_counter();lower=int(data['raw'].s.min())//86400*86400
    watch=qualifying_watches(events,lower,cutoff,data['schedule'])
    historical=data['schedule'].loc[(data['schedule'].s<cutoff)&(data['schedule'].e>lower)&(data['schedule'].item>0)].copy()
    target=target_schedule.loc[(target_schedule.s<end)&(target_schedule.e>cutoff)&(target_schedule.item>0)].copy()
    historical['cell']=habit_cell(historical.ch,historical.s)
    target['cell']=habit_cell(target.ch,target.s)
    watch['cell']=habit_cell(watch.ch,watch.broadcast_start)
    users=data['evaluation_users'];candidates=np.asarray(candidates,dtype=np.int64)
    train=watch.groupby(['u','item'],as_index=False).agg(count=('airing','nunique'),last_end=('overlap_end','max'))
    pop=train.groupby('item').u.nunique().reindex(candidates,fill_value=0).to_numpy(dtype=float)
    last=np.zeros((len(users),len(candidates)),dtype=np.int64)
    uid={u:i for i,u in enumerate(users)};iid={int(i):j for j,i in enumerate(candidates)}
    for row in train.loc[train.u.isin(users)&train.item.isin(candidates)].itertuples(index=False):last[uid[row.u],iid[row.item]]=row.last_end
    return {'watch':watch,'historical':historical,'target':target,'train':train,'users':users,'candidates':candidates,
            'fallback':pop,'last_end':last,'cutoff':cutoff,'end':end,'lower':lower,
            'history_days':(cutoff-lower)/86400,'forecast_days':(end-cutoff)/86400,
            'config':config,'context_seconds':time.perf_counter()-started}


def features(ctx,half_life):
    tick=time.perf_counter();w,h,t=ctx['watch'],ctx['historical'],ctx['target']
    users,candidates=ctx['users'],ctx['candidates'];cfg=ctx['config'];cutoff=ctx['cutoff']
    weights=decay_weights(w.overlap_end,cutoff,half_life)
    opportunity_weights=decay_weights(np.minimum(h.e,cutoff),cutoff,half_life)
    length=effective_days(ctx['history_days'],half_life)
    counts=dense_user_counts(w,weights,users,w.item,candidates)
    repeat=counts*(ctx['forecast_days']/length)
    past=pd.Series(opportunity_weights,index=h.item).groupby(level=0).sum().reindex(candidates,fill_value=0).to_numpy()
    future=t.groupby('item').size().reindex(candidates,fill_value=0).to_numpy(dtype=float)
    all_count=pd.Series(weights,index=w.item).groupby(level=0).sum().reindex(candidates,fill_value=0).to_numpy()
    n_users=max(1,w.u.nunique())
    global_rate=np.divide(all_count,n_users*past,out=np.zeros_like(past),where=past>0)
    k=cfg['opportunity_prior_airings']
    posterior=(counts+k*global_rate)/(past+k)
    posterior=np.clip(posterior,0,1)
    opportunity=posterior*future
    opportunity[counts<=0]=0 # Repeat-only schedule adjustment.
    ratio=(future+k)/(past+k)*(length/ctx['forecast_days'])
    ratio=np.clip(ratio,*cfg['opportunity_ratio_bounds'])
    capped=repeat*ratio
    # Hierarchical smoothing over labelled broadcast opportunities, not exposure.
    channels=np.sort(np.unique(np.r_[h.ch.to_numpy(),t.ch.to_numpy()]))
    cells=np.sort(np.unique(np.r_[h.cell.to_numpy(),t.cell.to_numpy()]))
    hist_ch=pd.Series(opportunity_weights,index=h.ch).groupby(level=0).sum().reindex(channels,fill_value=0).to_numpy()
    hist_cell=pd.Series(opportunity_weights,index=h.cell).groupby(level=0).sum().reindex(cells,fill_value=0).to_numpy()
    global_ch=pd.Series(weights,index=w.ch).groupby(level=0).sum().reindex(channels,fill_value=0).to_numpy()
    global_ch=np.divide(global_ch,n_users*hist_ch,out=np.zeros_like(hist_ch),where=hist_ch>0)
    user_ch=dense_user_counts(w,weights,users,w.ch,channels)
    user_cell=dense_user_counts(w,weights,users,w.cell,cells)
    shrink=cfg['habit_prior_airings']
    prob_ch=(user_ch+shrink*global_ch)/(hist_ch+shrink)
    channel_of_cell=pd.Index(channels).get_indexer(cells//8)
    assert (channel_of_cell>=0).all()
    prob_cell=(user_cell+shrink*prob_ch[:,channel_of_cell])/(hist_cell+shrink)
    prob_ch=np.clip(prob_ch,0,1);prob_cell=np.clip(prob_cell,0,1)
    channel_scores=(target_matrix(t,candidates,'ch',channels)@prob_ch.T).T
    time_scores=(target_matrix(t,candidates,'cell',cells)@prob_cell.T).T
    cold=~np.isin(users,w.u.unique())
    for scores in [repeat,opportunity,capped,channel_scores,time_scores]:
        scores[cold]=0
        assert np.isfinite(scores).all() and (scores>=0).all()
    detail={'half_life_days':half_life,'effective_history_days':float(length),'forecast_days':ctx['forecast_days'],
            'historical_airings':len(h),'target_airings':len(t),'qualifying_broadcast_watches':len(w),
            'channel_cells':len(channels),'channel_time_cells':len(cells),
            'historical_max_airing_start':int(h.s.max()),'qualifying_max_end':int(w.overlap_end.max()),
            'feature_cutoff':cutoff,'no_future_viewing_used':True,'cold_users':int(cold.sum()),
            'opportunity_ratio_min':float(ratio.min()),'opportunity_ratio_max':float(ratio.max()),
            'fit_feature_seconds':time.perf_counter()-tick}
    return {'repeat':repeat,'opportunity':opportunity,'capped_opportunity':capped,
            'channel_habit':np.asarray(channel_scores),'time_habit':np.asarray(time_scores),'details':detail}


def predictions(ctx,scores,name):
    tick=time.perf_counter()
    ids,fallback=ranked_lists(scores,ctx['candidates'],ctx['fallback'],ctx['last_end'])
    histories={int(u):set(g.item) for u,g in ctx['train'].groupby('u')}
    details={'distinct_lists':len({tuple(row) for row in ids}),'distinct_items':len({i for row in ids for i in row}),
             'fallback_users_any':int(fallback.any(axis=1).sum()),'fallback_users_full':int(fallback.all(axis=1).sum()),
             'fallback_slots':int(fallback.sum()),
             'repeat_share':float(np.mean([sum(i in histories.get(u,set()) for i in row)/5 for u,row in zip(ctx['users'],ids)])),
             'ranking_seconds':time.perf_counter()-tick,'fallback_version':ctx['config']['fallback_version']}
    return {'name':name,'predictions':ids,'fallback':fallback,'details':details}
