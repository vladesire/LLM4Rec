"""R1 independent count-input models with one common completion-popularity fallback."""
from __future__ import annotations

import inspect
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from models import content_document


def count_matrix(interactions, n_users, items):
    columns=pd.Index(items).get_indexer(interactions.item)
    keep=columns>=0
    x=csr_matrix((interactions['count'].to_numpy(dtype=np.float32)[keep],
                  (interactions.u.to_numpy()[keep], columns[keep])),shape=(n_users,len(items)))
    return x


def ranked_lists(scores, candidates, fallback, secondary=None):
    candidates=np.asarray(candidates,dtype=np.int64)
    fallback_order=np.lexsort((candidates,-fallback))
    predictions,flags=[],[]
    for u,row in enumerate(scores):
        valid=np.flatnonzero(np.isfinite(row)&(row>0))
        if secondary is None:
            order=valid[np.lexsort((candidates[valid],-row[valid]))]
        else:
            order=valid[np.lexsort((candidates[valid],-secondary[u,valid],-row[valid]))]
        chosen=order[:5].tolist(); used=[False]*len(chosen)
        for i in fallback_order:
            if len(chosen)==5:break
            if int(i) not in chosen:
                chosen.append(int(i));used.append(True)
        assert len(chosen)==5
        predictions.append(candidates[chosen].tolist());flags.append(used)
    return predictions,np.asarray(flags,dtype=bool)


def score_model(name, train, metadata, cutoff, n_users, evaluation_users, candidates, config, r0_broad=None):
    start=time.perf_counter()
    candidates=np.asarray(candidates,dtype=np.int64); users=np.asarray(evaluation_users,dtype=np.int64)
    fallback=train.groupby('item').u.nunique().reindex(candidates,fill_value=0).to_numpy(dtype=np.float32)
    cold=~np.isin(users,train.u.unique())
    details={'input':'distinct >=80%-continuous-session qualifying broadcast counts',
             'fitting_users':int(train.u.nunique()),'fitting_interactions':len(train),
             'cold_qualifying_history_users':int(cold.sum()),'candidate_items':len(candidates),
             'fallback_version':config['fallback_version'],'repeats_allowed':True}
    secondary=None
    if name in ['r0_strategy','completion_popularity']:
        source=r0_broad if name=='r0_strategy' else train
        values=source.groupby('item').u.nunique().reindex(candidates,fill_value=0).to_numpy(dtype=np.float32)
        fit_seconds=time.perf_counter()-start;tick=time.perf_counter()
        scores=np.broadcast_to(values,(len(users),len(candidates))).copy()
        details['input']='R0 original broad >=300-second interaction policy' if name=='r0_strategy' else details['input']
        details['method_known_candidates']=int((values>0).sum())
    elif name=='personal_frequency':
        matrix=count_matrix(train,n_users,candidates)
        fit_seconds=time.perf_counter()-start;tick=time.perf_counter()
        scores=matrix[users].toarray()
        # Epoch-second timestamps need exact integer tie precision, not float32.
        secondary=np.zeros_like(scores,dtype=np.int64)
        uid={int(u):i for i,u in enumerate(users)};iid={int(item):i for i,item in enumerate(candidates)}
        for row in train.loc[train.u.isin(users)&train.item.isin(candidates)].itertuples(index=False):
            secondary[uid[row.u],iid[row.item]]=row.last_end
        details['tie_break']='last qualifying overlap end descending, then ID ascending'
    elif name=='content':
        prefix=metadata.loc[metadata.s<cutoff].drop_duplicates('item',keep='first').sort_values('item')
        assert (prefix.s<cutoff).all()
        items=prefix.item.to_numpy(dtype=np.int64)
        features=TfidfVectorizer(lowercase=False,token_pattern=r'(?u)\S+',dtype=np.float32,norm='l2',smooth_idf=True).fit_transform(
            [content_document(row) for row in prefix.itertuples(index=False)])
        history=count_matrix(train,n_users,items);history.data=np.log1p(history.data)
        profiles=normalize(history@features,copy=False)
        fit_seconds=time.perf_counter()-start;tick=time.perf_counter()
        ix=pd.Index(items).get_indexer(candidates);known=ix>=0
        scores=np.full((len(users),len(candidates)),-np.inf,dtype=np.float32)
        scores[:,known]=(profiles[users]@features[ix[known]].T).toarray()
        scores[cold]=-np.inf
        details.update({'profile_weight':'log1p(qualifying broadcast count)', 'vocabulary_features':features.shape[1],
                        'metadata_max_start':int(prefix.s.max()),'metadata_cutoff':cutoff,
                        'method_known_candidates':int(known.sum()),'prefix_only_learned_preprocessing':True})
    elif name=='als':
        from implicit.cpu.als import AlternatingLeastSquares
        items=np.sort(train.item.unique());matrix=count_matrix(train,n_users,items)
        confidence=matrix.copy();confidence.data=1+config['als_confidence_alpha']*np.log1p(confidence.data)
        model=AlternatingLeastSquares(factors=config['als_factors'],iterations=config['als_iterations'],
                                     regularization=config['als_regularization'],alpha=1,random_state=config['seed'],
                                     num_threads=1,use_cg=False,dtype=np.float32)
        with threadpool_limits(limits=1):model.fit(confidence,show_progress=False)
        assert model.user_factors.shape==(n_users,config['als_factors'])
        assert model.item_factors.shape==(len(items),config['als_factors'])
        assert np.isfinite(model.user_factors).all() and np.isfinite(model.item_factors).all()
        fit_seconds=time.perf_counter()-start;tick=time.perf_counter()
        ix=pd.Index(items).get_indexer(candidates);known=ix>=0
        scores=np.full((len(users),len(candidates)),-np.inf,dtype=np.float32)
        with threadpool_limits(limits=1):scores[:,known]=model.user_factors[users]@model.item_factors[ix[known]].T
        scores[cold]=-np.inf
        details.update({'matrix_shape':list(matrix.shape),'matrix_orientation':'rows=user codes, columns=programme IDs',
                        'user_factors_shape':list(model.user_factors.shape),'item_factors_shape':list(model.item_factors.shape),
                        'confidence':'observed 1+10*log1p(count); unobserved 1; implicit alpha=1',
                        'recommend_signature':str(inspect.signature(model.recommend)),
                        'ranking_implementation':'direct dot products over eligible item factors; no already-watched exclusion',
                        'filter_already_liked_items':False,'method_known_candidates':int(known.sum())})
        # Inspect actual library repeat setting and ID mapping on one warm user.
        warm_positions=np.flatnonzero(~cold)
        if len(warm_positions) and known.sum()>=5:
            pos=int(warm_positions[0]);u=int(users[pos]); model_ids,model_scores=model.recommend(
                u,confidence[u],N=5,filter_already_liked_items=False,items=ix[known])
            mapped=items[model_ids].tolist();expected_scores={int(i):float(s) for i,s in zip(candidates[known],scores[pos,known])}
            assert all(np.isclose(float(s),expected_scores[int(i)],atol=1e-5) for i,s in zip(mapped,model_scores))
            details['library_recommend_explicit_repeat_and_mapping_check_passed']=True
    else:raise ValueError(name)
    predictions,backfill=ranked_lists(scores,candidates,fallback,secondary)
    infer_seconds=time.perf_counter()-tick
    histories={int(u):set(g.item) for u,g in train.groupby('u')}
    details.update({'fallback_users_any':int(backfill.any(axis=1).sum()),'fallback_users_full':int(backfill.all(axis=1).sum()),
                    'fallback_slots':int(backfill.sum()),'distinct_recommendation_lists':len({tuple(x) for x in predictions}),
                    'distinct_recommended_items':len({x for row in predictions for x in row}),
                    'qualifying_history_repeat_share':float(np.mean([sum(i in histories.get(int(u),set()) for i in row)/5 for u,row in zip(users,predictions)]))})
    return {'name':name,'predictions':predictions,'scores':scores,'fallback':backfill,'popularity':fallback,
            'secondary':secondary,'fit_seconds':fit_seconds,'inference_seconds':infer_seconds,'details':details}
