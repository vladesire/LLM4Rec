"""Popularity, sparse content filtering, and genuine implicit ALS."""
from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from core import top_five


def interaction_matrix(interactions, n_users, items):
    index = pd.Index(items)
    cols = index.get_indexer(interactions.item)
    keep = cols >= 0
    return csr_matrix((interactions.seconds.to_numpy(dtype=np.float32)[keep],
                       (interactions.u.to_numpy()[keep], cols[keep])), shape=(n_users, len(items)))


def content_document(row):
    # Field prefixes prevent category names and title words from colliding.
    tokens = ["title:" + x for x in re.findall(r"\w{2,}", str(row.tv_show_title).lower())]
    for col, weight in [("tv_show_category", 2), ("tv_show_genre_1", 2),
                        ("tv_show_genre_2", 2), ("tv_show_genre_3", 2),
                        ("year_of_production", 1), ("director", 1)]:
        value = str(getattr(row, col)).strip().lower()
        if value and value != "nan":
            token = col + ":" + re.sub(r"\s+", "_", value)
            tokens.extend([token] * weight)
    for actor in re.split(r"[,;|]", str(row.actors))[:20]:
        actor = actor.strip().lower()
        if actor and actor != "nan":
            tokens.append("actor:" + re.sub(r"\s+", "_", actor))
    return " ".join(tokens) or "missing_metadata"


def fit_score(method, interactions, metadata, cutoff, n_users, evaluation_users, candidates, config):
    candidates = np.asarray(candidates, dtype=np.int64)
    evaluation_users = np.asarray(evaluation_users, dtype=np.int64)
    t = time.perf_counter()
    popularity = interactions.groupby("item").u.nunique().reindex(candidates, fill_value=0).to_numpy(dtype=np.float32)
    warm = np.isin(evaluation_users, interactions.u.unique())
    details = {"fitting_users_with_interactions": int(interactions.u.nunique()),
               "fitting_unique_user_item_interactions": len(interactions),
               "fitting_interaction_items": int(interactions.item.nunique()),
               "evaluation_warm_users": int(warm.sum()), "evaluation_cold_users": int((~warm).sum()),
               "candidate_items": len(candidates)}
    if method == "popularity":
        fit_seconds = time.perf_counter() - t
        t = time.perf_counter()
        scores = np.broadcast_to(popularity, (len(evaluation_users), len(candidates))).copy()
        details.update({"definition": "Number of distinct fitting users with a retained watched label per programme.",
                        "method_known_candidates": int((popularity > 0).sum())})
    elif method == "content":
        # Metadata and the vocabulary/IDF are fitted on this prefix only.
        prefix = metadata.loc[metadata.s < cutoff].drop_duplicates("item", keep="first").sort_values("item")
        assert len(prefix) and (prefix.s < cutoff).all()
        items = prefix.item.to_numpy(dtype=np.int64)
        docs = [content_document(row) for row in prefix.itertuples(index=False)]
        vectorizer = TfidfVectorizer(lowercase=False, token_pattern=r"(?u)\S+", dtype=np.float32,
                                     norm="l2", smooth_idf=True, sublinear_tf=False)
        features = vectorizer.fit_transform(docs)
        history = interaction_matrix(interactions, n_users, items)
        history.data = np.log1p(history.data / config["watch_threshold_seconds"])
        # Normalizing the weighted sum gives the same direction as its mean.
        profiles = normalize(history @ features, norm="l2", copy=False)
        fit_seconds = time.perf_counter() - t
        t = time.perf_counter()
        ix = pd.Index(items).get_indexer(candidates)
        known = ix >= 0
        scores = np.full((len(evaluation_users), len(candidates)), -np.inf, dtype=np.float32)
        scores[:, known] = (profiles[evaluation_users] @ features[ix[known]].T).toarray()
        scores[~warm] = -np.inf
        details.update({"metadata_items": len(items), "vocabulary_size": len(vectorizer.vocabulary_),
                        "metadata_max_start": int(prefix.s.max()), "metadata_cutoff_exclusive": int(cutoff),
                        "vocabulary_fitted_on_prefix_only": True, "method_known_candidates": int(known.sum()),
                        "profile_weights": "log1p(retained watched seconds / 300); L2-normalized weighted mean of L2 TF-IDF item vectors",
                        "feature_definition": "Title word tokens; field-prefixed category and each genre repeated twice; year/director once; first 20 comma/semicolon/pipe-separated actors once."})
    elif method == "als":
        from implicit.cpu.als import AlternatingLeastSquares
        items = np.sort(interactions.item.unique())
        confidence = interaction_matrix(interactions, n_users, items)
        confidence.data = 1 + config["als_confidence_alpha"] * np.log1p(confidence.data / config["watch_threshold_seconds"])
        model = AlternatingLeastSquares(factors=config["als_factors"], iterations=config["als_iterations"],
                                        regularization=config["als_regularization"], alpha=1.0,
                                        random_state=config["seed"], num_threads=config["als_threads"],
                                        use_cg=False, dtype=np.float32)
        with threadpool_limits(limits=1):
            model.fit(confidence, show_progress=False)
        fit_seconds = time.perf_counter() - t
        t = time.perf_counter()
        ix = pd.Index(items).get_indexer(candidates)
        known = ix >= 0
        scores = np.full((len(evaluation_users), len(candidates)), -np.inf, dtype=np.float32)
        with threadpool_limits(limits=1):
            scores[:, known] = model.user_factors[evaluation_users] @ model.item_factors[ix[known]].T
        scores[~warm] = -np.inf
        details.update({"method_known_candidates": int(known.sum()), "factors": config["als_factors"],
                        "iterations": config["als_iterations"], "regularization": config["als_regularization"],
                        "seed": config["seed"], "threads": config["als_threads"], "solver": "CPU exact alternating least-squares (use_cg=False)",
                        "confidence": "Observed: c_ui = 1 + 10 log1p(retained seconds / 300), p_ui=1. Unobserved: c_ui=1, p_ui=0. implicit alpha=1 (no second scaling).",
                        "objective": "sum_ui c_ui (p_ui - x_u dot y_i)^2 + regularization*(sum_u ||x_u||^2 + sum_i ||y_i||^2)",
                        "observed_confidence_min": float(confidence.data.min()),
                        "observed_confidence_max": float(confidence.data.max())})
    else:
        raise ValueError(method)
    predictions, fallback = [], []
    for row in scores:
        ids, used = top_five(candidates, row, popularity)
        predictions.append(ids)
        fallback.append(used)
    predict_seconds = time.perf_counter() - t
    fallback = np.asarray(fallback, dtype=bool)
    details.update({"method_slots_fraction": float((~fallback).mean()),
                    "users_with_five_method_scored_items": int((~fallback).all(axis=1).sum()),
                    "users_needing_any_popularity_backfill": int(fallback.any(axis=1).sum()),
                    "users_needing_full_popularity_backfill": int(fallback.all(axis=1).sum()),
                    "distinct_recommended_items": len({x for row in predictions for x in row})})
    return {"method": method, "predictions": predictions, "scores": scores, "fallback": fallback,
            "popularity": popularity, "fit_seconds": fit_seconds, "inference_seconds": predict_seconds, "details": details}


def reciprocal_rank_fusion(results, candidates, weights=None, constant=60):
    """Optional fusion of completed individual methods; no additional fitting."""
    t = time.perf_counter()
    candidates = np.asarray(candidates)
    weights = weights or {name: 1.0 for name in results}
    fusion = np.zeros_like(next(iter(results.values()))["scores"])
    for name, result in results.items():
        weight = weights.get(name, 0)
        if not weight:
            continue
        for row_number, row in enumerate(result["scores"]):
            valid = np.flatnonzero(np.isfinite(row) & (row > 0))
            order = valid[np.lexsort((candidates[valid], -row[valid]))]
            fusion[row_number, order] += weight / (constant + np.arange(1, len(order) + 1))
    popularity = next(iter(results.values()))["popularity"]
    predictions, fallback = [], []
    for row in fusion:
        ids, used = top_five(candidates, row, popularity)
        predictions.append(ids)
        fallback.append(used)
    fallback = np.asarray(fallback, dtype=bool)
    return {"method": "hybrid", "predictions": predictions, "scores": fusion, "fallback": fallback,
            "popularity": popularity, "fit_seconds": sum(r["fit_seconds"] for n, r in results.items() if weights.get(n, 0)),
            "inference_seconds": sum(r["inference_seconds"] for n, r in results.items() if weights.get(n, 0)) + time.perf_counter() - t,
            "details": {"definition": "Weighted reciprocal rank fusion of positive method scores", "weights": weights, "constant": constant,
                        "method_slots_fraction": float((~fallback).mean()),
                        "users_needing_any_popularity_backfill": int(fallback.any(axis=1).sum()),
                        "distinct_recommended_items": len({x for row in predictions for x in row})}}
