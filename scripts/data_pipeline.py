"""Inspect genuine CSVs, canonicalize intervals, and cache temporal labels."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import match_channel

FILES = ["dataset11-30.csv", "export_arh_11-20-final.csv",
         "export_arh_21-30-final.csv", "export_arh_31-42-final.csv", "submission.csv"]
META = ["tv_show_title", "tv_show_category", "tv_show_genre_1", "tv_show_genre_2",
        "tv_show_genre_3", "year_of_production", "director", "actors"]


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seconds(series, fmt):
    return pd.to_datetime(series, format=fmt, errors="raise").to_numpy(dtype="datetime64[s]").astype(np.int64)


def human(value):
    return str(pd.to_datetime(int(value), unit="s"))


def quantiles(values):
    return {str(k): float(v) for k, v in pd.Series(values).quantile([0, .01, .1, .25, .5, .75, .9, .99, 1]).items()}


def read_schedule(path, metadata=True):
    cols = ["channel_id", "start_time", "duration", "tv_show_id"] + (META if metadata else [])
    d = pd.read_csv(path, usecols=cols, keep_default_na=False, encoding="utf-8-sig")
    d = d.rename(columns={"channel_id": "ch", "tv_show_id": "item"})
    d["s"] = seconds(d.start_time, "%d.%m.%Y %H:%M:%S")
    d["e"] = d.s + d.duration.astype(np.int64)
    d["ch"] = d.ch.astype(np.int32)
    d["item"] = d.item.astype(np.int64)
    assert (d.item >= 0).all()
    stats = {"rows": len(d), "nonzero_rows": int((d.item > 0).sum()),
             "nonzero_fraction": float((d.item > 0).mean()),
             "distinct_nonzero_items": int(d.loc[d.item > 0, "item"].nunique()),
             "min_start": human(d.s.min()), "max_start": human(d.s.max()), "max_end": human(d.e.max()),
             "duration_seconds_quantiles": quantiles(d.duration),
             "nonpositive_duration_rows": int((d.e <= d.s).sum()), "channels": sorted(d.ch.unique().tolist()),
             "delimiter": ",", "encoding": "utf-8-sig", "timestamp_format": "%d.%m.%Y %H:%M:%S"}
    numeric = d[["ch", "s", "e", "item"]].copy()
    meta = d.loc[(d.item > 0) & (d.e > d.s), ["item", "s", "e"] + META].copy() if metadata else None
    return numeric, meta, stats


def canonical_schedule(schedule):
    """Remove duplicates, nonpositive rows, and all overlapping airing rows.

    Dropping both participants is deliberately conservative: no seconds in a
    conflicting airing are assigned. Actual exclusions and IDs are audited.
    """
    duplicate = schedule.duplicated(["ch", "s", "e", "item"])
    invalid = schedule.e <= schedule.s
    d = schedule.loc[~duplicate & ~invalid].sort_values(["ch", "s", "e", "item"]).copy()
    previous_max = d.groupby("ch", sort=False).e.cummax()
    previous_max = previous_max.shift().where(d.ch.eq(d.ch.shift()), -1)
    next_start = d.s.shift(-1).where(d.ch.eq(d.ch.shift(-1)), np.inf)
    ambiguous = (d.s < previous_max) | (d.e > next_start)
    stats = {"duplicate_rows_removed": int(duplicate.sum()), "nonpositive_rows_removed": int(invalid.sum()),
             "overlapping_airings_removed": int(ambiguous.sum()),
             "labelled_overlapping_airings_removed": int(((d.item > 0) & ambiguous).sum()),
             "overlap_examples": [{**r, "start": human(r["s"]), "end": human(r["e"])}
                                  for r in d.loc[ambiguous].head(12).to_dict("records")]}
    d = d.loc[~ambiguous].reset_index(drop=True)
    d["airing"] = np.arange(len(d), dtype=np.int64)
    return d, stats


def read_views(path, chunk_size):
    chunks, user_to_code, n, mismatch, mismatch_examples = [], {}, 0, 0, []
    all_durations, channel_ids = [], set()
    for c in pd.read_csv(path, dtype={"user_id": str, "vsetv_id": "int32"}, chunksize=chunk_size, encoding="utf-8-sig"):
        for uid in c.user_id.unique():
            if uid not in user_to_code:
                user_to_code[uid] = len(user_to_code)
        s, e = seconds(c.start_time, "%Y-%m-%d %H:%M:%S"), seconds(c.stop_time, "%Y-%m-%d %H:%M:%S")
        difference = e - s
        bad = difference != c.duraton.to_numpy()
        mismatch += int(bad.sum())
        if len(mismatch_examples) < 5:
            mismatch_examples.extend(c.loc[bad].head(5 - len(mismatch_examples)).to_dict("records"))
        channel_ids.update(c.vsetv_id.unique().tolist())
        all_durations.append(difference)
        chunks.append(pd.DataFrame({"u": c.user_id.map(user_to_code).to_numpy(dtype=np.int32),
                                    "ch": c.vsetv_id.to_numpy(dtype=np.int32), "s": s, "e": e}))
        n += len(c)
    d = pd.concat(chunks, ignore_index=True)
    del chunks
    invalid = d.e <= d.s
    duplicates = d.duplicated(["u", "ch", "s", "e"])
    stats = {"raw_rows": n, "raw_users": len(user_to_code), "channels": sorted(channel_ids),
             "min_start": human(d.s.min()), "max_stop": human(d.e.max()),
             "duraton_timestamp_mismatch_rows": mismatch, "mismatch_examples": mismatch_examples,
             "timestamp_interval_seconds_quantiles": quantiles(np.concatenate(all_durations)),
             "invalid_rows_removed": int(invalid.sum()), "exact_duplicate_intervals_removed": int(duplicates.sum()),
             "delimiter": ",", "encoding": "utf-8-sig", "timestamp_format": "%Y-%m-%d %H:%M:%S",
             "duration_policy": "Use stop_time - start_time; duraton is viewing duration, never programme length."}
    d = d.loc[~invalid & ~duplicates].sort_values(["u", "ch", "s", "e"]).reset_index(drop=True)
    before = len(d)
    # Union repeated/overlapping visits per user and channel before the join.
    previous = d.groupby(["u", "ch"], sort=False).e.cummax().shift()
    new_group = d.u.ne(d.u.shift()) | d.ch.ne(d.ch.shift())
    starts_group = new_group | d.s.gt(previous)
    group = starts_group.to_numpy().cumsum()
    d = d.groupby(group, sort=False).agg(u=("u", "first"), ch=("ch", "first"), s=("s", "first"), e=("e", "max")).reset_index(drop=True)
    stats.update({"overlapping_or_adjacent_intervals_merged": before - len(d), "union_intervals": len(d),
                  "union_seconds": int((d.e - d.s).sum()),
                  "union_policy": "Within each user/channel, take the union of overlapping or adjacent intervals."})
    users = [None] * len(user_to_code)
    for uid, code in user_to_code.items():
        users[code] = uid
    return d, users, stats


def match_window(views, schedule, lower, upper, name):
    v = views.loc[(views.s < upper) & (views.e > lower)].copy()
    v["s"] = np.maximum(v.s, lower)
    v["e"] = np.minimum(v.e, upper)
    assert (v.s >= lower).all() and (v.e <= upper).all()
    groups = {int(ch): g for ch, g in schedule.groupby("ch", sort=False)}
    matched_rows = np.zeros(len(v), dtype=bool)
    labelled_rows = np.zeros(len(v), dtype=bool)
    v = v.reset_index(drop=True)
    frames, examples = [], []
    total_schedule_seconds = 0
    labelled_seconds = 0
    for ch, cv in v.groupby("ch", sort=False):
        if ch not in groups:
            continue
        sc = groups[ch]
        vi, ai, amount = match_channel(cv.s.to_numpy(), cv.e.to_numpy(), sc.s.to_numpy(), sc.e.to_numpy())
        if not len(vi):
            continue
        matched_rows[cv.index.to_numpy()[vi]] = True
        total_schedule_seconds += int(amount.sum())
        keep = sc.item.to_numpy()[ai] > 0
        labelled_rows[cv.index.to_numpy()[vi[keep]]] = True
        labelled_seconds += int(amount[keep].sum())
        if keep.any():
            frame = pd.DataFrame({"u": cv.u.to_numpy()[vi[keep]], "item": sc.item.to_numpy()[ai[keep]],
                                  "airing": sc.airing.to_numpy()[ai[keep]], "seconds": amount[keep]})
            frames.append(frame)
        if len(examples) < 4:
            for j in np.flatnonzero(keep)[:2]:
                vr, sr = cv.iloc[int(vi[j])], sc.iloc[int(ai[j])]
                examples.append({"u": int(vr.u), "channel": int(ch), "item": int(sr["item"]),
                                 "view_start": human(vr.s), "view_end": human(vr.e),
                                 "airing_start": human(sr.s), "airing_end": human(sr.e), "overlap_seconds": int(amount[j])})
                if len(examples) == 4:
                    break
    events = pd.concat(frames, ignore_index=True).groupby(["u", "item", "airing"], sort=True, as_index=False).seconds.sum()
    total = int((v.e - v.s).sum())
    stats = {"name": name, "lower_inclusive": human(lower), "upper_exclusive": human(upper),
             "clipped_view_intervals": len(v), "view_seconds": total,
             "no_schedule_overlap_intervals": int((~matched_rows).sum()),
             "no_labelled_overlap_intervals": int((~labelled_rows).sum()),
             "any_schedule_matched_seconds": total_schedule_seconds,
             "labelled_matched_seconds": labelled_seconds,
             "unlabelled_matched_seconds": total_schedule_seconds - labelled_seconds,
             "unmatched_seconds": total - total_schedule_seconds,
             "labelled_seconds_fraction": labelled_seconds / total if total else 0,
             "user_airings_before_threshold": len(events),
             "max_clipped_observation_end": human(v.e.max()),
             "min_clipped_observation_start": human(v.s.min()), "examples": examples,
             "assertions": {"observations_within_window": True, "no_zero_items_in_events": bool((events.item > 0).all())}}
    assert total_schedule_seconds <= total
    return events, stats


def labels(events, threshold):
    kept = events.loc[events.seconds >= threshold]
    interactions = kept.groupby(["u", "item"], as_index=False, sort=True).seconds.sum()
    stats = {"user_airings_before_threshold": len(events), "user_airings_dropped_below_threshold": len(events) - len(kept),
             "retained_user_airings": len(kept), "unique_user_item_interactions": len(interactions),
             "users_with_labels": int(interactions.u.nunique()), "items_with_labels": int(interactions.item.nunique()),
             "retained_viewing_seconds": int(kept.seconds.sum()), "threshold_seconds": threshold,
             "label_policy": "At least threshold seconds per user-airing after union and clipping; sum retained airings by user-item."}
    return interactions, stats


def prepare(data_dir, cache_dir, output_dir, config, rebuild=False):
    started = time.perf_counter()
    data_dir, cache_dir, output_dir = Path(data_dir), Path(cache_dir), Path(output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprints = {name: {"bytes": (data_dir / name).stat().st_size, "sha256": sha256(data_dir / name)} for name in FILES}
    signature = {"inputs": fingerprints, "validation_days": config["validation_days"],
                 "threshold": config["watch_threshold_seconds"], "preprocessing_code": sha256(__file__),
                 "join_code": sha256(Path(__file__).with_name("core.py"))}
    cache_file = cache_dir / "prepared.pkl"
    if not rebuild and cache_file.exists() and json.loads((cache_dir / "manifest.json").read_text()) == signature:
        data = pd.read_pickle(cache_file)  # Only the cache produced locally by this script.
        data["audit"]["cache_used"] = True
        data["audit"]["current_cache_load_and_fingerprint_seconds"] = time.perf_counter() - started
        dump(output_dir / "data_audit.json", data["audit"])
        print("Loaded matching local preprocessing cache", flush=True)
        return data
    audit = {"fingerprints": fingerprints, "cache_used": False, "stage_seconds": {},
             "time_convention": "Naive dataset wall-clock strings, parsed explicitly and compared on the same axis; no timezone inferred.",
             "channel_policy": "Direct equality vsetv_id == channel_id, supported by observed ID intersection; no remapping invented."}
    numeric, metadata, schedule_audits = [], [], {}
    t = time.perf_counter()
    for name in FILES[1:3]:
        n, m, a = read_schedule(data_dir / name)
        numeric.append(n)
        metadata.append(m)
        schedule_audits[name] = a
    future, _, a = read_schedule(data_dir / FILES[3], metadata=False)
    schedule_audits[FILES[3]] = a
    test_start, test_end = int(future.s.min()), int(future.e.max())
    cutoff = test_start - config["validation_days"] * 86400
    schedule, can_audit = canonical_schedule(pd.concat(numeric, ignore_index=True))
    schedule = schedule.loc[schedule.s < test_start].copy()
    schedule["e"] = np.minimum(schedule.e, test_start)
    schedule = schedule.loc[schedule.e > schedule.s]
    metadata = pd.concat(metadata, ignore_index=True).sort_values(["s", "item"], kind="stable")
    metadata = metadata.loc[metadata.s < test_start]
    audit["schedules"], audit["canonical_schedule"] = schedule_audits, can_audit
    audit["split"] = {"fit_start_inclusive": human(schedule.s.min()), "fit_cutoff_exclusive": human(cutoff),
                      "validation_end_exclusive": human(test_start), "test_start_inclusive": human(test_start),
                      "test_end_exclusive": human(test_end), "validation_days": config["validation_days"],
                      "rationale": "A substantial late four-week window retains an earlier fitting prefix. It is shorter than the twelve-week future schedule, so horizon transfer is a limitation."}
    audit["stage_seconds"]["schedule_read_and_canonicalize"] = time.perf_counter() - t
    print("Schedules inspected; cutoff", human(cutoff), "test starts", human(test_start), flush=True)
    t = time.perf_counter()
    views, users, view_audit = read_views(data_dir / FILES[0], config["chunk_size"])
    view_audit["channels_missing_in_training_schedule"] = sorted(set(view_audit["channels"]) - set(schedule.ch.unique().tolist()))
    view_audit["view_intervals_starting_in_test"] = int((views.s >= test_start).sum())
    view_audit["view_seconds_clipped_at_test_start"] = int(np.maximum(views.e - np.maximum(views.s, test_start), 0).sum())
    audit["viewing"] = view_audit
    audit["stage_seconds"]["view_read_dedup_and_union"] = time.perf_counter() - t
    lower = min(int(views.s.min()), int(schedule.s.min()))
    t = time.perf_counter()
    fit_events, fit_audit = match_window(views, schedule, lower, cutoff, "fitting prefix")
    validation_events, val_audit = match_window(views, schedule, cutoff, test_start, "validation")
    audit["stage_seconds"]["interval_join"] = time.perf_counter() - t
    threshold = config["watch_threshold_seconds"]
    train, fit_label_audit = labels(fit_events, threshold)
    val, val_label_audit = labels(validation_events, threshold)
    all_events = pd.concat([fit_events, validation_events]).groupby(["u", "item", "airing"], as_index=False).seconds.sum()
    production, prod_label_audit = labels(all_events, threshold)
    audit.update({"fit_join": fit_audit, "validation_join": val_audit,
                  "fit_labels": fit_label_audit, "validation_labels": val_label_audit,
                  "production_labels": prod_label_audit})
    validation_candidates = sorted(schedule.loc[(schedule.e > cutoff) & (schedule.s < test_start) & (schedule.item > 0), "item"].unique().tolist())
    test_candidates = sorted(future.loc[(future.e > test_start) & (future.e > future.s) & (future.item > 0), "item"].unique().tolist())
    template = pd.read_csv(data_dir / FILES[4], dtype=str, keep_default_na=False)
    assert list(template.columns) == ["user_id", "tv_show_id"] and template.user_id.is_unique
    uid_lookup = {uid: i for i, uid in enumerate(users)}
    evaluation_users = [uid_lookup.get(uid, -1) for uid in template.user_id]
    # Unknown official users retain unique codes, and receive popularity fallback.
    for j, code in enumerate(evaluation_users):
        if code == -1:
            evaluation_users[j] = len(users)
            users.append(template.user_id.iloc[j])
    audit["template"] = {"rows": len(template), "columns": list(template.columns),
                         "user_id_dtype": "string (preserves unsigned 64-bit identifiers exactly)",
                         "users_with_raw_history": sum(uid in uid_lookup for uid in template.user_id)}
    audit["candidate_policy"] = {"validation_items": len(validation_candidates), "test_items": len(test_candidates),
                                 "nonzero_scheduled_in_target_window_only": True, "allow_repeats": True,
                                 "fallback": "Distinct-user popularity from fitting labels, restricted to target candidates; ascending integer ID breaks ties, including zero-popularity ties."}
    audit["learned_preprocessing"] = {"validation_metadata_filter": "airing start < fitting cutoff; first available row per item",
                                      "production_metadata_filter": "airing start < test start; first available row per item",
                                      "future_metadata_features_read_by_pipeline": False,
                                      "no_viewing_at_or_after_test_start_used": True}
    assert set(val.item).issubset(set(validation_candidates))
    audit["stage_seconds"]["total_preparation_including_fingerprints"] = time.perf_counter() - started
    data = {"train": train, "val": val, "production": production, "metadata": metadata,
            "users": users, "evaluation_users": evaluation_users, "template": template,
            "validation_candidates": validation_candidates, "test_candidates": test_candidates,
            "cutoff": cutoff, "test_start": test_start, "test_end": test_end, "audit": audit}
    pd.to_pickle(data, cache_file)
    dump(cache_dir / "manifest.json", signature)
    dump(output_dir / "data_audit.json", audit)
    print("Prepared labels", {"fit": len(train), "validation": len(val), "production": len(production)}, flush=True)
    return data
