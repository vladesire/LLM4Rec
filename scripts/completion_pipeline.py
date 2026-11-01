"""Session-aware completion events; corrected late-audit targets stay sealed."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import match_channel
from data_pipeline import FILES, canonical_schedule, dump, read_schedule, seconds, sha256


def epoch(value):
    return int(pd.Timestamp(value).to_datetime64().astype('datetime64[s]').astype(np.int64))


def build_sessions(raw):
    """Consecutive same-channel union, interrupted at any recorded channel change."""
    d = raw.drop_duplicates(['u', 'ch', 's', 'e']).sort_values(['u', 's', 'e', 'ch', 'raw_id']).copy()
    d = d.loc[d.e > d.s].copy()
    run_start = d.u.ne(d.u.shift()) | d.ch.ne(d.ch.shift())
    run = run_start.to_numpy().cumsum()
    previous_max = d.groupby(run, sort=False).e.cummax().shift()
    session_start = run_start | d.s.gt(previous_max)
    d['session'] = session_start.to_numpy().cumsum() - 1
    sessions = d.groupby('session', sort=False, as_index=False).agg(
        u=('u', 'first'), ch=('ch', 'first'), s=('s', 'min'), e=('e', 'max'),
        first_raw_id=('raw_id', 'min'), raw_fragment_count=('raw_id', 'size'))
    original_end = sessions.e.copy()
    next_start = sessions.s.shift(-1)
    switch = sessions.u.eq(sessions.u.shift(-1)) & sessions.ch.ne(sessions.ch.shift(-1))
    clip = switch & sessions.e.gt(next_start)
    sessions.loc[clip, 'e'] = next_start.loc[clip].astype(np.int64)
    stats = {'raw_rows': len(raw), 'exact_duplicates_removed': len(raw) - len(raw.drop_duplicates(['u', 'ch', 's', 'e'])),
             'same_channel_fragments_unioned': len(d) - len(sessions),
             'sessions_clipped_at_overlapping_channel_change': int(clip.sum()),
             'seconds_clipped_at_channel_changes': int((original_end - sessions.e).sum()),
             'zero_length_sessions_removed': int((sessions.e <= sessions.s).sum()),
             'same_start_cross_channel_changes': int((switch & sessions.s.eq(next_start)).sum()),
             'interpretation': 'Single chronological stream per user; no device identifier. A later different-channel observation interrupts the earlier channel session.'}
    sessions = sessions.loc[sessions.e > sessions.s].copy()
    sessions[['session', 'u', 'ch', 's', 'e']] = sessions[['session', 'u', 'ch', 's', 'e']].astype(np.int64)
    previous = sessions.e.shift().where(sessions.u.eq(sessions.u.shift()), -1)
    assert (sessions.s >= previous).all(), 'Continuous-session timeline must not double count user time'
    stats['retained_sessions'] = len(sessions)
    stats['retained_session_seconds'] = int((sessions.e - sessions.s).sum())
    return sessions, d, stats


def join_events(sessions, schedule):
    frames = []
    matched_seconds, labelled_seconds = 0, 0
    groups = {int(ch): sc for ch, sc in schedule.groupby('ch', sort=False)}
    for ch, sv in sessions.groupby('ch', sort=False):
        if ch not in groups:
            continue
        sc = groups[ch]
        vi, ai, overlap = match_channel(sv.s.to_numpy(), sv.e.to_numpy(), sc.s.to_numpy(), sc.e.to_numpy())
        if not len(vi):
            continue
        matched_seconds += int(overlap.sum())
        keep = sc.item.to_numpy()[ai] > 0
        vi, ai, overlap = vi[keep], ai[keep], overlap[keep]
        labelled_seconds += int(overlap.sum())
        if not len(vi):
            continue
        frames.append(pd.DataFrame({
            'u': sv.u.to_numpy()[vi].astype(np.int32), 'item': sc.item.to_numpy()[ai],
            'ch': np.full(len(vi), ch, dtype=np.int32), 'session': sv.session.to_numpy()[vi],
            'airing': sc.airing.to_numpy()[ai],
            'overlap_start': np.maximum(sv.s.to_numpy()[vi], sc.s.to_numpy()[ai]),
            'overlap_end': np.minimum(sv.e.to_numpy()[vi], sc.e.to_numpy()[ai]),
            'original_duration': (sc.e.to_numpy()[ai] - sc.s.to_numpy()[ai]).astype(np.int32)}))
    events = pd.concat(frames, ignore_index=True)
    events['overlap_seconds'] = events.overlap_end - events.overlap_start
    events['completion_fraction'] = events.overlap_seconds / events.original_duration
    assert (events.item > 0).all() and (events.completion_fraction <= 1).all()
    return events, {'labelled_session_airing_intersections': len(events), 'schedule_matched_seconds': matched_seconds,
                    'labelled_matched_seconds': labelled_seconds,
                    'labelled_seconds_fraction_of_sessions': labelled_seconds / int((sessions.e-sessions.s).sum())}


def clip_events(events, lower, upper):
    clipped = events.loc[(events.overlap_start < upper) & (events.overlap_end > lower)].copy()
    clipped['overlap_start'] = np.maximum(clipped.overlap_start, lower)
    clipped['overlap_end'] = np.minimum(clipped.overlap_end, upper)
    clipped['overlap_seconds'] = clipped.overlap_end - clipped.overlap_start
    clipped['completion_fraction'] = clipped.overlap_seconds / clipped.original_duration
    assert (clipped.overlap_end <= upper).all() and (clipped.overlap_start >= lower).all()
    return clipped


def qualifying_interactions(events, lower, upper, threshold=.8, aggregate=False):
    clipped = clip_events(events, lower, upper)
    if aggregate:
        # Session stream is disjoint. Summing per airing therefore unions sitting time.
        broadcast = clipped.groupby(['u', 'item', 'airing'], as_index=False).agg(
            overlap_seconds=('overlap_seconds', 'sum'), original_duration=('original_duration', 'first'),
            last_end=('overlap_end', 'max'))
        assert (broadcast.overlap_seconds <= broadcast.original_duration).all()
        kept = broadcast.loc[broadcast.overlap_seconds >= threshold * broadcast.original_duration].copy()
    else:
        kept = clipped.loc[clipped.overlap_seconds >= threshold * clipped.original_duration].copy()
        kept = kept.sort_values(['u', 'airing', 'overlap_seconds', 'overlap_end'], ascending=[True, True, False, False])
        kept = kept.drop_duplicates(['u', 'airing'])
        kept = kept.rename(columns={'overlap_end': 'last_end'})
    grouped = kept.groupby(['u', 'item'], as_index=False, sort=True).agg(
        count=('airing', 'nunique'), qualified_seconds=('overlap_seconds', 'sum'), last_end=('last_end', 'max'))
    stats = {'lower': lower, 'upper': upper, 'threshold': threshold, 'aggregate_across_sittings': aggregate,
             'clipped_intersections': len(clipped), 'qualifying_distinct_user_broadcasts': len(kept),
             'user_programme_pairs': len(grouped), 'users': int(grouped.u.nunique()),
             'items': int(grouped.item.nunique()), 'max_observation_end': int(clipped.overlap_end.max()) if len(clipped) else None,
             'max_qualifying_observation_end': int(grouped.last_end.max()) if len(grouped) else None}
    return grouped, stats


def target_sets(interactions, descending_id=False):
    ordered = interactions.sort_values(['u', 'count', 'item'], ascending=[True, False, not descending_id])
    top = ordered.groupby('u', sort=False).head(5)
    return {int(u): set(g.item.tolist()) for u, g in top.groupby('u')}, top


def prepare_completion():
    start = time.perf_counter()
    cache = Path('cache/completion')
    cache.mkdir(parents=True, exist_ok=True)
    r0 = pd.read_pickle('cache/prepared.pkl')
    fingerprints = {name: sha256(Path('data')/name) for name in FILES}
    for name, digest in fingerprints.items():
        assert digest == r0['audit']['fingerprints'][name]['sha256'], 'Raw input differs from preserved R0'
    signature = {'inputs': fingerprints, 'code': sha256(__file__), 'join_code': sha256(Path(__file__).with_name('core.py'))}
    manifest = cache/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) == signature and (cache/'events.pkl').exists():
        out = pd.read_pickle(cache/'prepared.pkl')
        out['events'] = pd.read_pickle(cache/'events.pkl')
        out['preparation_cache_used'] = True
        return out
    users = r0['users']
    mapping = {uid:i for i,uid in enumerate(users)}
    chunks, offset = [], 0
    for c in pd.read_csv('data/dataset11-30.csv', dtype={'user_id':str, 'vsetv_id':'int32'}, chunksize=300000):
        assert c.user_id.isin(mapping).all()
        chunks.append(pd.DataFrame({'u':c.user_id.map(mapping).to_numpy(dtype=np.int32),
                                    'ch':c.vsetv_id.to_numpy(dtype=np.int32),
                                    's':seconds(c.start_time, '%Y-%m-%d %H:%M:%S'),
                                    'e':seconds(c.stop_time, '%Y-%m-%d %H:%M:%S'),
                                    'raw_id':np.arange(offset,offset+len(c),dtype=np.int64)}))
        offset += len(c)
    raw = pd.concat(chunks, ignore_index=True)
    sessions, provenance, session_audit = build_sessions(raw)
    numeric, metadata, schedule_stats = [], [], {}
    for name in FILES[1:3]:
        n,m,a = read_schedule(Path('data')/name)
        n['source_file_index'] = FILES[1:3].index(name)
        n['source_record'] = np.arange(len(n), dtype=np.int64)
        numeric.append(n); metadata.append(m); schedule_stats[name] = a
    schedule_all = pd.concat(numeric, ignore_index=True)
    schedule, canonical_audit = canonical_schedule(schedule_all)
    future, _, future_audit = read_schedule(Path('data')/FILES[3], metadata=False)
    cutoff = epoch('2020-07-27')
    assert int(future.s.min()) == cutoff
    assert int(raw.e.max()) < cutoff, 'Future viewing detected'
    # Keep original full schedule duration even for an airing crossing a cutoff.
    events, join_audit = join_events(sessions, schedule)
    meta = pd.concat(metadata, ignore_index=True).sort_values(['s','item'], kind='stable')
    meta = meta.loc[meta.s < cutoff]
    audit = {'input_sha256': fingerprints, 'sessionization':session_audit, 'join':join_audit,
             'canonical_schedule':canonical_audit, 'schedules':schedule_stats,
             'future_schedule':future_audit, 'future_metadata_read':False,
             'max_raw_observation_end':int(raw.e.max()), 'no_test_viewing_used':True,
             'original_duration_preserved':True, 'development_start':epoch('2020-06-01'),
             'development_end':epoch('2020-06-29'), 'production_cutoff':cutoff,
             'corrected_late_audit_targets_constructed':False,
             'corrected_late_audit_metrics_inspected':False,
             'preparation_seconds':time.perf_counter()-start}
    out = {'users':users, 'evaluation_users':r0['evaluation_users'], 'template':r0['template'],
           'metadata':meta, 'schedule':schedule, 'availability':schedule_all[['ch','s','e','item']],
           'test_candidates':r0['test_candidates'], 'events':events, 'raw':provenance,
           'sessions':sessions, 'audit':audit, 'preparation_cache_used':False}
    # Raw and event caches stay outside git. No corrected late targets are made.
    pd.to_pickle(events, cache/'events.pkl')
    pd.to_pickle({k:v for k,v in out.items() if k!='events'}, cache/'prepared.pkl')
    dump(manifest, signature)
    return out


def legacy_broad_prefix(raw, schedule, cutoff):
    """Replay R0's actual session-union and >=300-second feature policy."""
    from data_pipeline import labels, match_window
    d = raw[['u','ch','s','e']].drop_duplicates().sort_values(['u','ch','s','e']).copy()
    previous = d.groupby(['u','ch'], sort=False).e.cummax().shift()
    new = d.u.ne(d.u.shift()) | d.ch.ne(d.ch.shift()) | d.s.gt(previous)
    d = d.groupby(new.to_numpy().cumsum(), sort=False).agg(u=('u','first'),ch=('ch','first'),s=('s','first'),e=('e','max')).reset_index(drop=True)
    events, audit = match_window(d, schedule, int(d.s.min()), cutoff, 'R0 strategy development prefix replay')
    interactions, label_audit = labels(events, 300)
    return interactions, {'join':audit, 'labels':label_audit}
