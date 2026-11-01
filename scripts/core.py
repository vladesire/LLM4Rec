"""Small, independently checkable interval and ranking primitives."""
from __future__ import annotations

import numpy as np


def overlap_seconds(view_start, view_end, airing_start, airing_end):
    return max(0, min(view_end, airing_end) - max(view_start, airing_start))


def clip_interval(start, end, lower, upper):
    return max(start, lower), min(end, upper)


def match_channel(view_starts, view_ends, airing_starts, airing_ends):
    """Vectorized interval join on a disjoint, sorted channel timeline.

    Each viewing row starts at its first possibly overlapping airing. A vector
    of active viewing rows advances through airings until its viewing end.
    A long viewing interval therefore matches every broadcast it intersects.
    Returns viewing indices, airing indices and positive overlap seconds.
    """
    vs = np.asarray(view_starts, dtype=np.int64)
    ve = np.asarray(view_ends, dtype=np.int64)
    ss = np.asarray(airing_starts, dtype=np.int64)
    se = np.asarray(airing_ends, dtype=np.int64)
    if not len(ss) or not len(vs):
        return (np.empty(0, dtype=np.int64),) * 3
    assert np.all(ss < se)
    assert np.all(se[:-1] <= ss[1:]), "Canonical timeline must be disjoint"
    pos = np.searchsorted(se, vs, side="right")
    active = np.flatnonzero((pos < len(ss)) & (ve > vs))
    vi, ai, seconds = [], [], []
    while len(active):
        ix = pos[active]
        active = active[ss[ix] < ve[active]]
        if not len(active):
            break
        ix = pos[active]
        amount = np.minimum(ve[active], se[ix]) - np.maximum(vs[active], ss[ix])
        keep = amount > 0
        vi.append(active[keep])
        ai.append(ix[keep])
        seconds.append(amount[keep])
        pos[active] += 1
        active = active[pos[active] < len(ss)]
    if not vi:
        return (np.empty(0, dtype=np.int64),) * 3
    return tuple(np.concatenate(x) for x in (vi, ai, seconds))


def ap_at_5(relevant, recommendations):
    """AP@5; empty relevance contributes zero. Duplicate predictions rejected."""
    relevant = set(relevant)
    ranked = list(recommendations)[:5]
    if len(set(ranked)) != len(ranked):
        raise ValueError("Duplicate recommendations are not valid output")
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for rank, item in enumerate(ranked, 1):
        if item in relevant:
            hits += 1
            total += hits / rank
    return total / min(5, len(relevant))


def top_five(candidates, scores, popularity):
    """Positive finite method scores, then common eligible-popularity fallback.

    Candidates are unique integer IDs; all ties resolve by ascending item ID.
    Returns IDs and whether each slot came from the shared fallback.
    """
    candidates = np.asarray(candidates, dtype=np.int64)
    scores = np.asarray(scores, dtype=float)
    popularity = np.asarray(popularity, dtype=float)
    assert len(candidates) >= 5 and len(set(candidates)) == len(candidates)
    assert len(scores) == len(popularity) == len(candidates)
    valid = np.flatnonzero(np.isfinite(scores) & (scores > 0))
    ordered = valid[np.lexsort((candidates[valid], -scores[valid]))]
    chosen = ordered[:5].tolist()
    fallback = [False] * len(chosen)
    for ix in np.lexsort((candidates, -popularity)):
        if len(chosen) == 5:
            break
        if int(ix) not in chosen:
            chosen.append(int(ix))
            fallback.append(True)
    return candidates[chosen].tolist(), fallback
