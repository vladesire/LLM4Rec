# Completion/frequency proxy contract, version 1 (established in R1)

The competition description supplied by the student asks for five preferred programmes based on subsequent twelve-week viewing. The public competition page was attempted again before R1 but was not readable through the web tool. The supplied wording does not specify the exact hidden preference calculation.

The historical participant [article](https://medium.com/data-science/two-layered-recommender-system-methodology-a-prize-winning-solution-c4bdb815b156) describes completion without a channel change. Its [get_target implementation](https://github.com/ndmel/2nd_place_recsys_cinema_challenge_2020/blob/main/utils.py) uses completion >=0.8 and per-user programme frequency, then takes five items. These are attributed participant methods, not independent proof of the hidden evaluator. R1 reimplements the ideas locally; no external predictions, reports, preprocessed data or third-party code are imported. The article's own personal baseline uses a lower training threshold; R1 follows the student's specified 80% training threshold instead.

## Primary rule

1. Read the genuine CSVs and preserve exact IDs. Use the original dataset wall-clock representations; no guessed timezone shift.
2. Remove exact duplicate views. Sort each user's records chronologically. Merge overlapping/touching same-channel observations only inside a consecutive same-channel run. A gap or a different channel starts another session. At an overlapping channel change, the previous session ends at the next session's start. Same-time competing channels use deterministic end/channel/raw-row order. This is a conservative single-stream interpretation; there is no device ID to disambiguate simultaneous devices. Audit the resulting clipping instead of claiming device identity.
3. Match every session to all intersected broadcasts with the checked channel-wise interval join. Retain raw-row provenance, session ID, airing ID, original airing start/end/duration and clipped overlap endpoints. Remove invalid/ambiguous schedule rows using R0's recorded conservative policy.
4. Clip overlap to each fitting/target time window, retaining the **original full broadcast duration** as denominator. A qualifying watch has at least 80% in one session. Count each user/broadcast only once even if duplicate fragments or sessions would qualify it again. Multiple airings with the same programme ID are distinct frequency events.
5. For evaluation, rank programmes by qualifying broadcast count descending, then integer programme ID ascending; keep up to five. Do not backfill or filter relevance by retrievability. Empty relevance remains in the primary population with AP=0. An alternative ID-descending tie and aggregate-across-sittings completion are sensitivity checks, never silent replacements.

## Development, audit and production

- Development features: observations strictly before 1 June 2020 00:00; target: [1 June,29 June 2020 00:00).
- Later corrected-label audit: [29 June,27 July 2020 00:00), reserved until R4. It was already examined with old broad labels, so it is not pristine. R1 does not construct its corrected targets, evaluate it, or output its corrected metrics. Its history may enter the explicitly separate production refit, as authorized.
- Production features: all eligible observations before 27 July 2020 00:00; candidates from the supplied future schedule only. No future viewing is available or used.
- All populations use the same 1,260 template users; separately report nonempty and at-least-five-target cohorts. These population conventions are not claimed to match the hidden scorer.
- Candidate IDs are all nonzero programmes scheduled in the target window. Repeats are allowed. Common fallback, fixed algorithm from R1 onward: distinct-user popularity among qualifying historical continuous-session watches, descending count with ID-ascending ties, restricted to target candidates. Zero-popularity candidates remain available for deterministic final backfill.
- Primary MAP@5 sums precision at relevant predicted ranks and divides by min(5, the full declared relevance-set size). Predictions are distinct. Unreachable labels remain in the denominator and are counted explicitly.

## Model inputs and interpretation

R1 personal scores are distinct qualifying historical broadcast counts, with latest qualifying overlap end then ID as secondary ties. There is no decay, opportunity factor or fusion. Content and ALS are independently refitted on qualifying **counts**, replacing R0's broad watched-second weights. Content otherwise retains prefix-only metadata/TF-IDF. ALS retains 32 factors/15 exact CPU iterations/regularization 0.1/seed 42 and confidence 1+10 log1p(count). Scoring uses user-factor × item-factor dot products, permits repeated items explicitly, and avoids the library's default repeat filter.

R0 strategy is replayed using its original broad >=300-second interactions fitted only before the development cutoff, then evaluated on this new target. This is a new-label strategy comparison; R0's original 0.595717 remains a historical broad-label proxy result. R0's returned private score remains approximate, user-reported 0.27 until exact evidence is supplied.

Unresolved: precise hidden preference definition; session/device ambiguity; target tie conventions; empty-label population treatment; naive timestamps/DST; transfer from four-week development to twelve-week prediction. Repeated development experiments and the four authorized Kaggle feedback queries are adaptive selection.
