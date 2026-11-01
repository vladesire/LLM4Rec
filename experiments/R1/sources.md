# Sources actually consulted for R1

Consulted on 1 November 2026 in this genuine local session.

- [Competition overview](https://www.kaggle.com/competitions/sweettv-tv-program-recommender/overview): attempted before implementation; the web tool could not retrieve readable competition text. The student's pasted description is the available account-side evidence: predict five most preferred IDs, based on subsequent twelve-week views, scored with MAP@5. Exact hidden preference construction remains unverified.
- [N. Melentev, historical second-place methodology](https://medium.com/data-science/two-layered-recommender-system-methodology-a-prize-winning-solution-c4bdb815b156): consulted for the completion-without-channel-change interpretation and personal repetition idea. It is participant testimony, not an organizer specification or our result.
- [Participant target/features implementation, utils.py](https://github.com/ndmel/2nd_place_recsys_cinema_challenge_2020/blob/main/utils.py), including its [raw view](https://raw.githubusercontent.com/ndmel/2nd_place_recsys_cinema_challenge_2020/main/utils.py): inspected get_target and get_features. The target uses inclusive 0.8 completion and frequency. Our session join, clipping, provenance, ties, scoring and checks are independently implemented; no third-party source code or predictions were copied.
- [Implicit recommendation interface](https://benfred.github.io/implicit/api/models/recommender_base.html): consulted for matrix orientation and the default already-liked filter. Installed implicit 0.7.3 signature was inspected. Our ALS scores use explicit dot products; a real call with filter_already_liked_items=False checked the mapping and scores.

Other proposed public repositories, the Kaggle notebook and LightGBM documentation have not been consulted for R1, because their later-round ideas are not needed for this round.
