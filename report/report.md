# Temporal TV Programme Recommendation

**Vladislav Neklyudov** · vsneklyudov@edu.hse.ru

**Individual LLM4Rec/FKN assignment · 1 November 2026**

GitHub repository: TODO: accessible GitHub repository URL

**DRAFT — required student and Kaggle evidence pending**



## Abstract

This work predicts five television programme IDs per user from implicit viewing logs and broadcast schedules. I compare distinct-user popularity, sparse TF-IDF content filtering, and confidence-weighted implicit ALS on a chronological four-week hold-out. Half-open interval overlap, a five-minute per-airing threshold, and prefix-only learning produce 764,336 fitting user–programme pairs. Popularity achieves MAP@5 of 0.595717 across all 1,260 official-template users, outperforming both personalized methods. It is therefore refitted on eligible pre-test history for the designated CSV. The result supports a simple recurring-programme baseline for these labels; it does not establish individual enjoyment. Private-score evidence and the student's personal review remain pending in this draft.



## 1. Introduction and related work

The task is to rank programmes that will actually air. For example, a viewer who repeatedly watches a news programme could receive its next scheduled occurrence. Programme IDs recur across broadcasts, so previously watched items remain eligible. Availability, channel changes, and unlabelled airtime make a direct viewing-log-to-item mapping unreliable.

The supplied competition files [1] provide implicit feedback rather than ratings. Hu, Koren and Volinsky [2] motivate separating binary preference from confidence in an observation. Content filtering instead compares a profile of consumed attributes with programme features. I use sparse TF-IDF as documented by scikit-learn [3], and the CPU implicit implementation [4] for a genuine factor model. No language model or deep neural recommender is trained.



## 2. Data and label construction

The viewing CSV contains 5,821,987 rows and 4,914 users, from 2020-03-09 00:00:02 to 2020-07-26 23:59:58. All CSVs use comma delimiters and UTF-8-compatible text. User IDs are preserved as strings, including values above signed 64-bit range. Schedule dates use day.month.year; viewing dates use year-month-day. Both are parsed onto the same naive wall-clock axis; the files do not establish a timezone.

Schedule duration is in seconds (first-file median 1500); viewing duraton is an interval length, not programme length. 117 values disagree with timestamp subtraction, so start/stop timestamps define the joins. Direct vsetv_id=channel_id matching leaves five observed channels unmapped: 149, 1174, 1508, 1509 and 1510. No invented remapping is used.

Nonzero-ID row coverage is 10.76% in the first schedule and 11.31% in the second. Zero IDs are excluded from matrices and recommendations. 17 duplicate viewing intervals are removed; overlapping/adjacent intervals within each user/channel are unioned (380,197 merges). The schedule has 1 nonpositive-duration row and 10 conflicting airings; all are removed, including 1 labelled airing. This conservative policy prevents duplicate coverage.



<!-- pagebreak -->



## 3. Method and leakage controls

For viewing [v_start,v_end) and airing [a_start,a_end), overlap is max(0, min(v_end,a_end) − max(v_start,a_start)). A channel-wise sorted interval join advances each viewing through every intersected broadcast. It is not a Cartesian join, and a long viewing may yield several labels. Visits to the same airing are summed after interval union.

A watched label requires at least 300 seconds per user–airing. This absolute threshold was chosen after inspecting viewing lengths (full-data median 396 seconds), to reduce brief zapping without requiring completion. It is not tuned. Retained airings are aggregated into user–programme seconds. Unlabelled airtime is missing information, not a dislike.

Fitting uses observations before 29 June 2020 00:00; validation is [29 June,27 July 2020 00:00). Viewing is clipped at the cutoff before overlap and thresholding, so crossing intervals cannot leak future seconds. Four late weeks provide forward prediction with substantial history. This is shorter than the future schedule [27 July,19 October 2020 12:00), which limits horizon comparability. These dates come from parsed rows, not filename week numbers.

All methods rank the same nonzero IDs scheduled in the target window. Repeats are allowed. Positive, finite method scores are sorted descending; integer programme ID breaks ties. Missing or insufficient method scores use the same prefix-popularity order, without duplicates. Unseen programmes with zero popularity are deterministic ID-tie backfill candidates, rather than inferred dislikes.

Popularity scores each programme by the number of distinct fitting users with a retained label. This definition also supplies every fallback. It uses neither held-out viewing nor future metadata.

Content filtering uses one earliest available metadata row per nonzero programme before the fitting cutoff. Features are title word tokens, category and three genres (field tokens repeated twice), year, director, and up to 20 actor names. Prefix-only vocabulary and smoothed IDF produce L2-normalized sparse item vectors. A user profile is the L2-normalized weighted mean with weight log1p(retained seconds/300). Cosine similarity ranks candidates with prefix metadata; items without that metadata receive no content score. No held-out or test metadata enters vocabulary, IDF, or profiles.

ALS minimizes sum_ui c_ui(p_ui − x_u·y_i)^2 + λ(sum_u ||x_u||² + sum_i ||y_i||²). Observed pairs have p_ui=1 and c_ui=1+10 log1p(retained seconds/300); unobserved pairs have p_ui=0,c_ui=1. Settings are 32 factors, 15 exact ALS iterations, λ=0.1, seed 42, float32 and one CPU thread; implicit alpha=1 avoids a second confidence scaling. Unknown items and cold users use the common fallback. It is matrix factorization, not nearest-neighbour similarity.

The installed environment is arm64 macOS, Python 3.15.0, NumPy 2.5.4, pandas 3.1.0, SciPy 1.18.1, scikit-learn 1.9.1 and implicit 0.7.3. Sparse matrices and a fingerprinted local preprocessing cache keep the pipeline small. Exact package versions, input SHA-256 values, parameters and stage timings are saved.



<!-- pagebreak -->



## 4. Experiments, checks and discussion

Fitting contains 4,856 labelled users, 764,336 distinct user–programme pairs and 3,837 interacted items. Validation has 260,581 labelled pairs across log users; 89,041 belong to the 1,260 official-template users used for primary evaluation. There are 3,975 available candidate items.

AP@5 sums precision at each relevant recommendation rank and divides by min(5, number of distinct relevant items). The primary MAP averages over every template user. Empty relevance contributes zero; relevance sets and outputs are distinct, and duplicate predictions are rejected. This follows the supplied brief. Current authenticated Kaggle metric/account behavior was unavailable, so exact server convention remains to be confirmed rather than assumed identical.

The population includes 22 users without held-out labels and 6 users without fitting labels. No unreachable relevant user–item labels are removed (0 observed). Secondary nonempty-user MAP is 0.606303 for popularity; warm/cold MAP is 0.597443/0.235000. All comparisons below use the identical split and policy.


| Method | MAP@5 | Fit (s) | Infer (s) | Scored-slot coverage | Distinct Top-5 IDs |
|---|---:|---:|---:|---:|---:|
| Popularity | 0.595717 | 0.015 | 0.252 | 100.00% | 5 |
| Content | 0.177915 | 0.186 | 0.467 | 99.52% | 460 |
| Als | 0.157976 | 3.962 | 0.303 | 99.52% | 814 |


Content has metadata for 3204/3975 candidates; ALS has fitted factors for 1978/3975. Both use full popularity fallback for the six cold users. Preparation took 11.602 s and the successful entry point 17.595 s in this installed environment, excluding setup and exploratory runs. These are observations, not runtime guarantees.

Only one configuration of each individual method was compared; there was no hyperparameter search, hybrid or Kaggle score tuning. The same hold-out selected popularity, so results are selection-biased rather than an independent final evaluation. The production refit uses 949,489 pre-test pairs and 6,636 future candidates. Its validation score is not recomputed on refitted history.

The join assigns 51.61% of fitting and 57.49% of held-out viewing seconds to labelled airings (52.89% overall). It drops 773,647/161,614 below-threshold fitting/held-out user–airings. Missing schedules and unlabelled broadcasts limit observed relevance. Background viewing, unequal exposure, and the shorter validation horizon further limit interpretation.

Popularity wins under these broad recurring-TV labels but recommends just five distinct IDs to the whole population. Personalized content and ALS spread exposure more widely and score worse; their similarity/confidence objectives do not directly optimize this AP@5 target. Retaining the measured winner is a practical mitigation, not a claim that diversity or personal preference has been solved. No live A/B, enjoyment, retention or revenue result is claimed.

An actual preparation failure occurred when pandas Series.item resolved to a method during example serialization. Explicit column indexing fixed it; the added integration check and completed full entry point confirmed the repair. 6 focused tests passed. A scalar reference checked 10,000 interval pairs. A 230-user, 22,850-pair probe produced identical scores and Top-5 twice for all three methods, including when 17,519 post-cutoff metadata rows were replaced. This is a small rerun, not a second full clean run.



<!-- pagebreak -->



## 5. The same two users across all methods

Examples use observed validation viewing; true test-period viewing is unavailable. Histories and all ordered Top-5 IDs, titles, categories, method scores and fallback flags are saved in results/two_user_examples.json and results/two_user_histories.csv. The users are chosen near the 10th and 90th percentiles of fitting history length among evaluable template users, rather than selected for model wins.



### User 1: 8848657562275607589

Fitting history: 73 distinct programmes; held-out history: 22.


| Method | Ordered Top-5 programme IDs | AP@5 |
|---|---|---:|
| Popularity | 2400480 → 240081 → 6500479 → 2400508 → 2400467 | 0.200000 |
| Content | 200337 → 200477 → 700389 → 20088 → 500315 | 0.050000 |
| Als | 2400480 → 2400508 → 240081 → 90079839 → 6500479 | 0.233333 |


Fitting: ТСН. Неделя. (ID 2400480, 9,639 s); 'Холостяк', 10 сезон, 1 эп., 1 ч. (ID 400357, 8,100 s) Held-out: Сегодня. Итоги с Олегом Панютой. (ID 2400508, 5,826 s); Х/ф 'Ты только верь', 1 и 2 с. (ID 90082215, 4,470 s)



### User 2: 7124081213039446293

Fitting history: 368 distinct programmes; held-out history: 101.


| Method | Ordered Top-5 programme IDs | AP@5 |
|---|---|---:|
| Popularity | 2400480 → 240081 → 6500479 → 2400508 → 2400467 | 0.710000 |
| Content | 90057357 → 90053892 → 90073872 → 90067324 → 90049509 | 0.000000 |
| Als | 90077791 → 90049405 → 90066769 → 90071307 → 90071783 | 0.000000 |


Fitting: Кто сверху?, 6 сезон, 7 эп. (ID 700389, 50,137 s); М/с 'Губка Боб Квадратные Штаны'. (ID 12001682, 48,193 s) Held-out: 'Мамахохотала-шоу'. Лучшее. (ID 200405, 56,862 s); 'Затерянный мир'. (ID 2500430, 26,375 s)



### Baseline titles, scores and actual relevance

The ordered popularity list is shared by the two users. Scores are distinct fitting-user counts; Y/N indicates held-out relevance for user 1/user 2.


| ID | Programme title | Category | Score | Relevant 1/2 |
|---:|---|---|---:|---|
| 2400480 | ТСН. Неделя. | Инфо | 4439 | N/Y |
| 240081 | Факты. День. | Инфо | 4177 | Y/Y |
| 6500479 | 'Мир наизнанку'. | Познавательное | 3827 | N/N |
| 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | 3734 | Y/Y |
| 2400467 | 'Викна-Новости'. | Инфо | 3718 | N/Y |


For user 8848657562275607589, the saved real AP@5 example has 22 distinct relevant IDs and hits at ranks 2 and 4: (1/2+2/4)/5=0.200000. This matches the recorded result. Student personal verification remains pending unless completed below.

Agent-observed interpretation: ALS slightly improves on popularity for user 1; both personalized lists score zero for user 2. High fitting history volume alone therefore does not guarantee a useful forward ranking. The measured evidence supports the aggregate baseline choice while retaining individual failure examples.

**Student interpretation:** TODO: Review the two-user examples and rewrite the interpretation in your own words.

**Personal join check:** TODO: Personally inspect a raw viewing/schedule overlap and explain your arithmetic.

**Personal AP@5 check:** TODO: Personally recompute the saved real user's AP@5 and explain your reasoning.



<!-- pagebreak -->



## 6. External validation and AI disclosure

**Kaggle submission:** TODO: designated final submission identifier. Status: TODO: wait for COMPLETE. Private score: TODO: genuine returned PRIVATE score.

Description: TODO: full Russian name and Kaggle username entered on the designated submission. Observed quota: 100 submissions shown in Kaggle, reported by the student on 1 November 2026; remaining quota not independently checked. Current account behavior: TODO: confirm current Late Submission and private-score availability in your account.

The designated file is submission/final_submission.csv (1260 rows, exact template order and schema). SHA-256: 73d71295c19c61c3702a3c8e0fae951f27a307ce9b05cd39fd37e9b93e0996e0. Uploaded identity: TODO: checksum of the exact uploaded CSV.



![Genuine designated final Kaggle submission, including PRIVATE score and submission identity](assets/kaggle_private_score.png)



ChatGPT assisted with planning before the local implementation session. That upstream activity is disclosed separately and is not represented as part of the local Codex audit. In this genuine local session, Codex inspected data, wrote and debugged code, ran joins/models/checks, selected the recorded baseline, and drafted documentation and this PDF. Web browsing consulted the sources below; Python libraries generated numerical outputs and the PDF. No subagents or other model APIs were used.

The agent ran the checks recorded in results/verification.json. Student personal checks completed: False; full report review completed: False. Library implementations and source descriptions were trusted within the documented checks; the competition server metric, account quota and private-score behavior were not verified by the agent. Numerical results come from saved outputs, not generated prose.

Original local audit reference: TODO: basename of unchanged original native session file submitted. Native format: JSONL. The student will copy the original native JSONL file unchanged after the documented work. Earlier ChatGPT planning is disclosed separately and is not included in this local session file. The located native session is rollout-2026-11-01T14-39-42-01a196e8-8117-7b61-a05e-d4d86d8f1472.jsonl. This is JSONL, not a single JSON document; course acceptance confirmed: False. The agent has not copied, reconstructed, edited, sanitized, simulated or converted that original. Saved chat history is distinct from earlier external planning [5].



<!-- pagebreak -->



## References

[1] sweet.tv, “sweet.tv – TV program recommender,” Kaggle, competition data. [Competition data page](https://www.kaggle.com/competitions/sweettv-tv-program-recommender/data), accessed Nov. 1, 2026. Local downloaded CSVs were inspected; the public page exposed no readable metric details.

[2] Y. Hu, Y. Koren, and C. Volinsky, “Collaborative Filtering for Implicit Feedback Datasets,” in Proc. IEEE ICDM, 2008, pp. 263–272, doi: 10.1109/ICDM.2008.22. [Author-hosted paper](https://yifanhu.net/PUB/cf.pdf), consulted Nov. 1, 2026.

[3] scikit-learn developers, “Feature extraction,” scikit-learn documentation. [Official feature-extraction documentation](https://scikit-learn.org/stable/modules/feature_extraction.html), consulted Nov. 1, 2026.

[4] B. Frederickson and implicit contributors, “AlternatingLeastSquares,” Implicit 0.7.2 API documentation. [CPU ALS documentation](https://benfred.github.io/implicit/api/models/cpu/als.html), consulted Nov. 1, 2026. Installed library version: 0.7.3.

[5] OpenAI, “Projects and chats,” ChatGPT/Codex documentation. [Official chat documentation](https://learn.chatgpt.com/docs/projects), consulted Nov. 1, 2026. It describes saved transcripts; it does not establish a JSON export mechanism for this session.

