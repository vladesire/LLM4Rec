# R3 plan before new candidate/reranker results

R2 returned user-reported public 0.39904/private 0.39378, becoming the scored incumbent. Both scores improve over R1; the package remains the supported reference, without isolating causal ingredients.

First diagnose a target-independent pool of at most 200 eligible items per user from personal frequency, R2, existing ALS/content and the common popularity. Preserve R2 Top-5 inclusion. Calculate micro/macro candidate recall and oracle MAP@5 with the full declared relevance denominator. Oracle values are ceilings. If micro recall is below 0.8, investigate a complementary source before training. Never inject held-out positives.

If recall supports reranking, use the installed scikit-learn histogram boosted classifier with binary log loss; official API documentation and the installed signature support explicit chronological X_val/y_val. No extra model dependency is required. A binary objective is not an exact MAP@5 optimizer.

- Training snapshots: Mar23–Apr20 and Apr6–May4, with features strictly before each snapshot's start. Their target windows overlap; this is disclosed.
- Inner early-stopping/configuration window: May4–June1. All fitting labels end at or before May4, so no inner-window outcomes are fitting examples.
- Development: June1–June29. Refit on earlier snapshots plus the inner window, using the selected stopped iteration count, then evaluate. All supervised fitting labels end before June1.
- Production: refit the chosen model on the above historical examples plus the June development example; all labels end before June29. Use all qualifying feature history before July27 for production scoring. No corrected June29–July27 targets are made or inspected.

Two compact leaf/regularization settings are predeclared. Early stopping uses inner log loss, configuration choice uses checked inner MAP@5. Feature groups: personal rates/ranks/completion/recency, schedule/habit forecasts, ALS/content scores/ranks, and reliability/history/missingness. Counts and schedule forecasts become per-week rates or shares; include observed history length. Train histories of roughly 2/4 weeks, inner 8 weeks and development 12 weeks provide a short-to-long-history analogue before 20-week production; it does not remove extrapolation uncertainty.

Evaluate two family-drop ablations and two fixed R2/reranker rank blends on development. Do not ship an inferior reranker. If R2 remains best, report that and choose a justified simple challenger only after inspecting the failure, rather than silently duplicating R2.

Keep the R1 target/candidate/repeat/fallback contract and original evidence intact. The later corrected audit remains sealed until R4. Stop with one immutable R3 CSV and await the student's score/comments before R4.
