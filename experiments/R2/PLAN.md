# R2 plan, fixed before inspecting R2 development scores

R1 returned user-reported public 0.39334/private 0.38480 and is the scored incumbent. This supports its personal/completion package over approximate R0 private 0.27. Exact contributions and future transfer remain uncertain; original Kaggle evidence has not been inspected.

Keep the R1 target, session construction, 80% threshold, user population, candidates, repeat policy and common fallback. All feature models stop before 1 June; June 1–29 is development. The corrected June 29–July 27 audit remains sealed. The explicitly separate production refit may use all pre-test history.

1. Compare repeat counts with no decay and exponential 28/56-day half-lives. Decay uses each distinct qualifying broadcast's clipped observation end. Select the best repeat variant by the saved primary rule.
2. At that half-life, compare a smoothed per-programme watch rate projected over future scheduled airings and a capped schedule-rate multiplier. Historical and target counts are normalized by effective history days/forecast days. Availability is a proxy for opportunity, not actual user exposure.
3. Compare channel-only and channel/time habit forecasts. Use labelled historical broadcast opportunities, >=80% qualifying watches, coarse six-hour cells plus weekday/weekend, and fixed hierarchical shrinkage. No future outcomes or metadata are fitted. Cold users retain the common fallback.
4. Test two global repeat/schedule blends (schedule weights 0.25/0.5) and two repeat/habit blends (habit weights 0.1/0.25). If both additions independently improve on the repeat reference, allow one predeclared three-way blend (0.6/0.3/0.1). No user segmentation or expanded search.

Choose one nonduplicate R2 CSV with the highest primary development MAP@5; exact ties prefer simplicity and longer/no decay. Report all ablations and unchanged-prediction target sensitivities. This is adaptive local selection, not independent confirmation. If no change is locally supported, pause to explain the result rather than silently duplicate R1.

R2 tests a package when multiple ingredients are selected; individual effects are attributed only to corresponding local ablations. No upload is performed by the agent, and R3 waits for the student's R2 result and instruction.
