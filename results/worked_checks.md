# Worked checks for student review

Agent-executed checks are recorded in verification.json. Student personal verification remains pending.

The numbers below are synthetic seconds on a common time axis, not fabricated dataset rows.

| Case | Viewing | Broadcast | Hand calculation |
|---|---|---|---|
| Partial overlap | [90,150) | [100,200) | min(150,200)-max(90,100) = 50 seconds |
| Half-open no overlap | [0,100) | [100,200) | max(0,100-100) = 0 seconds |
| One viewing spans two broadcasts | [50,250) | [100,200), [200,300) | 100 seconds in the first and 50 in the second; neither is lost |
| Crossing split at 200: fit | [90,250) clipped to [90,200) | [100,300) | 100 seconds; no second at or after cutoff enters fitting |
| Crossing split at 200: hold-out | [90,250) clipped to [200,250) | [100,300) | 50 seconds; total across the two sides is 150 |

## Genuine matched rows

| Channel | Programme | Viewing | Broadcast | Overlap |
|---|---:|---|---|---:|
| 3 | 2400480 | 2020-03-10 19:42:11 to 2020-03-10 21:02:04 | 2020-03-10 19:30:00 to 2020-03-10 20:45:00 | 3769 s |
| 3 | 20088 | 2020-03-11 07:22:08 to 2020-03-11 08:45:52 | 2020-03-11 07:10:00 to 2020-03-11 08:00:00 | 2272 s |
| 4 | 90043107 | 2020-03-21 11:51:29 to 2020-03-21 11:54:20 | 2020-03-21 11:00:00 to 2020-03-21 13:40:00 | 171 s |
| 4 | 90034350 | 2020-03-21 15:27:17 to 2020-03-21 15:34:21 | 2020-03-21 13:40:00 to 2020-03-21 16:30:00 | 424 s |

For example, 19:42:11–20:45:00 is 1,800 + 1,800 + 169 = 3,769 seconds in the first genuine row. The rest of that viewing interval can match subsequent broadcasts.

## AP@5 reference checks

- Relevant {A,B,C}; [A,B,D,C,E]: (1 + 1 + 3/4)/3 = 11/12.
- Relevant {A,B,C}; [D,A,E,B,C]: (1/2 + 2/4 + 3/5)/3 = 8/15.
- Empty relevance contributes AP=0 and remains in the population.
- Duplicate outputs are rejected before scoring, rather than receiving repeat credit.

## One actual user calculation

User `8848657562275607589`, method `popularity`, has 22 distinct held-out relevant programmes. Denominator = min(5,22) = 5.

| Rank | Programme | Held-out hit | Precision | Numerator contribution |
|---:|---:|---|---:|---:|
| 1 | 2400480 | False | 0.000000 | 0.000000 |
| 2 | 240081 | True | 0.500000 | 0.500000 |
| 3 | 6500479 | False | 0.333333 | 0.000000 |
| 4 | 2400508 | True | 0.500000 | 0.500000 |
| 5 | 2400467 | False | 0.400000 | 0.000000 |

AP@5 = 1.0/5 = 0.200000.

The complete relevant-ID set is in real_user_ap_example.json. The fitting and held-out histories for the same two example users are in two_user_histories.csv.

**Student action:** inspect the original CSV rows and recompute at least one join and this AP@5 example personally. Record your own reasoning in report/student_details.json; no personal check is claimed yet.
