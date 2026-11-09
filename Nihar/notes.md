
*** “AccZ variation inside bumps grows strongly with speed; at ~40 km/h the bump at 4,228 m is invisible in AccZ variation (ratio 1.0).”**


### Section 5: How many bumps of each type and size?

**Real frost-heave bumps: 19** (each bump counted once, even if it was driven several times)

| posneg \ vaikeus | 1 | 2 | 3 | 5 | Total |
|---|---|---|---|---|---|
| 1 (up) | 3 | 1 | 5 | 6 | 15 |
| 2 (down) | 2 | 0 | 1 | 1 | 4 |
| **Total** | 5 | 1 | 6 | 7 | 19 |

- Sizes used: 1, 2, 3 and 5. No size 4, and only one size 2. This matches the project legend (1 = small, 3 = medium, 5 = severe).
- Proposed size groups:

| Group | vaikeus | Real bumps |
|---|---|---|
| Small | 1–2 | 6 |
| Medium | 3 | 6 |
| Large | 5 | 7 |

**What can realistically be predicted**

| Target | Data available | Realistic? |
|---|---|---|
| 1. Is there a frost heave? | 19 real bumps, seen 47 times across drives |  Main goal |
| 2. Up or down? | 15 up vs. 4 down, all 4 on one road |  Secondary goal, a limitation |
| 3. How big? | 3 groups: 6 / 6 / 7 |  Feasible as small / medium / large, with very few examples per group |


## Methodology decisions (after exploration, 19 Oct)

### Done
- Cleaning: decimal commas, blended label transitions (settle rule), 1 missing speed value per file (interpolated), trimmed slow start/end. Verified: no bumps or labelled rows lost.
- Exploration: bump shape, up vs. down, effect of speed, label alignment, counts by type/size.
- Column selection: candidate inputs = AccX/Y/Z, GyrX/Y/Z, Roll, Pitch, Speed. Excluded: Speed_kmh (duplicate), Loc/Timer (position/time), East/North (location → model would memorise places; kept for dashboard map), CElev (hills), Yaw (direction). RBCSV_80/60: ask supervisor.

### Not done, and why
- **Correlation on raw rows (phik etc.):** misleading for this data. A bump makes the signal *swing*, not rise, so raw values average out (correlation ≈ 0) even for sensors that react strongly. Information is in the *variation over a stretch of road* → correlation is done on window features instead.
- **Cardinality:** only applies to categorical inputs; all inputs are continuous. Categorical labels already examined in exploration.
- **SMOTE for posneg = 2:** only 4 negative bumps, all from one road and one drive. Synthetic samples would only blend those 4 (no new information), and with one-road-out testing there is either nothing to oversample or nothing to test on. Use class weights instead, ask for more data, report as a limitation.

### Moved to later
- **Correlation / phik and multicollinearity** → 04, on window features (all 8 sensors).
- **Noise filtering** → 04, tested as a feature choice (raw vs. smoothed), because part of the "noise" (shaking) is the signal itself.
- **Class imbalance (up/down)** → 05, with class weights.

- **Testing strategy:** split by road (train on 2 roads, test on the 3rd, 3 rounds). A random split would put windows of the same bump, and the same bump from other drives, into both training and test (data leakage) and give over-optimistic scores.


- Switched `fix_transitions` to tol = 0.02 (see 02b). All files rebuilt. Effect: 8 Iisingintie edge windows moved from "left out" to "no bump"; RF results unchanged (F1 0.45, AUC 0.76 → 0.77).

- RBCSV (workshop point 1): strong, speed-independent, new info (corr 0.4–0.5 with sensors). Road split: AUC 0.76→0.79 but F1 0.44→0.41 (Kaittiaisentie false alarms up). Random split: F1 0.59→0.66, AUC 0.83→0.88 → leakage inflates both. Main RF stays sensors only (fair comparison with MiniRocket/CNN). Random vs road split done (workshop leakage demo).

- 07 MiniRocket (sktime 1.2.0; sklearn pipeline gave an error → steps written by hand): alpha hit the range edge (1000) → widened to 10⁶ → alpha 10,000. Avg F1 0.40, AUC 0.74 (RF: 0.44 / 0.76). Ranks nearly as well as the RF but more false alarms at the default threshold. Predictions saved.

- 08 1D CNN (keras): overfits fast; 5 seeds → stable (F1 std ≤ 0.03). Avg F1 0.40, AUC 0.70. Best on sinetänsalmentie (most training data), weakest on Iisingintie (least). All three models end at F1 0.40–0.44 → the limit seems to be the data, not the model. Predictions saved.

- 08 LSTM (3 seeds): avg F1 0.35, AUC 0.69, most false alarms → weakest. Ranking: RF 0.44 > MiniRocket 0.40 = CNN 0.40 > LSTM 0.35. Arsenal skipped for now (same Ridge limits as MiniRocket, slow).

- 07 closed: Arsenal predictions saved (road split, F1 0.49). Next: 09 comparison (bump level), then 10_arsenal_tuning.