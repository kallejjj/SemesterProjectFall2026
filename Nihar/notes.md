
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