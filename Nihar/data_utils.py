import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

LABELS = ["posneg", "vaikeus", "muu"]

FEATURES = ["GyrY_std", "AccZ_std", "GyrX_std", "GyrZ_std",
            "AccY_std", "AccX_std", "Pitch_std", "speed"]

def fix_transitions(s, settle=3, tol=0.02):
    """Give every label transition the value it settles on (method from a friend,
    extended: hovering near a whole number also counts as settled)."""
    result = [None] * len(s)
    pending = []
    current = None
    count = 0
    last = 0

    for i, value in enumerate(s):
        nearest = round(value)
        if abs(value - nearest) <= tol:
            if nearest == current:
                count = count + 1
            else:
                current = nearest
                count = 1
        else:
            current = None
            count = 0

        pending.append(i)

        if count >= settle:
            for j in pending:
                result[j] = current
            pending = []
            last = current

    for j in pending:
        result[j] = last

    return pd.Series(result, index=s.index)


def trim_ends(df, frac=0.8):
    """Cut off the start and end of a drive where the car is not yet up to speed."""
    typical = df["Speed_kmh"].median()
    fast = df["Speed_kmh"] >= frac * typical
    first = fast.idxmax()
    last = fast[::-1].idxmax()
    return df.loc[first:last]

def keep_longest_continuous(df, max_gap_ms=5):
    """Keep only the longest stretch of rows without a time jump."""
    gap = df["Timer(ms)"].diff() > max_gap_ms
    stretch = gap.cumsum()
    longest = stretch.value_counts().idxmax()
    return df[stretch == longest]



def load_clean(path, trim=True, continuous=True, transfix=True):
    # 1. Read the tab-separated file as text, fix decimal commas, turn into numbers
    df = pd.read_csv(path, sep="\t", dtype=str)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col].str.replace(",", "."), errors="coerce")

    # 2. Fix label transitions
    if transfix:
        df[LABELS] = df[LABELS].ffill()
        for col in LABELS:
            df[col] = fix_transitions(df[col]).astype(int)

    # 3. Fill the single missing speed value with the average of its neighbours
    df["Speed (m/s)"] = df["Speed (m/s)"].interpolate().bfill()


    # 4. Speed in km/h, drop the row counter
    df["Speed_kmh"] = df["Speed (m/s)"] * 3.6
    df = df.drop(columns=["Num"])

    # 5. Cut the slow start and end of the drive
    if trim:
        df = trim_ends(df)

    # 6. Keep the longest stretch without time jumps
    if continuous:
        df = keep_longest_continuous(df)

    print(f"{path}: {len(df)} rows")
    return df


def plot_bump(df, center, title, cols=("AccZ", "Pitch(deg)", "GyrY")):
    part = df[df["Loc(m)"].between(center - 30, center + 50)]
    fig, axes = plt.subplots(len(cols), 1, figsize=(12, 7), sharex=True)
    for ax, col in zip(axes, cols):
        ax.plot(part["Loc(m)"], part[col], linewidth=0.7)
        ax.fill_between(part["Loc(m)"], 0, 1, where=part["muu"] == 1,
                        transform=ax.get_xaxis_transform(), color="red", alpha=0.15)
        ax.set_ylabel(col)
    axes[0].set_title(title)
    axes[-1].set_xlabel("Loc (m)")
    plt.tight_layout()
    return fig

SENSORS = ["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ", "Roll(deg)", "Pitch(deg)"]


def label_window(frac):
    if frac > 0.5:
        return "bump"
    elif frac == 0:
        return "no bump"
    else:
        return "left out"


def size_group(v):
    if v in (1, 2):
        return "small"
    elif v == 3:
        return "medium"
    elif v == 5:
        return "large"
    return None


def window_features(g):
    # Compute features for one window of data. Returns a Series with the features.
    feats = {"speed": g["Speed (m/s)"].mean()}
    for col in SENSORS:
        x = g[col]
        name = col.split("(")[0]
        feats[f"{name}_std"] = x.std()
        feats[f"{name}_range"] = x.max() - x.min()
        feats[f"{name}_mean"] = x.mean()
    # Compute RBCSV features
    for col in ["RBCSV_60(cf)", "RBCSV_80(cf)"]:
        x = g[col]
        name = col.split("(")[0].replace("_", "")
        feats[f"{name}_mean"] = x.mean()
        feats[f"{name}_max"] = x.max()
        feats[f"{name}_std"] = x.std()
    d = g["RBCSV_80(cf)"] - g["RBCSV_60(cf)"]
    feats["RBCSV_delta"] = d.mean()
    feats["RBCSV_delta_absmax"] = d.abs().max()
    feats["RBCSV_delta_std"] = d.std()
    return pd.Series(feats)


def make_windows(df, drive, window_m=5):
    """Cut one cleaned drive into windows, label them and compute features."""
    df = df.copy()

    # 1. Cut
    df["window"] = (df["Loc(m)"] // window_m).astype(int)
    df["is_heave"] = df["muu"].isin([1, 2])

    # 2. Summarise position and frost-heave fraction per window
    w = df.groupby("window").agg(
        start_m=("Loc(m)", "min"),
        end_m=("Loc(m)", "max"),
        heave_frac=("is_heave", "mean"),
    )
    w["length_m"] = w["end_m"] - w["start_m"]
    w = w[w["length_m"] >= window_m * 0.9]          # drop incomplete windows

    # 3. Label
    w["label"] = w["heave_frac"].apply(label_window)
    heave = df[df["is_heave"]]
    w["vaikeus"] = heave.groupby("window")["vaikeus"].agg(lambda s: s.mode()[0])
    w["posneg"] = heave.groupby("window")["posneg"].agg(lambda s: s.mode()[0])
    w["size_group"] = w["vaikeus"].apply(size_group)
    w.loc[w["label"] != "bump", ["vaikeus", "posneg", "size_group"]] = None

    # 4. Features
    feats = df.groupby("window")[SENSORS + ["Speed (m/s)", "RBCSV_60(cf)", "RBCSV_80(cf)"]].apply(window_features)
    w = w.join(feats)

    # 5. Remember where each window comes from
    w["drive"] = drive
    w["road"] = drive.split("_")[0]
    return w.reset_index()


def bump_level(p):
    bumps_total = 0
    bumps_found = 0
    real_false_alarms = 0
    false_alarm_spots = 0

    for drive in p["drive"].unique():
        # This drive's windows, in road order.
        d = p[p["drive"] == drive].sort_values("window")

        # --- 1. Group the bump windows into bump passes ---
        bump_windows = d[d["is_bump"] == 1]["window"].tolist()
        bumps = []
        for w in bump_windows:
            if len(bumps) > 0 and w - bumps[-1][-1] == 1:
                bumps[-1].append(w)     # next to the previous bump window -> same bump
            else:
                bumps.append([w])       # a gap -> a new bump starts

        # --- 2. Was each bump found? ---
        flagged = d[d["pred"] == 1]["window"].tolist()
        for bump in bumps:
            bumps_total = bumps_total + 1
            for w in bump:
                if w in flagged:
                    bumps_found = bumps_found + 1
                    break

        # --- 3. Real false alarms ---
        alarms = d[(d["pred"] == 1) & (d["is_bump"] == 0)]["window"].tolist()
        real_fa_windows = []
        for a in alarms:
            near_a_bump = False
            for b in bump_windows:
                if abs(a - b) <= 2:
                    near_a_bump = True
            if not near_a_bump:
                real_false_alarms = real_false_alarms + 1
                real_fa_windows.append(a)

        # --- 4. Group neighbouring false alarm windows into spots ---
        for i in range(len(real_fa_windows)):
            if i == 0 or real_fa_windows[i] - real_fa_windows[i - 1] > 1:
                false_alarm_spots = false_alarm_spots + 1

    km = len(p) * 5 / 1000
    return {
        "bumps_total": bumps_total,
        "bumps_found": bumps_found,
        "detection_rate": bumps_found / bumps_total,
        "real_false_alarms": real_false_alarms,
        "false_alarms_per_km": real_false_alarms / km,
        "spots_per_km": false_alarm_spots / km,
    }


# ===================== Pipeline for new files (dashboard) =====================

STD_FEATURES = ["GyrY_std", "AccZ_std", "GyrX_std", "GyrZ_std", "AccY_std", "AccX_std", "Pitch_std"]


def load_clean_file(file, trim=True, continuous=True, sep="\t"):
    """Same cleaning as load_clean, but works for an uploaded file, and labels are optional."""
    df = pd.read_csv(file, sep=sep, dtype=str, encoding="utf-8-sig")
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col].str.replace(",", "."), errors="coerce")

    has_labels = True
    for col in LABELS:
        if col not in df.columns:
            has_labels = False

    if has_labels:
        df[LABELS] = df[LABELS].ffill()
        for col in LABELS:
            df[col] = fix_transitions(df[col]).astype(int)

    df["Speed (m/s)"] = df["Speed (m/s)"].interpolate().bfill()
    df["Speed_kmh"] = df["Speed (m/s)"] * 3.6
    df = df.drop(columns=["Num"], errors="ignore")
    if trim:
        df = trim_ends(df)
    if continuous:
        df = keep_longest_continuous(df)
    return df, has_labels


def window_table(df, window_m=5):
    """5 m windows with the features the final model needs. Labels are added only if present."""
    df = df.copy()
    # Cut into 5 m windows, drop incomplete ones, compute features
    df["window"] = (df["Loc(m)"] // window_m).astype(int)

    rows = []
    for w, g in df.groupby("window"):
        length = g["Loc(m)"].max() - g["Loc(m)"].min()
        if length < window_m * 0.9:
            continue
        row = {"window": w, "start_m": g["Loc(m)"].min(), "speed_kmh": g["Speed_kmh"].mean()}
        for col in ["GyrY", "AccZ", "GyrX", "GyrZ", "AccY", "AccX", "Pitch(deg)"]:
            name = col.split("(")[0]
            row[name + "_std"] = g[col].std()
        row["Pitch_mean"] = g["Pitch(deg)"].mean()
        if "East(m)" in g.columns:
            row["east"] = g["East(m)"].mean()
            row["north"] = g["North(m)"].mean()
        if "muu" in g.columns:
            row["heave_frac"] = g["muu"].isin([1, 2]).mean()
        rows.append(row)

    w = pd.DataFrame(rows)
    if "heave_frac" in w.columns:
        w["label"] = w["heave_frac"].apply(label_window)
    return w


def add_model_features(w):
    """Relative std, pitch tilt, and their neighbours ±1 and ±2 windows (as in notebook 10). One drive at a time."""
    w = w.sort_values("window").copy()

    for col in STD_FEATURES:
        w[col + "_rel"] = w[col] / w[col].quantile(0.25)
    w["Pitch_tilt"] = w["Pitch_mean"] - w["Pitch_mean"].median()

    cols = [col + "_rel" for col in STD_FEATURES] + ["Pitch_tilt"]
    for shift, name in [(1, "_prev"), (-1, "_next"), (2, "_prev2"), (-2, "_next2")]:
        other = w[["window"] + cols].copy()
        other["window"] = other["window"] + shift
        other = other.rename(columns={col: col + name for col in cols})
        w = w.merge(other, on="window", how="left")
        for col in cols:
            w[col + name] = w[col + name].fillna(w[col])
    return w

def spots_at_goal(windows, prob, goal=0.9):
    """False alarm spots per km at the best threshold that still finds `goal` of the bumps.
    Returns 100 if no threshold reaches the goal (a penalty)."""
    p = windows[["drive", "window", "is_bump"]].copy()
    p["prob"] = prob
    best = 100.0
    for t in np.arange(0.05, 1.0, 0.05):
        p["pred"] = (p["prob"] >= t).astype(int)
        b = bump_level(p)
        if b["detection_rate"] >= goal and b["spots_per_km"] < best:
            best = b["spots_per_km"]
    return best

def threshold_at_goal(windows, prob, goal=0.9):
    """The threshold with the fewest false alarm spots that still finds `goal` of the bumps (0.05 if none does)."""
    p = windows[["drive", "window", "is_bump"]].copy()
    p["prob"] = prob
    best_spots = None
    best_t = 0.05
    for t in np.arange(0.05, 1.0, 0.05):
        p["pred"] = (p["prob"] >= t).astype(int)
        b = bump_level(p)
        if b["detection_rate"] >= goal and (best_spots is None or b["spots_per_km"] < best_spots):
            best_spots = b["spots_per_km"]
            best_t = t
    return best_t

def detect_separator(first_line):
    """Guess the column separator from the header line: tab, semicolon or comma."""
    counts = {"\t": first_line.count("\t"), ";": first_line.count(";"), ",": first_line.count(",")}
    return max(counts, key=counts.get)