import pandas as pd
from matplotlib import pyplot as plt

LABELS = ["posneg", "vaikeus", "muu"]

def fix_transitions(s, settle=3, tol=0.1):
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


def load_clean(path, trim=True):
    # 1. Read the tab-separated file as text, fix decimal commas, turn into numbers
    df = pd.read_csv(path, sep="\t", dtype=str)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col].str.replace(",", "."), errors="coerce")

    # 2. Fix label transitions
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