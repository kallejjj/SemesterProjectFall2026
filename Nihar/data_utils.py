import pandas as pd

LABELS = ["posneg", "vaikeus", "muu"]

def load_clean(path):
    # 1. Read the tab-separated file as text, fix decimal commas, turn into numbers
    df = pd.read_csv(path, sep="\t", dtype=str)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col].str.replace(",", "."), errors="coerce")

    # 2. Fix blended labels: replace them with the nearest real label
    blended = (df[LABELS] % 1 != 0).any(axis=1)
    df.loc[blended, LABELS] = None
    df[LABELS] = df[LABELS].interpolate(method="nearest").ffill().bfill()
    df[LABELS] = df[LABELS].astype(int)

    # 3. Speed in km/h, drop the row counter
    df["Speed_kmh"] = df["Speed (m/s)"] * 3.6
    df = df.drop(columns=["Num"])

    print(f"{path}: {len(df)} rows, fixed {blended.sum()} blended rows")
    return df
