import io
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import streamlit as st
import pydeck as pdk
from pyproj import Transformer
from data_utils import bump_level, load_clean_file, window_table, add_model_features

st.set_page_config(page_title="Frost heave detection", layout="wide")


# ---------- Loading and processing (done once, then remembered) ----------
@st.cache_data
def load_predictions():
    return pd.read_csv("predictions/predictions_rf_final.csv")


@st.cache_resource
def load_model():
    return joblib.load("models/rf_final.joblib")

@st.cache_resource
def get_transformer():
    return Transformer.from_crs("EPSG:3067", "EPSG:4326", always_xy=True)

@st.cache_data
def process_file(file_bytes):
    raw_rows = file_bytes.count(b"\n") - 1
    df_clean, has_labels = load_clean_file(io.BytesIO(file_bytes))
    w = add_model_features(window_table(df_clean))
    return raw_rows, df_clean, w, has_labels


def marked_places(w):
    """Group neighbouring marked windows into places (from-to metres, highest probability)."""
    places = []
    current = None
    # Group neighbouring marked windows into places
    for _, row in w[w["marked"]].iterrows():
        # If the current place is not None and the current window is the next one after the last window of the current place, extend the current place
        if current is not None and row["window"] == current["last_window"] + 1:
            # Extend the current place
            current["to_m"] = row["dist_m"] + 5
            # Update the highest probability of the current place
            current["max_prob"] = max(current["max_prob"], row["prob"])
            # Update the last window of the current place
            current["last_window"] = row["window"]
        else:
            if current is not None:
                # Append the current place to the list of places
                places.append(current)
            # Start a new place
            current = {"from_m": row["dist_m"], "to_m": row["dist_m"] + 5,
                       "max_prob": row["prob"], "last_window": row["window"]}
    if current is not None:
        places.append(current)
    return pd.DataFrame(places)


pred = load_predictions()
saved = load_model()


# ---------- Sidebar: sensitivity ----------
SENSITIVITY = {
    "Very careful": 0.80,
    "Careful": 0.70,
    "Balanced (recommended)": saved["threshold"],
    "Sensitive": 0.45,
    "Very sensitive": 0.30,
}

st.sidebar.header("Settings")
level = st.sidebar.select_slider("Sensitivity", options=list(SENSITIVITY.keys()),
                                 value="Balanced (recommended)")
threshold = SENSITIVITY[level]
st.sidebar.caption(
    "More sensitive: finds more frost heaves, but also marks more places that turn out to be fine. "
    "More careful: fewer false alarms, but some frost heaves may be missed."
)

show_map = st.sidebar.checkbox("Show map", value=True)


# ---------- Page header and tabs ----------
st.title("Frost heave detection")
st.caption("Automatic detection of frost heaves from car sensor data")

tab_check, tab_results = st.tabs(["Check a drive", "Test results"])


# ---------- Tab 1: Check a drive ----------
with tab_check:
    uploaded = st.file_uploader("Upload a drive file (tab-separated .txt, same format as the project data)",
                                type=["txt"])

    if uploaded is None:
        st.info("Upload a drive file to start.")
    else:
        with st.spinner("Cleaning the data and checking the road..."):
            raw_rows, df_clean, w, has_labels = process_file(uploaded.getvalue())

        # Choose the model: for project roads, the version that never saw that road
        road = uploaded.name.split("_")[0]

        if road in saved["road_models"]:
            model = saved["road_models"][road]
            st.caption(f"This road ({road}) was part of the project data, so a version of the "
                       f"system that never saw this road is used.")
        else:
            model = saved["model"]

        # Score every 5 m of road
        w = w.copy()
        w["prob"] = model.predict_proba(w[saved["features"]])[:, 1]
        w["marked"] = w["prob"] >= threshold
        w["dist_m"] = w["start_m"] - w["start_m"].min()
        places = marked_places(w)

        # ---- 1. The answer ----
        st.subheader(f"The system marked {len(places)} places on this drive.")
        if has_labels:
            known = w[w["label"] != "left out"].copy()
            known["drive"] = uploaded.name
            known["is_bump"] = (known["label"] == "bump").astype(int)
            known["pred"] = known["marked"].astype(int)
            r = bump_level(known)
            false_places = round(r["spots_per_km"] * len(known) * 5 / 1000)

            col1, col2, col3 = st.columns(3)
            col1.metric("Real frost heaves (marked by an expert)", r["bumps_total"])
            col2.metric("Found by the system", r["bumps_found"])
            col3.metric("Marked places that were not frost heaves", false_places)
        else:
            st.caption("This file has no expert markings, so the marked places cannot be checked here.")

                # ---- 2. The road: map and graph ----
        if show_map and "east" in w.columns:
            view_map, view_graph = st.tabs(["Map", "Graph"])
        else:
            view_map = None
            view_graph = st.container()

        if view_map is not None:
            with view_map:
                # Convert the Finnish coordinates to latitude/longitude
                lon, lat = get_transformer().transform(w["east"].to_numpy(), w["north"].to_numpy())
                map_df = pd.DataFrame({
                    "lon": lon,
                    "lat": lat,
                    "marked": w["marked"].to_numpy(),
                    "dist_text": w["dist_m"].round(0).astype(int).astype(str).to_numpy(),
                    "prob_text": (w["prob"] * 100).round(0).astype(int).astype(str).to_numpy(),
                })
                if has_labels:
                    map_df["real"] = (w["label"] == "bump").to_numpy()
                map_df = map_df.dropna(subset=["lon", "lat"])

                # Layers: the whole road (grey), real frost heaves (green), marked places (orange)
                layers = [pdk.Layer("ScatterplotLayer", data=map_df, get_position="[lon, lat]",
                                    get_fill_color=[110, 110, 110, 140], get_radius=3,
                                    radius_min_pixels=2, pickable=True)]
                if has_labels:
                    layers.append(pdk.Layer("ScatterplotLayer", data=map_df[map_df["real"]],
                                            get_position="[lon, lat]", get_fill_color=[0, 150, 0, 200],
                                            get_radius=9, radius_min_pixels=6, pickable=True))
                layers.append(pdk.Layer("ScatterplotLayer", data=map_df[map_df["marked"]],
                                        get_position="[lon, lat]", get_fill_color=[255, 140, 0, 230],
                                        get_radius=5, radius_min_pixels=4, pickable=True))

                view = pdk.ViewState(latitude=map_df["lat"].mean(), longitude=map_df["lon"].mean(), zoom=14)
                st.pydeck_chart(pdk.Deck(
                    layers=layers,
                    initial_view_state=view,
                    map_provider="carto",
                    map_style="https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
                    tooltip={"text": "{dist_text} m from start\nProbability: {prob_text} %"},
                ))
                legend = "⚪ grey: the drive · 🟠 orange: marked by the system"
                if has_labels:
                    legend = legend + " · 🟢 green: real frost heave (expert)"
                st.caption(legend)

        with view_graph:
            fig, ax = plt.subplots(figsize=(12, 3.5))
            ax.plot(w["dist_m"], w["prob"], color="tab:blue")
            ax.axhline(threshold, color="grey", linestyle="--")
            for _, row in w[w["marked"]].iterrows():
                ax.axvspan(row["dist_m"], row["dist_m"] + 5, color="orange", alpha=0.3)
            if has_labels:
                for _, row in w[w["label"] == "bump"].iterrows():
                    ax.axvspan(row["dist_m"], row["dist_m"] + 5, ymin=0, ymax=0.06, color="green")

            handles = [Line2D([0], [0], color="tab:blue", label="Frost heave probability"),
                       Line2D([0], [0], color="grey", linestyle="--", label="Marking line (sensitivity)"),
                       Patch(color="orange", alpha=0.3, label="Marked by the system")]
            if has_labels:
                handles.append(Patch(color="green", label="Real frost heave (expert)"))
            ax.legend(handles=handles, loc="upper right")
            ax.set_xlabel("Distance along the drive (m)")
            ax.set_ylabel("Probability")
            ax.set_ylim(0, 1)
            st.pyplot(fig)

        # ---- 3. The marked places ----
        if len(places) > 0:
            st.markdown("**Marked places**")
            table = pd.DataFrame({
                "Place": range(1, len(places) + 1),
                "From (m)": places["from_m"].round(0).astype(int),
                "To (m)": places["to_m"].round(0).astype(int),
                "Highest probability": (places["max_prob"] * 100).round(0).astype(int).astype(str) + " %",
            })
            st.dataframe(table, hide_index=True)

        # ---- 4. The cleaned data ----
        with st.expander("Cleaned data"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows in the file", f"{raw_rows:,}")
            c2.metric("Rows after cleaning", f"{len(df_clean):,}")
            length_km = (df_clean["Loc(m)"].max() - df_clean["Loc(m)"].min()) / 1000
            c3.metric("Length of the drive", f"{length_km:.2f} km")
            st.caption("Cleaning: decimal commas fixed, slow start and end of the drive removed, "
                       "gaps in the recording removed.")
            st.dataframe(df_clean.head(500))
            st.download_button("Download the cleaned file",
                               df_clean.to_csv(index=False).encode("utf-8"),
                               file_name=uploaded.name.replace(".txt", "_clean.csv"),
                               mime="text/csv")


# ---------- Tab 2: Test results (the project's overall result) ----------
with tab_results:
    p = pred.copy()
    p["pred"] = (p["prob"] >= threshold).astype(int)
    result = bump_level(p)

    st.subheader(
        f"In testing, the system found {result['bumps_found']} of {result['bumps_total']} frost heaves "
        f"({result['detection_rate']:.0%}) on roads it had never seen before."
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Frost heaves found", f"{result['bumps_found']} of {result['bumps_total']}")
    col2.metric("Share found", f"{result['detection_rate']:.0%}")
    col3.metric("False alarms per km", f"{result['spots_per_km']:.1f}",
                help="Places marked by the system where there was no frost heave. Neighbouring marks count as one place.")

    st.success(
        f"What this means: instead of inspecting every metre of road, an inspector can focus on the "
        f"places the system marks. They include {result['detection_rate']:.0%} of the frost heaves, "
        f"with about {result['spots_per_km']:.1f} false alarms per kilometre."
    )
    st.markdown(
        f"**Tested on:** {pred['road'].nunique()} roads, {pred['drive'].nunique()} drives, "
        f"{len(pred) * 5 / 1000:.1f} km of driving. Each road was checked by a version of the system "
        f"that had never seen that road."
    )
    st.caption(
        "Limits: tested on 3 paved roads, with one car and one type of sensor, at about 40–80 km/h, "
        "on a single day. Results on other roads, cars or seasons may differ."
    )