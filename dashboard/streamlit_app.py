from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PLOTS = REPORTS / "plots"
TABLES = REPORTS / "tables"
PREDICTIONS = REPORTS / "predictions"


st.set_page_config(
    page_title="ST-GNN Weather Forecasting",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(47,111,115,0.20), transparent 32rem),
            linear-gradient(135deg, #f8f2e8 0%, #eef4f1 48%, #f8fafc 100%);
        color: #172a3a;
    }
    .hero {
        padding: 2rem 2.4rem;
        border-radius: 1.4rem;
        background: linear-gradient(120deg, #172a3a 0%, #2f6f73 70%, #d88745 100%);
        color: white;
        box-shadow: 0 18px 50px rgba(23, 42, 58, 0.18);
    }
    .hero h1 {
        font-size: 2.5rem;
        margin-bottom: 0.4rem;
    }
    .card {
        padding: 1.2rem 1.3rem;
        border-radius: 1rem;
        background: rgba(255,255,255,0.78);
        border: 1px solid rgba(23,42,58,0.08);
    }
    .small-note {
        color: #52616b;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def show_plot(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Missing `{path.relative_to(ROOT)}`. Run `python -m src.evaluate` to generate it.")


def drop_streamlit_only_columns(df: pd.DataFrame) -> pd.DataFrame:
    
    blocked_columns = {
        "within_2c_accuracy_pct",
        "overall_within_2c_accuracy_pct",
    }
    keep_cols = [col for col in df.columns if col not in blocked_columns]
    return df[keep_cols]


def show_station_map(station_df: pd.DataFrame) -> None:
    station_coords = {
        "Atlanta": (33.7490, -84.3880),
        "Decatur": (33.7748, -84.2963),
        "Marietta": (33.9526, -84.5499),
        "Sandy_Springs": (33.9304, -84.3733),
        "Roswell": (34.0232, -84.3616),
        "Alpharetta": (34.0754, -84.2941),
        "Peachtree_City": (33.3968, -84.5958),
        "Lawrenceville": (33.9562, -83.9879),
    }
    map_df = station_df.copy()
    map_df["latitude"] = map_df["station"].map(lambda station: station_coords.get(station, (None, None))[0])
    map_df["longitude"] = map_df["station"].map(lambda station: station_coords.get(station, (None, None))[1])
    map_df = map_df.dropna(subset=["latitude", "longitude"])

    fig = px.scatter_mapbox(
        map_df,
        lat="latitude",
        lon="longitude",
        color="mae_c",
        size="mae_c",
        size_max=22,
        hover_name="station",
        hover_data={
            "mae_c": ":.2f",
            "rmse_c": ":.2f",
            "mae_1h_c": ":.2f",
            "mae_6h_c": ":.2f",
            "mae_12h_c": ":.2f",
            "mae_24h_c": ":.2f",
            "latitude": False,
            "longitude": False,
        },
        color_continuous_scale="YlOrRd",
        center={"lat": 33.7490, "lon": -84.3880},
        zoom=8,
        mapbox_style="open-street-map",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=420)
    st.plotly_chart(fig, use_container_width=True)


st.markdown(
    """
    <div class="hero">
        <h1>Regional Weather Forecasting with an ST-GNN</h1>
        <p>
            A reproducible deep learning and DLOps pipeline for multi-station, multi-horizon
            temperature forecasting across the Atlanta region.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

horizon_path = TABLES / "horizon_metrics.csv"
station_path = TABLES / "station_metrics.csv"
baseline_path = TABLES / "baseline_comparison.csv"
prediction_path = PREDICTIONS / "sample_predictions.csv"
metrics_json_path = REPORTS / "metrics.json"

if horizon_path.exists():
    horizon_metrics = read_csv(horizon_path)
    mean_mae = horizon_metrics["mae_c"].mean()
    mean_rmse = horizon_metrics["rmse_c"].mean()
    best_row = horizon_metrics.sort_values("mae_c").iloc[0]
    station_metrics_top = read_csv(station_path) if station_path.exists() else None

 
results_tab, = st.tabs(["Results"])

with results_tab:
    left, right = st.columns([1.1, 0.9])
    with left:
        st.markdown(
            """
            ### Methodology
            This project uses a Spatial-Temporal Graph Neural Network (ST-GNN) to jointly model spatial dependencies between weather stations and temporal dynamics over time. Unlike LSTM-based models that treat each station independently, the ST-GNN captures how nearby locations influence each other, enabling it to learn regional weather patterns. This leads to improved accuracy and better generalization across multiple forecast horizons.

            ### Forecast Task
            The network receives a 24-hour weather window and predicts temperature at 1h, 6h,
            12h, and 24h horizons for every station.
            """
        )
    with right:
        st.markdown(
            """
            <div class="card">
            <b>Outline</b><br>
            The project compares forecast quality across horizons and stations, reports metrics
            in physical units, and keeps the pipeline reproducible with DVC, MLflow, and Airflow.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Model Comparison")
    if baseline_path.exists():
        st.dataframe(
            drop_streamlit_only_columns(read_csv(baseline_path)),
            use_container_width=True,
            hide_index=True,
        )
        show_plot(PLOTS / "baseline_comparison.png", "ST-GNN vs persistence and LSTM-only baselines")

    st.subheader("Forecast Metrics")
    if horizon_path.exists():
        st.dataframe(
            drop_streamlit_only_columns(read_csv(horizon_path)),
            use_container_width=True,
            hide_index=True,
        )
    if station_path.exists():
        station_df = read_csv(station_path)
        st.subheader("Weather Station Error Map")
        show_station_map(station_df)
        st.dataframe(
            drop_streamlit_only_columns(station_df),
            use_container_width=True,
            hide_index=True,
        )

    p1, p2 = st.columns(2)
    with p1:
        show_plot(PLOTS / "horizon_metrics.png", "MAE/RMSE by forecast horizon")
        show_plot(PLOTS / "training_loss_curve.png", "Training and validation loss")
    with p2:
        show_plot(PLOTS / "actual_vs_predicted.png", "Actual vs predicted temperature traces")

    st.subheader("Sample Predictions")
    if prediction_path.exists():
        st.dataframe(read_csv(prediction_path).head(40), use_container_width=True, hide_index=True)
    else:
        st.info("Sample predictions will appear after evaluation is run.")
