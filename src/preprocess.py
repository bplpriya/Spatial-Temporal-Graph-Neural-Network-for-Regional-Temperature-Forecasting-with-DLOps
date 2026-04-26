from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.graph import build_adjacency_matrix, normalize_adjacency
from src.utils import ensure_dir, load_yaml


def create_windows(data: np.ndarray, input_window: int, horizons: list[int], target_feature_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert a station-feature tensor into supervised ST-GNN windows.

    Parameters
    ----------
    data:
        Array shaped as ``[time, stations, features]`` after scaling.
    input_window:
        Number of historical hourly steps shown to the model.
    horizons:
        Forecast offsets, in hours, to predict from the end of each window.
    target_feature_idx:
        Feature index of the prediction target in ``data``.
    """
    max_h = max(horizons)
    xs, ys = [], []
    total_steps = data.shape[0]
    for t in range(input_window, total_steps - max_h):
        x = data[t - input_window:t]
        y = []
        for h in horizons:
            y.append(data[t + h, :, target_feature_idx])
        y = np.stack(y, axis=0)
        xs.append(x)
        ys.append(y)
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def create_target_times(times: np.ndarray, input_window: int, horizons: list[int]) -> np.ndarray:
    """Return timestamps that align with each generated target horizon."""
    max_h = max(horizons)
    target_times = []
    for t in range(input_window, len(times) - max_h):
        target_times.append([times[t + h] for h in horizons])
    return np.array(target_times, dtype="datetime64[ns]")


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add periodic time signals that help the model learn daily/weekly seasonality."""
    enriched = df.copy()
    hour = enriched["time"].dt.hour
    day_of_week = enriched["time"].dt.dayofweek
    day_of_year = enriched["time"].dt.dayofyear

    enriched["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    enriched["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    enriched["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    enriched["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    enriched["doy_sin"] = np.sin(2 * np.pi * day_of_year / 366)
    enriched["doy_cos"] = np.cos(2 * np.pi * day_of_year / 366)
    return enriched


def main() -> None:
    ensure_dir("data/processed")
    params = load_yaml("params.yaml")
    stations_cfg = load_yaml("configs/stations.yaml")

    df = pd.read_csv("data/raw/weather_raw.csv", parse_dates=["time"])
    weather_feature_cols = params["data"]["feature_cols"]
    feature_cols = weather_feature_cols + params["data"].get("time_features", [])
    target_col = params["data"]["target_col"]

    df = df.sort_values(["time", "station"]).copy()
    df[weather_feature_cols] = df.groupby("station")[weather_feature_cols].ffill().bfill()
    df = add_cyclical_time_features(df)

    station_names = [s["name"] for s in stations_cfg["stations"]]
    reference_times = (
        df[df["station"] == station_names[0]]
        .sort_values("time")["time"]
        .to_numpy(dtype="datetime64[ns]")
    )
    pivot_frames = []
    for station in station_names:
        station_df = df[df["station"] == station].sort_values("time")
        pivot_frames.append(station_df[feature_cols].to_numpy())

    data = np.stack(pivot_frames, axis=1)
    t, n, f = data.shape
    flat = data.reshape(-1, f)

    scaler = StandardScaler()
    flat_scaled = scaler.fit_transform(flat)
    data_scaled = flat_scaled.reshape(t, n, f)

    target_idx = feature_cols.index(target_col)
    x, y = create_windows(
        data=data_scaled,
        input_window=params["data"]["input_window"],
        horizons=params["data"]["horizons"],
        target_feature_idx=target_idx,
    )
    target_times = create_target_times(
        reference_times,
        input_window=params["data"]["input_window"],
        horizons=params["data"]["horizons"],
    )

    total = len(x)
    train_end = int(total * params["data"]["train_ratio"])
    val_end = int(total * (params["data"]["train_ratio"] + params["data"]["val_ratio"]))

    stations = stations_cfg["stations"]
    adj = build_adjacency_matrix(
        stations=stations,
        threshold_km=params["graph"]["distance_threshold_km"],
        method=params["graph"].get("method", "threshold"),
        k_neighbors=params["graph"].get("k_neighbors", 3),
        distance_sigma_km=params["graph"].get("distance_sigma_km", 45.0),
    )
    adj_norm = normalize_adjacency(adj)

    np.savez_compressed(
        "data/processed/dataset.npz",
        x_train=x[:train_end],
        y_train=y[:train_end],
        x_val=x[train_end:val_end],
        y_val=y[train_end:val_end],
        x_test=x[val_end:],
        y_test=y[val_end:],
        test_target_times=target_times[val_end:],
        station_names=np.array(station_names),
        feature_names=np.array(feature_cols),
        horizons=np.array(params["data"]["horizons"], dtype=np.int32),
        input_window=np.array(params["data"]["input_window"], dtype=np.int32),
        adj=adj_norm.astype(np.float32),
        adj_raw=adj.astype(np.float32),
    )
    joblib.dump(scaler, "data/processed/scaler.joblib")
    print("Saved processed dataset and scaler.")


if __name__ == "__main__":
    main()
