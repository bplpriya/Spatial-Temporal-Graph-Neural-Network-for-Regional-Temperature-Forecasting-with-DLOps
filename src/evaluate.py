from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.model import LSTMBaseline, STGNN
from src.utils import ensure_dir, get_device, load_yaml


REPORTS_DIR = Path("reports")
PLOTS_DIR = REPORTS_DIR / "plots"
TABLES_DIR = REPORTS_DIR / "tables"
PREDICTIONS_DIR = REPORTS_DIR / "predictions"
MODELS_DIR = Path("models")


def anomaly_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Measure whether predictions follow the same anomaly pattern as truth."""
    true_anom = y_true - np.mean(y_true)
    pred_anom = y_pred - np.mean(y_pred)
    numerator = np.sum(true_anom * pred_anom)
    denominator = np.sqrt(np.sum(true_anom**2) * np.sum(pred_anom**2))
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def inverse_target_scale(values: np.ndarray, scaler, target_idx: int) -> np.ndarray:
    """Convert normalized target values back to original target units."""
    return values * scaler.scale_[target_idx] + scaler.mean_[target_idx]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def get_station_names(ds: np.lib.npyio.NpzFile) -> list[str]:
    if "station_names" in ds.files:
        return [str(name) for name in ds["station_names"]]
    stations_cfg = load_yaml("configs/stations.yaml")
    configured_names = [station["name"] for station in stations_cfg["stations"]]
    if len(configured_names) == ds["y_test"].shape[-1]:
        return configured_names
    return [f"station_{idx + 1}" for idx in range(ds["y_test"].shape[-1])]


def get_target_times(ds: np.lib.npyio.NpzFile, sample_count: int, horizons: list[int]) -> np.ndarray:
    if "test_target_times" in ds.files:
        return ds["test_target_times"]
    fallback = np.arange(sample_count)
    return np.repeat(fallback[:, None], len(horizons), axis=1)


def build_prediction_table(
    y_true_c: np.ndarray,
    preds_c: np.ndarray,
    target_times: np.ndarray,
    horizons: list[int],
    station_names: list[str],
) -> pd.DataFrame:
    """Create a long-form predictions table for dashboards and poster examples."""
    records = []
    sample_count, horizon_count, station_count = y_true_c.shape
    for sample_idx in range(sample_count):
        for horizon_idx in range(horizon_count):
            for station_idx in range(station_count):
                actual = float(y_true_c[sample_idx, horizon_idx, station_idx])
                predicted = float(preds_c[sample_idx, horizon_idx, station_idx])
                records.append(
                    {
                        "sample_index": sample_idx,
                        "target_time": str(target_times[sample_idx, horizon_idx]),
                        "horizon_hours": horizons[horizon_idx],
                        "station": station_names[station_idx],
                        "actual_temperature_c": actual,
                        "predicted_temperature_c": predicted,
                        "absolute_error_c": abs(actual - predicted),
                    }
                )
    return pd.DataFrame.from_records(records)


def compute_horizon_metrics(
    y_true_scaled: np.ndarray,
    preds_scaled: np.ndarray,
    y_true_c: np.ndarray,
    preds_c: np.ndarray,
    horizons: list[int],
) -> pd.DataFrame:
    rows = []
    for idx, horizon in enumerate(horizons):
        yt_scaled = y_true_scaled[:, idx, :].reshape(-1)
        yp_scaled = preds_scaled[:, idx, :].reshape(-1)
        yt_c = y_true_c[:, idx, :].reshape(-1)
        yp_c = preds_c[:, idx, :].reshape(-1)
        rows.append(
            {
                "horizon_hours": horizon,
                "mae_c": float(mean_absolute_error(yt_c, yp_c)),
                "rmse_c": rmse(yt_c, yp_c),
                "mae_normalized": float(mean_absolute_error(yt_scaled, yp_scaled)),
                "rmse_normalized": rmse(yt_scaled, yp_scaled),
                "anomaly_correlation": anomaly_correlation_coefficient(yt_scaled, yp_scaled),
            }
        )
    return pd.DataFrame(rows)


def compute_model_horizon_metrics(
    model_name: str,
    y_true_scaled: np.ndarray,
    preds_scaled: np.ndarray,
    y_true_c: np.ndarray,
    preds_c: np.ndarray,
    horizons: list[int],
) -> pd.DataFrame:
    metrics = compute_horizon_metrics(y_true_scaled, preds_scaled, y_true_c, preds_c, horizons)
    metrics.insert(0, "model", model_name)
    return metrics


def compute_station_metrics(
    y_true_c: np.ndarray,
    preds_c: np.ndarray,
    horizons: list[int],
    station_names: list[str],
) -> pd.DataFrame:
    rows = []
    for station_idx, station in enumerate(station_names):
        yt_station = y_true_c[:, :, station_idx].reshape(-1)
        yp_station = preds_c[:, :, station_idx].reshape(-1)
        row = {
            "station": station,
            "mae_c": float(mean_absolute_error(yt_station, yp_station)),
            "rmse_c": rmse(yt_station, yp_station),
        }
        for horizon_idx, horizon in enumerate(horizons):
            yt = y_true_c[:, horizon_idx, station_idx]
            yp = preds_c[:, horizon_idx, station_idx]
            row[f"mae_{horizon}h_c"] = float(mean_absolute_error(yt, yp))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mae_c")


def save_metrics_json(horizon_metrics: pd.DataFrame, station_metrics: pd.DataFrame) -> None:
    """Save DVC-friendly metrics while keeping units explicit."""
    metrics = {
        f"horizon_{int(row.horizon_hours)}h": {
            "mae_c": float(row.mae_c),
            "rmse_c": float(row.rmse_c),
            "mae_normalized": float(row.mae_normalized),
            "rmse_normalized": float(row.rmse_normalized),
            "anomaly_correlation": float(row.anomaly_correlation),
        }
        for row in horizon_metrics.itertuples(index=False)
    }
    metrics["overall"] = {
        "mean_horizon_mae_c": float(horizon_metrics["mae_c"].mean()),
        "mean_horizon_rmse_c": float(horizon_metrics["rmse_c"].mean()),
        "mean_anomaly_correlation": float(horizon_metrics["anomaly_correlation"].mean()),
        "best_station_mae_c": float(station_metrics["mae_c"].min()),
        "worst_station_mae_c": float(station_metrics["mae_c"].max()),
    }
    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def load_best_config(params: dict) -> dict:
    config_path = MODELS_DIR / "best_config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "input_window": params["data"]["input_window"],
        "hidden_dim": params["model"]["hidden_dim"],
        "dropout": params["model"]["dropout"],
        "learning_rate": params["train"]["learning_rate"],
        "batch_size": params["train"]["batch_size"],
        "run_name": "configured_model",
    }


def crop_window(x: np.ndarray, input_window: int) -> np.ndarray:
    return x[:, -int(input_window) :, :, :]


def predict_stgnn(
    params: dict,
    best_config: dict,
    x_test: np.ndarray,
    adj: torch.Tensor,
    device: torch.device,
    target_idx: int,
) -> np.ndarray:
    model = STGNN(
        num_features=x_test.shape[-1],
        num_horizons=len(params["data"]["horizons"]),
        hidden_dim=int(best_config["hidden_dim"]),
        gcn_out_dim=int(best_config["hidden_dim"]),
        temporal_hidden_dim=int(best_config["hidden_dim"]),
        temporal_layers=int(params["model"].get("temporal_layers", 2)),
        dropout=float(best_config["dropout"]),
        target_feature_idx=target_idx,
        predict_residual=bool(params["model"].get("predict_residual", True)),
    ).to(device)
    model.load_state_dict(torch.load(MODELS_DIR / "best_model.pt", map_location=device))
    model.eval()
    xb = torch.tensor(crop_window(x_test, best_config["input_window"]), dtype=torch.float32).to(device)
    with torch.no_grad():
        mean, _ = model(xb, adj.to(device))
    return mean.cpu().numpy()


def predict_lstm(params: dict, best_config: dict, x_test: np.ndarray, device: torch.device) -> np.ndarray | None:
    model_path = MODELS_DIR / "lstm_baseline.pt"
    if not model_path.exists():
        return None
    model = LSTMBaseline(
        num_features=x_test.shape[-1],
        num_horizons=len(params["data"]["horizons"]),
        hidden_dim=int(best_config["hidden_dim"]),
        dropout=float(best_config["dropout"]),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    xb = torch.tensor(crop_window(x_test, best_config["input_window"]), dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(xb)
    return pred.cpu().numpy()


def predict_persistence(x_test: np.ndarray, horizons: list[int], target_idx: int) -> np.ndarray:
    last_observed = x_test[:, -1, :, target_idx]
    return np.repeat(last_observed[:, None, :], len(horizons), axis=1)


def plot_horizon_metrics(horizon_metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    x = np.arange(len(horizon_metrics))
    width = 0.36
    ax.bar(x - width / 2, horizon_metrics["mae_c"], width, label="MAE", color="#2f6f73")
    ax.bar(x + width / 2, horizon_metrics["rmse_c"], width, label="RMSE", color="#d88745")
    ax.set_title("Forecast Error by Horizon", fontsize=15, weight="bold")
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Temperature error (deg C)")
    ax.set_xticks(x, [f"{int(h)}h" for h in horizon_metrics["horizon_hours"]])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "horizon_metrics.png", bbox_inches="tight")
    plt.close(fig)


def plot_baseline_comparison(baseline_comparison: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 3), dpi=160)
    sorted_df = baseline_comparison.sort_values("overall_mae_c")
    ax.bar(sorted_df["model"], sorted_df["overall_mae_c"], color=["#2f6f73", "#d88745", "#4062bb"][: len(sorted_df)])
    ax.set_title("Model Comparison on Test Set", fontsize=15, weight="bold")
    ax.set_ylabel("Overall MAE (deg C)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "baseline_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_station_metrics(station_metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    ax.bar(station_metrics["station"], station_metrics["mae_c"], color="#4062bb")
    ax.set_title("Station-Level MAE", fontsize=15, weight="bold")
    ax.set_ylabel("MAE (deg C)")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "station_metrics.png", bbox_inches="tight")
    plt.close(fig)


def plot_actual_vs_predicted(predictions: pd.DataFrame, station: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=160, sharex=True)
    axes = axes.flatten()
    for ax, (horizon, group) in zip(axes, predictions[predictions["station"] == station].groupby("horizon_hours")):
        subset = group.sort_values("sample_index").head(120)
        ax.plot(subset["sample_index"], subset["actual_temperature_c"], label="Actual", color="#1f2933", linewidth=1.8)
        ax.plot(subset["sample_index"], subset["predicted_temperature_c"], label="Predicted", color="#e85d75", linewidth=1.6)
        ax.set_title(f"{station}: {int(horizon)}h forecast")
        ax.set_ylabel("deg C")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Test sample")
    axes[-2].set_xlabel("Test sample")
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Actual vs Predicted Temperature", fontsize=16, weight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "actual_vs_predicted.png", bbox_inches="tight")
    plt.close(fig)


def plot_training_history() -> None:
    history_path = REPORTS_DIR / "training_history.csv"
    if not history_path.exists():
        return
    history = pd.read_csv(history_path)
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=160)
    train_col = "train_loss" if "train_loss" in history.columns else "train_nll"
    val_col = "val_loss" if "val_loss" in history.columns else "val_nll"
    ax.plot(history["epoch"], history[train_col], marker="o", label="Train loss", color="#264653")
    ax.plot(history["epoch"], history[val_col], marker="o", label="Validation loss", color="#f4a261")
    ax.set_title("Training vs Validation Loss", fontsize=15, weight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training objective")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "training_loss_curve.png", bbox_inches="tight")
    plt.close(fig)


def write_best_run_summary(best_config: dict, horizon_metrics: pd.DataFrame, baseline_comparison: pd.DataFrame) -> None:
    best_model = baseline_comparison.sort_values("overall_mae_c").iloc[0]
    stgnn = baseline_comparison[baseline_comparison["model"] == "ST-GNN"].iloc[0]
    persistence = baseline_comparison[baseline_comparison["model"] == "Persistence"].iloc[0]
    relative_gain = 100 * (persistence["overall_mae_c"] - stgnn["overall_mae_c"]) / persistence["overall_mae_c"]
    lines = [
        "# Best Run Summary",
        "",
        "## Selected ST-GNN Configuration",
        f"- Run: {best_config.get('run_name', 'configured_model')}",
        f"- Input window: {int(best_config['input_window'])} hours",
        f"- Hidden dimension: {int(best_config['hidden_dim'])}",
        f"- Dropout: {float(best_config['dropout'])}",
        f"- Learning rate: {float(best_config['learning_rate'])}",
        f"- Batch size: {int(best_config['batch_size'])}",
        f"- Best validation MAE, normalized: {float(best_config.get('best_val_mae_normalized', np.nan)):.4f}",
        "",
        "## Test Results",
        f"- Best overall model by MAE: {best_model['model']} ({best_model['overall_mae_c']:.3f} deg C)",
        f"- ST-GNN overall MAE: {stgnn['overall_mae_c']:.3f} deg C",
        f"- ST-GNN overall RMSE: {stgnn['overall_rmse_c']:.3f} deg C",
        f"- ST-GNN anomaly correlation: {stgnn['mean_anomaly_correlation']:.3f}",
        f"- Persistence overall MAE: {persistence['overall_mae_c']:.3f} deg C",
        f"- ST-GNN improvement over persistence: {relative_gain:.1f}%",
        "",
        "## Horizon Metrics for Final ST-GNN",
        horizon_metrics.round(4).to_string(index=False),
        "",
        "## Interpretation",
        "The final model uses cyclical time features, a distance-weighted kNN station graph, residual graph convolutions, and a GRU temporal encoder. This keeps the method explainable while improving validation behavior and direct MAE/RMSE performance.",
    ]
    (REPORTS_DIR / "best_run_summary.md").write_text("\n".join(lines), encoding="utf-8")


def plot_poster_summary(horizon_metrics: pd.DataFrame, station_metrics: pd.DataFrame, predictions: pd.DataFrame) -> None:
    best_station = station_metrics.iloc[0]["station"]
    sample = predictions[
        (predictions["station"] == best_station)
        & (predictions["horizon_hours"] == horizon_metrics.iloc[0]["horizon_hours"])
    ].sort_values("sample_index").head(120)

    fig = plt.figure(figsize=(13, 8), dpi=170)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 0.95])

    ax_line = fig.add_subplot(grid[0, 0])
    ax_line.plot(sample["sample_index"], sample["actual_temperature_c"], color="#172a3a", label="Actual", linewidth=1.8)
    ax_line.plot(sample["sample_index"], sample["predicted_temperature_c"], color="#ff6b35", label="Predicted", linewidth=1.6)
    ax_line.set_title(f"Best Station Trace: {best_station}", weight="bold")
    ax_line.set_ylabel("deg C")
    ax_line.grid(alpha=0.2)
    ax_line.legend(frameon=False)

    ax_horizon = fig.add_subplot(grid[0, 1])
    ax_horizon.bar(
        [f"{int(h)}h" for h in horizon_metrics["horizon_hours"]],
        horizon_metrics["mae_c"],
        color=["#2f6f73", "#3d8c8f", "#d88745", "#c8553d"],
    )
    ax_horizon.set_title("Error Growth Across Horizons", weight="bold")
    ax_horizon.set_ylabel("MAE (deg C)")
    ax_horizon.grid(axis="y", alpha=0.2)

    ax_station = fig.add_subplot(grid[1, 0])
    ax_station.barh(station_metrics["station"], station_metrics["mae_c"], color="#4062bb")
    ax_station.invert_yaxis()
    ax_station.set_title("Spatial Robustness by Station", weight="bold")
    ax_station.set_xlabel("MAE (deg C)")
    ax_station.grid(axis="x", alpha=0.2)

    ax_text = fig.add_subplot(grid[1, 1])
    ax_text.axis("off")
    mean_mae = horizon_metrics["mae_c"].mean()
    mean_acc = horizon_metrics["anomaly_correlation"].mean()
    best_horizon = horizon_metrics.sort_values("mae_c").iloc[0]
    ax_text.text(
        0.02,
        0.9,
        "ST-GNN Weather Forecasting",
        fontsize=18,
        weight="bold",
        color="#172a3a",
    )
    ax_text.text(
        0.02,
        0.66,
        "Pipeline: Open-Meteo -> preprocessing -> graph construction -> ST-GNN -> MLflow/DVC/Airflow",
        fontsize=11,
        wrap=True,
    )
    ax_text.text(
        0.02,
        0.42,
        f"Mean horizon MAE: {mean_mae:.2f} deg C\nBest horizon: {int(best_horizon.horizon_hours)}h at {best_horizon.mae_c:.2f} deg C\nMean anomaly correlation: {mean_acc:.3f}\nStations evaluated: {len(station_metrics)}",
        fontsize=13,
        linespacing=1.5,
    )
    ax_text.text(
        0.02,
        0.13,
        "Poster takeaway: the model combines temporal memory with spatial station links, then reports interpretable temperature errors.",
        fontsize=11,
        style="italic",
        wrap=True,
    )
    fig.suptitle("Regional Temperature Forecasting: Results Summary", fontsize=20, weight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "poster_summary.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dir(REPORTS_DIR)
    ensure_dir(PLOTS_DIR)
    ensure_dir(TABLES_DIR)
    ensure_dir(PREDICTIONS_DIR)

    params = load_yaml("params.yaml")
    ds = np.load("data/processed/dataset.npz", allow_pickle=True)
    x_test, y_test = ds["x_test"], ds["y_test"]
    adj = torch.tensor(ds["adj"], dtype=torch.float32)
    horizons = [int(h) for h in params["data"]["horizons"]]
    station_names = get_station_names(ds)

    device = get_device(params["train"]["device"])
    best_config = load_best_config(params)
    scaler = joblib.load("data/processed/scaler.joblib")
    feature_names = [str(name) for name in ds["feature_names"]] if "feature_names" in ds.files else params["data"]["feature_cols"]
    target_idx = feature_names.index(params["data"]["target_col"])
    preds_scaled = predict_stgnn(params, best_config, x_test, adj, device, target_idx)
    y_true_scaled = y_test
    preds_c = inverse_target_scale(preds_scaled, scaler, target_idx)
    y_true_c = inverse_target_scale(y_true_scaled, scaler, target_idx)
    target_times = get_target_times(ds, sample_count=len(x_test), horizons=horizons)

    horizon_metrics = compute_horizon_metrics(y_true_scaled, preds_scaled, y_true_c, preds_c, horizons)
    station_metrics = compute_station_metrics(y_true_c, preds_c, horizons, station_names)
    predictions = build_prediction_table(y_true_c, preds_c, target_times, horizons, station_names)

    model_metric_frames = [compute_model_horizon_metrics("ST-GNN", y_true_scaled, preds_scaled, y_true_c, preds_c, horizons)]
    persistence_scaled = predict_persistence(x_test, horizons, target_idx)
    persistence_c = inverse_target_scale(persistence_scaled, scaler, target_idx)
    model_metric_frames.append(
        compute_model_horizon_metrics("Persistence", y_true_scaled, persistence_scaled, y_true_c, persistence_c, horizons)
    )
    lstm_scaled = predict_lstm(params, best_config, x_test, device)
    if lstm_scaled is not None:
        lstm_c = inverse_target_scale(lstm_scaled, scaler, target_idx)
        model_metric_frames.append(compute_model_horizon_metrics("LSTM-only", y_true_scaled, lstm_scaled, y_true_c, lstm_c, horizons))
    all_model_horizon_metrics = pd.concat(model_metric_frames, ignore_index=True)
    baseline_comparison = (
        all_model_horizon_metrics.groupby("model", as_index=False)
        .agg(
            overall_mae_c=("mae_c", "mean"),
            overall_rmse_c=("rmse_c", "mean"),
            mean_anomaly_correlation=("anomaly_correlation", "mean"),
        )
        .sort_values("overall_mae_c")
    )

    horizon_metrics.to_csv(TABLES_DIR / "horizon_metrics.csv", index=False)
    station_metrics.to_csv(TABLES_DIR / "station_metrics.csv", index=False)
    all_model_horizon_metrics.to_csv(TABLES_DIR / "model_horizon_metrics.csv", index=False)
    baseline_comparison.to_csv(TABLES_DIR / "baseline_comparison.csv", index=False)
    predictions.to_csv(PREDICTIONS_DIR / "predictions_long.csv", index=False)
    predictions.head(200).to_csv(PREDICTIONS_DIR / "sample_predictions.csv", index=False)
    save_metrics_json(horizon_metrics, station_metrics)

    plot_horizon_metrics(horizon_metrics)
    plot_station_metrics(station_metrics)
    plot_actual_vs_predicted(predictions, station=station_names[0])
    plot_training_history()
    plot_baseline_comparison(baseline_comparison)
    plot_poster_summary(horizon_metrics, station_metrics, predictions)
    write_best_run_summary(best_config, horizon_metrics, baseline_comparison)

    print(json.dumps(json.loads((REPORTS_DIR / "metrics.json").read_text(encoding="utf-8")), indent=2))


if __name__ == "__main__":
    main()