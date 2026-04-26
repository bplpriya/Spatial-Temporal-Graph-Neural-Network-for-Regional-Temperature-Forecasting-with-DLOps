from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import WeatherDataset
from src.model import LSTMBaseline, STGNN, forecast_loss
from src.utils import ensure_dir, get_device, load_yaml, set_seed


MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
EXPERIMENTS_DIR = REPORTS_DIR / "experiments"


def crop_window(x: np.ndarray, input_window: int) -> np.ndarray:
    """Use the most recent input hours from the preprocessed maximum window."""
    if input_window > x.shape[1]:
        raise ValueError(f"Requested window {input_window}, but dataset only has {x.shape[1]} steps.")
    return x[:, -input_window:, :, :]


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(WeatherDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def deterministic_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = pred - target
    return {
        "mae": float(torch.mean(torch.abs(error)).item()),
        "rmse": float(torch.sqrt(torch.mean(error**2)).item()),
    }


def evaluate_stgnn(
    model: STGNN,
    loader: DataLoader,
    adj: torch.Tensor,
    device: torch.device,
    loss_name: str,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            mean, logvar = model(xb, adj)
            losses.append(float(forecast_loss(mean, yb, loss_name, logvar).item()))
            preds.append(mean.cpu())
            targets.append(yb.cpu())
    metrics = deterministic_metrics(torch.cat(preds), torch.cat(targets))
    metrics["loss"] = float(np.mean(losses))
    return metrics


def evaluate_lstm(
    model: LSTMBaseline,
    loader: DataLoader,
    device: torch.device,
    loss_name: str,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            losses.append(float(forecast_loss(pred, yb, loss_name).item()))
            preds.append(pred.cpu())
            targets.append(yb.cpu())
    metrics = deterministic_metrics(torch.cat(preds), torch.cat(targets))
    metrics["loss"] = float(np.mean(losses))
    return metrics


def build_stgnn(config: dict[str, Any], num_features: int, num_horizons: int, params: dict, target_idx: int) -> STGNN:
    hidden_dim = int(config["hidden_dim"])
    return STGNN(
        num_features=num_features,
        num_horizons=num_horizons,
        hidden_dim=hidden_dim,
        gcn_out_dim=hidden_dim,
        temporal_hidden_dim=hidden_dim,
        temporal_layers=int(params["model"].get("temporal_layers", 2)),
        dropout=float(config["dropout"]),
        target_feature_idx=target_idx,
        predict_residual=bool(params["model"].get("predict_residual", True)),
    )


def train_stgnn_config(
    config: dict[str, Any],
    arrays: dict[str, np.ndarray],
    adj: torch.Tensor,
    params: dict,
    device: torch.device,
    run_name: str,
    target_idx: int,
) -> tuple[dict[str, Any], list[dict[str, float]], dict[str, torch.Tensor]]:
    input_window = int(config["input_window"])
    batch_size = int(config["batch_size"])
    loss_name = params["model"].get("loss", "huber")
    train_loader = make_loader(crop_window(arrays["x_train"], input_window), arrays["y_train"], batch_size, True)
    val_loader = make_loader(crop_window(arrays["x_val"], input_window), arrays["y_val"], batch_size, False)

    model = build_stgnn(
        config=config,
        num_features=arrays["x_train"].shape[-1],
        num_horizons=arrays["y_train"].shape[1],
        params=params,
        target_idx=target_idx,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(params["train"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.6, patience=2)

    best_state = copy.deepcopy(model.state_dict())
    best_val_mae = float("inf")
    best_val_rmse = float("inf")
    no_improve_epochs = 0
    history = []

    for epoch in range(int(params["train"]["epochs"])):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            mean, logvar = model(xb, adj)
            loss = forecast_loss(mean, yb, loss_name, logvar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_metrics = evaluate_stgnn(model, val_loader, adj, device, loss_name)
        train_loss = float(np.mean(train_losses))
        scheduler.step(val_metrics["mae"])
        history_row = {
            "run_name": run_name,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
        }
        history.append(history_row)

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_val_rmse = val_metrics["rmse"]
            best_state = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= int(params["train"].get("patience", 4)):
            break

    summary = {
        **config,
        "run_name": run_name,
        "best_val_mae_normalized": best_val_mae,
        "best_val_rmse_normalized": best_val_rmse,
        "epochs_ran": len(history),
    }
    return summary, history, best_state


def train_lstm_baseline(
    best_config: dict[str, Any],
    arrays: dict[str, np.ndarray],
    params: dict,
    device: torch.device,
) -> list[dict[str, float]]:
    input_window = int(best_config["input_window"])
    batch_size = int(best_config["batch_size"])
    loss_name = params["model"].get("loss", "huber")
    train_loader = make_loader(crop_window(arrays["x_train"], input_window), arrays["y_train"], batch_size, True)
    val_loader = make_loader(crop_window(arrays["x_val"], input_window), arrays["y_val"], batch_size, False)
    model = LSTMBaseline(
        num_features=arrays["x_train"].shape[-1],
        num_horizons=arrays["y_train"].shape[1],
        hidden_dim=int(best_config["hidden_dim"]),
        dropout=float(best_config["dropout"]),
        num_layers=1,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(best_config["learning_rate"]),
        weight_decay=float(params["train"]["weight_decay"]),
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_mae = float("inf")
    history = []
    for epoch in range(int(params["train"].get("lstm_baseline_epochs", 12))):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = forecast_loss(pred, yb, loss_name)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))
        val_metrics = evaluate_lstm(model, val_loader, device, loss_name)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(train_losses)),
                "val_loss": val_metrics["loss"],
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
            }
        )
        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_state = copy.deepcopy(model.state_dict())

    torch.save(best_state, MODELS_DIR / "lstm_baseline.pt")
    return history


def default_sweep(params: dict) -> list[dict[str, Any]]:
    configured = params["train"].get("sweep", [])
    if configured:
        return configured
    return [
        {
            "input_window": params["data"]["input_window"],
            "hidden_dim": params["model"]["hidden_dim"],
            "dropout": params["model"]["dropout"],
            "learning_rate": params["train"]["learning_rate"],
            "batch_size": params["train"]["batch_size"],
        }
    ]


def main() -> None:
    params = load_yaml("params.yaml")
    set_seed(int(params["train"]["seed"]))
    ensure_dir(MODELS_DIR)
    ensure_dir(REPORTS_DIR)
    ensure_dir(EXPERIMENTS_DIR)

    ds = np.load("data/processed/dataset.npz", allow_pickle=True)
    arrays = {
        "x_train": ds["x_train"],
        "y_train": ds["y_train"],
        "x_val": ds["x_val"],
        "y_val": ds["y_val"],
    }
    feature_names = [str(name) for name in ds["feature_names"]] if "feature_names" in ds.files else params["data"]["feature_cols"]
    target_idx = feature_names.index(params["data"]["target_col"])
    adj = torch.tensor(ds["adj"], dtype=torch.float32)
    device = get_device(params["train"]["device"])
    adj = adj.to(device)

    sweep = default_sweep(params) if params["train"].get("tune", True) else [default_sweep(params)[0]]
    mlflow.set_experiment("stgnn-weather-forecasting")
    all_summaries = []
    all_histories = []
    best_summary: dict[str, Any] | None = None
    best_state: dict[str, torch.Tensor] | None = None

    with mlflow.start_run(run_name="compact-stgnn-sweep"):
        mlflow.log_params(
            {
                "candidate_runs": len(sweep),
                "max_epochs": params["train"]["epochs"],
                "loss": params["model"].get("loss", "huber"),
                "graph_method": params["graph"].get("method", "threshold"),
            }
        )
        for idx, config in enumerate(sweep, start=1):
            run_name = f"stgnn_sweep_{idx:02d}"
            print(f"Training {run_name}: {config}")
            set_seed(int(params["train"]["seed"]) + idx)
            summary, history, state = train_stgnn_config(config, arrays, adj, params, device, run_name, target_idx)
            all_summaries.append(summary)
            all_histories.extend(history)
            mlflow.log_metric(f"{run_name}_best_val_mae", summary["best_val_mae_normalized"])
            if best_summary is None or summary["best_val_mae_normalized"] < best_summary["best_val_mae_normalized"]:
                best_summary = summary
                best_state = state
            print(
                f"{run_name}: val_mae={summary['best_val_mae_normalized']:.4f}, "
                f"val_rmse={summary['best_val_rmse_normalized']:.4f}, epochs={summary['epochs_ran']}"
            )

        if best_summary is None or best_state is None:
            raise RuntimeError("No training runs completed.")

        torch.save(best_state, MODELS_DIR / "best_model.pt")
        with open(MODELS_DIR / "best_config.json", "w", encoding="utf-8") as f:
            json.dump(best_summary, f, indent=2)

        sweep_df = pd.DataFrame(all_summaries).sort_values("best_val_mae_normalized")
        sweep_df.to_csv(EXPERIMENTS_DIR / "sweep_results.csv", index=False)
        pd.DataFrame(all_histories).to_csv(EXPERIMENTS_DIR / "all_training_history.csv", index=False)
        best_history = pd.DataFrame(all_histories)
        best_history = best_history[best_history["run_name"] == best_summary["run_name"]].copy()
        best_history.to_csv(REPORTS_DIR / "training_history.csv", index=False)

        print(f"Best ST-GNN config: {best_summary}")
        lstm_history = train_lstm_baseline(best_summary, arrays, params, device)
        pd.DataFrame(lstm_history).to_csv(REPORTS_DIR / "baseline_lstm_training_history.csv", index=False)

        mlflow.log_params({f"best_{k}": v for k, v in best_summary.items()})
        mlflow.log_artifact(str(MODELS_DIR / "best_model.pt"))
        mlflow.log_artifact(str(MODELS_DIR / "best_config.json"))
        mlflow.log_artifact(str(EXPERIMENTS_DIR / "sweep_results.csv"))
        mlflow.log_artifact(str(REPORTS_DIR / "training_history.csv"))


if __name__ == "__main__":
    main()
