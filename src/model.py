from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    """Graph convolution using a pre-normalized station adjacency matrix."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = self.linear(x)
        return torch.einsum("ij,bjf->bif", adj, support)


class STGNN(nn.Module):
    """Residual spatial-temporal GNN for multi-horizon station forecasting."""

    def __init__(
        self,
        num_features: int,
        num_horizons: int,
        hidden_dim: int,
        gcn_out_dim: int,
        dropout: float,
        temporal_hidden_dim: int | None = None,
        temporal_layers: int = 2,
        target_feature_idx: int = 0,
        predict_residual: bool = True,
    ):
        super().__init__()
        self.target_feature_idx = target_feature_idx
        self.predict_residual = predict_residual
        temporal_hidden_dim = temporal_hidden_dim or hidden_dim
        self.gcn1 = GraphConv(num_features, hidden_dim)
        self.gcn2 = GraphConv(hidden_dim, gcn_out_dim)
        self.residual_proj = nn.Linear(num_features, gcn_out_dim)
        self.norm = nn.LayerNorm(gcn_out_dim)
        self.dropout = nn.Dropout(dropout)
        self.temporal = nn.GRU(
            input_size=gcn_out_dim,
            hidden_size=temporal_hidden_dim,
            num_layers=temporal_layers,
            batch_first=True,
            dropout=dropout if temporal_layers > 1 else 0.0,
        )
        self.horizon_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(temporal_hidden_dim),
                    nn.Dropout(dropout),
                    nn.Linear(temporal_hidden_dim, 1),
                )
                for _ in range(num_horizons)
            ]
        )
        self.logvar_head = nn.Linear(temporal_hidden_dim, num_horizons)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, window, nodes, _ = x.shape
        xt = x.reshape(batch_size * window, nodes, -1)
        h = F.gelu(self.gcn1(xt, adj))
        h = self.dropout(h)
        h = self.gcn2(h, adj)
        h = self.norm(F.gelu(h + self.residual_proj(xt)))
        spatial_seq = h.view(batch_size, window, nodes, -1)
        spatial_seq = spatial_seq.permute(0, 2, 1, 3).contiguous()
        spatial_seq = spatial_seq.view(batch_size * nodes, window, -1)

        temporal_out, _ = self.temporal(spatial_seq)
        last_hidden = temporal_out[:, -1, :]

        horizon_means = [head(last_hidden) for head in self.horizon_heads]
        pred_mean = torch.cat(horizon_means, dim=1)
        pred_logvar = self.logvar_head(last_hidden).clamp(min=-6.0, max=4.0)

        pred_mean = pred_mean.view(batch_size, nodes, -1).permute(0, 2, 1)
        if self.predict_residual:
            last_target = x[:, -1, :, self.target_feature_idx].unsqueeze(1)
            pred_mean = pred_mean + last_target
        pred_logvar = pred_logvar.view(batch_size, nodes, -1).permute(0, 2, 1)
        return pred_mean, pred_logvar


class LSTMBaseline(nn.Module):
    """Station-independent LSTM baseline using the same engineered features."""

    def __init__(
        self,
        num_features: int,
        num_horizons: int,
        hidden_dim: int,
        dropout: float,
        num_layers: int = 1,
    ):
        super().__init__()
        self.num_horizons = num_horizons
        self.temporal = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_horizons),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, window, nodes, features = x.shape
        station_series = x.permute(0, 2, 1, 3).contiguous().view(batch_size * nodes, window, features)
        temporal_out, _ = self.temporal(station_series)
        pred = self.head(temporal_out[:, -1, :])
        return pred.view(batch_size, nodes, self.num_horizons).permute(0, 2, 1)


def gaussian_nll_loss(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Gaussian negative log likelihood for mean and log-variance forecasts."""
    return torch.mean(0.5 * (logvar + ((target - mean) ** 2) / torch.exp(logvar)))


def forecast_loss(
    mean: torch.Tensor,
    target: torch.Tensor,
    loss_name: str = "huber",
    logvar: torch.Tensor | None = None,
) -> torch.Tensor:
    """Training loss selector; Huber is a stable default for MAE/RMSE-focused tuning."""
    if loss_name == "gaussian_nll" and logvar is not None:
        return gaussian_nll_loss(mean, logvar, target)
    if loss_name == "mse":
        return F.mse_loss(mean, target)
    return F.smooth_l1_loss(mean, target, beta=0.5)
