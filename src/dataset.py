from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class WeatherDataset(Dataset):
    """Torch dataset wrapper for precomputed weather windows and targets."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]
