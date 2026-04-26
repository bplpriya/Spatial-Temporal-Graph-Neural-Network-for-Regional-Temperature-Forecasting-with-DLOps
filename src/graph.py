from __future__ import annotations

import math
from typing import List

import numpy as np


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two latitude/longitude points."""
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def pairwise_station_distances(stations: List[dict]) -> np.ndarray:
    """Compute station-to-station great-circle distances in kilometers."""
    n = len(stations)
    distances = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i != j:
                distances[i, j] = haversine_km(
                    stations[i]["latitude"],
                    stations[i]["longitude"],
                    stations[j]["latitude"],
                    stations[j]["longitude"],
                )
    return distances


def build_adjacency_matrix(
    stations: List[dict],
    threshold_km: float,
    method: str = "threshold",
    k_neighbors: int = 3,
    distance_sigma_km: float = 45.0,
) -> np.ndarray:
    """Build an explainable weather-station graph.

    ``threshold`` creates unweighted links for stations within a distance cutoff.
    ``distance_weighted_knn`` links each station to its nearest neighbors with
    exponentially decaying distance weights, then symmetrizes the graph.
    """
    n = len(stations)
    distances = pairwise_station_distances(stations)
    adj = np.eye(n, dtype=np.float32)

    if method == "distance_weighted_knn":
        sigma = max(float(distance_sigma_km), 1.0)
        k = min(max(int(k_neighbors), 1), n - 1)
        for i in range(n):
            neighbor_order = np.argsort(np.where(np.arange(n) == i, np.inf, distances[i]))
            for j in neighbor_order[:k]:
                weight = math.exp(-float(distances[i, j]) / sigma)
                adj[i, j] = max(adj[i, j], weight)
                adj[j, i] = max(adj[j, i], weight)
        return adj

    for i in range(n):
        for j in range(n):
            if i != j and distances[i, j] <= threshold_km:
                adj[i, j] = 1.0
    return adj


def normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    """Apply symmetric GCN normalization to the adjacency matrix."""
    degree = np.sum(adj, axis=1)
    degree_inv_sqrt = np.zeros_like(degree, dtype=np.float32)
    np.power(degree, -0.5, out=degree_inv_sqrt, where=degree > 0)
    d_hat = np.diag(degree_inv_sqrt)
    return d_hat @ adj @ d_hat
