from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import requests

from src.utils import ensure_dir, load_yaml

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"


HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
]


def fetch_station_data(name: str, latitude: float, longitude: float, start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "America/New_York",
    }
    response = requests.get(BASE_URL, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    hourly = payload["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df["station"] = name
    df["latitude"] = latitude
    df["longitude"] = longitude
    return df


def main() -> None:
    params = load_yaml("params.yaml")
    stations_cfg = load_yaml("configs/stations.yaml")
    ensure_dir("data/raw")

    all_frames: List[pd.DataFrame] = []
    for station in stations_cfg["stations"]:
        df = fetch_station_data(
            name=station["name"],
            latitude=station["latitude"],
            longitude=station["longitude"],
            start_date=params["data"]["start_date"],
            end_date=params["data"]["end_date"],
        )
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv("data/raw/weather_raw.csv", index=False)
    print(f"Saved raw data: {combined.shape}")


if __name__ == "__main__":
    main()