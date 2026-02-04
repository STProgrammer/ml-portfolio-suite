from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import zipfile

from ml.common import DATA_DIR, new_run_dir, download_with_cache


M4_CACHE = DATA_DIR / "m4_subset.csv"
M4_ZIP = DATA_DIR / "m4_monthly_dataset.zip"
M4_URL = "https://zenodo.org/records/4656480/files/m4_monthly_dataset.zip?download=1"
DEFAULT_SERIES_ID = "T1"


@dataclass
class SeriesBundle:
    series_id: str
    series: pd.DataFrame


def download_m4_subset(
    *,
    cache_path: Path | None = None,
    subset_size: int = 5,
) -> Path:
    if cache_path is None:
        cache_path = M4_CACHE
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        return cache_path

    download_with_cache(M4_URL, M4_ZIP)

    with zipfile.ZipFile(M4_ZIP, "r") as zf:
        csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        tsf_names = [name for name in zf.namelist() if name.lower().endswith(".tsf")]

        if csv_names:
            train_candidates = [n for n in csv_names if "train" in n.lower()]
            target_name = train_candidates[0] if train_candidates else csv_names[0]
            with zf.open(target_name) as handle:
                wide = pd.read_csv(handle)
            if wide.shape[1] < 2:
                raise ValueError("Unexpected M4 train format; needs ID column + values.")
        elif tsf_names:
            target_name = tsf_names[0]
            with zf.open(target_name) as handle:
                content = handle.read().decode("latin-1", errors="ignore")
            lines = content.splitlines()
            data_idx = next(
                (i for i, line in enumerate(lines) if line.strip().lower() == "@data"),
                None,
            )
            if data_idx is None:
                raise ValueError("TSF file missing @data section.")
            data_lines = lines[data_idx + 1 :]
            rows = []
            for line in data_lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(":")
                series_id = parts[0].strip()
                values = parts[-1].split(",")
                rows.append([series_id] + values)
            if not rows:
                raise ValueError("No series found in TSF data.")
            wide = pd.DataFrame(rows)
        else:
            raise ValueError("No CSV or TSF files found inside the M4 monthly zip.")

    wide = wide.head(subset_size)
    id_col = wide.columns[0]
    value_cols = wide.columns[1:]

    rows = []
    for _, row in wide.iterrows():
        series_id = row[id_col]
        values = row[value_cols].to_numpy()
        for t, value in enumerate(values):
            if pd.isna(value):
                continue
            rows.append({"series_id": str(series_id), "time": t, "value": float(value)})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(cache_path, index=False)
    return cache_path


def load_series_list(cache_path: Path | None = None) -> List[str]:
    if cache_path is None:
        cache_path = M4_CACHE
    if not cache_path.exists():
        download_m4_subset(cache_path=cache_path)
    df = pd.read_csv(cache_path)
    return sorted(df["series_id"].unique().tolist())


def load_series(series_id: str = DEFAULT_SERIES_ID, cache_path: Path | None = None) -> SeriesBundle:
    if cache_path is None:
        cache_path = M4_CACHE
    if not cache_path.exists():
        download_m4_subset(cache_path=cache_path)
    df = pd.read_csv(cache_path)
    series_df = df[df["series_id"] == series_id].copy()
    if series_df.empty:
        raise ValueError(f"Series id '{series_id}' not found in cached subset.")
    series_df = series_df.sort_values("time")
    return SeriesBundle(series_id=series_id, series=series_df)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class SeasonalNaiveModel:
    def __init__(self, seasonal_period: int = 12):
        self.seasonal_period = seasonal_period
        self.history: Optional[np.ndarray] = None

    def fit(self, values: np.ndarray) -> None:
        self.history = values.astype(np.float32)

    def predict(self, horizon: int) -> np.ndarray:
        if self.history is None:
            raise ValueError("Model not fitted.")
        if len(self.history) < self.seasonal_period:
            return np.repeat(self.history[-1], horizon)
        last_season = self.history[-self.seasonal_period :]
        reps = int(np.ceil(horizon / self.seasonal_period))
        return np.tile(last_season, reps)[:horizon]


def train_baseline(series: pd.DataFrame, *, seasonal_period: int = 12) -> SeasonalNaiveModel:
    model = SeasonalNaiveModel(seasonal_period=seasonal_period)
    model.fit(series["value"].to_numpy())
    return model


class NBeatsTiny(nn.Module):
    def __init__(self, input_len: int, output_len: int, hidden_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_len, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_len)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


@dataclass
class NBeatsWrapper:
    model: NBeatsTiny
    input_len: int
    output_len: int
    device: torch.device

    def predict(self, horizon: int, series: pd.DataFrame) -> np.ndarray:
        values = series["value"].to_numpy().astype(np.float32)
        if len(values) < self.input_len:
            raise ValueError("Series too short for prediction.")
        window = values[-self.input_len :]
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(x).cpu().numpy().flatten()
        if horizon <= self.output_len:
            return out[:horizon]
        reps = int(np.ceil(horizon / self.output_len))
        return np.tile(out, reps)[:horizon]


def train_nbeats(
    series: pd.DataFrame,
    *,
    epochs: int = 10,
    input_chunk_length: int = 24,
    output_chunk_length: int = 12,
    random_state: int = 42,
) -> NBeatsWrapper:
    torch.manual_seed(random_state)
    values = series["value"].to_numpy().astype(np.float32)
    if len(values) <= input_chunk_length + output_chunk_length:
        raise ValueError("Series too short for the chosen input/output lengths.")

    X, y = [], []
    for i in range(len(values) - input_chunk_length - output_chunk_length + 1):
        X.append(values[i : i + input_chunk_length])
        y.append(values[i + input_chunk_length : i + input_chunk_length + output_chunk_length])
    X = torch.tensor(np.stack(X))
    y = torch.tensor(np.stack(y))

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NBeatsTiny(input_chunk_length, output_chunk_length).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = loss_fn(preds, batch_y)
            loss.backward()
            optimizer.step()

    return NBeatsWrapper(
        model=model,
        input_len=input_chunk_length,
        output_len=output_chunk_length,
        device=device,
    )


def evaluate(
    model,
    series: pd.DataFrame,
    *,
    horizon: int = 24,
) -> Dict[str, float]:
    values = series["value"].to_numpy().astype(np.float32)
    if len(values) <= horizon + 1:
        horizon = max(1, len(values) // 4)
    train = values[:-horizon]
    test = values[-horizon:]

    if hasattr(model, "predict"):
        if isinstance(model, NBeatsWrapper):
            forecast = model.predict(horizon, series)
        else:
            forecast = model.predict(horizon)
    else:
        raise ValueError("Model does not support prediction.")

    forecast = np.asarray(forecast).astype(np.float32)
    mae_val = np.mean(np.abs(test - forecast))
    smape_val = np.mean(2 * np.abs(test - forecast) / (np.abs(test) + np.abs(forecast) + 1e-8))

    return {"mae": float(mae_val), "smape": float(smape_val)}


def plot_forecast(
    model,
    series: pd.DataFrame,
    *,
    horizon: int = 24,
) -> plt.Figure:
    values = series["value"].to_numpy().astype(np.float32)
    if len(values) <= horizon + 1:
        horizon = max(1, len(values) // 4)
    train = values[:-horizon]
    test = values[-horizon:]
    if isinstance(model, NBeatsWrapper):
        forecast = model.predict(horizon, series)
    else:
        forecast = model.predict(horizon)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(train)), train, label="history")
    ax.plot(range(len(train), len(train) + len(test)), test, label="actual")
    ax.plot(
        range(len(train), len(train) + len(forecast)),
        forecast,
        label="forecast",
    )
    ax.set_title("Forecast vs Actual")
    ax.legend()
    fig.tight_layout()
    return fig


def export_artifacts(
    model,
    metrics: Dict[str, float],
    fig: plt.Figure,
    *,
    run_dir: Optional[Path] = None,
) -> Path:
    if run_dir is None:
        run_dir = new_run_dir("forecasting")
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics.json"
    plot_path = run_dir / "forecast.png"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fig.savefig(plot_path, dpi=150)

    model_path = run_dir / "model.pt"
    if isinstance(model, NBeatsWrapper):
        torch.save(model.model.state_dict(), model_path)

    return run_dir
