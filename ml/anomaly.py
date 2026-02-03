from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score

from ml.common import DATA_DIR, new_run_dir, download_with_cache


DEFAULT_SERIES = "realKnownCause/nyc_taxi.csv"
NAB_BASE_URL = "https://raw.githubusercontent.com/numenta/NAB/master"


@dataclass
class SeriesData:
    timestamps: pd.Series
    values: pd.Series
    labels: Optional[List[Tuple[pd.Timestamp, pd.Timestamp]]]


def download_nab_subset(
    *,
    series: str = DEFAULT_SERIES,
    data_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    if data_dir is None:
        data_dir = DATA_DIR / "nab"
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / Path(series).name
    labels_path = data_dir / "combined_labels.json"

    csv_url = f"{NAB_BASE_URL}/data/{series}"
    labels_url = f"{NAB_BASE_URL}/labels/combined_labels.json"

    download_with_cache(csv_url, csv_path)
    download_with_cache(labels_url, labels_path)

    return csv_path, labels_path


def load_series(
    csv_path: Path,
    labels_path: Optional[Path] = None,
    *,
    series_key: str = DEFAULT_SERIES,
) -> SeriesData:
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError("NAB CSV must have at least two columns (timestamp, value).")

    timestamps = pd.to_datetime(df.iloc[:, 0])
    values = pd.to_numeric(df.iloc[:, 1], errors="coerce")

    if values.isna().any():
        raise ValueError("Series contains missing or non-numeric values.")

    labels = None
    if labels_path is not None and labels_path.exists():
        labels_json = json.loads(labels_path.read_text(encoding="utf-8"))
        windows = labels_json.get(series_key)
        if windows:
            parsed: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
            for item in windows:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    start, end = item
                    parsed.append((pd.to_datetime(start), pd.to_datetime(end)))
                else:
                    ts = pd.to_datetime(item)
                    parsed.append((ts, ts))
            labels = parsed

    return SeriesData(timestamps=timestamps, values=values, labels=labels)


def baseline_zscore(
    series: pd.Series,
    *,
    window: int = 24,
    threshold: float = 3.0,
) -> pd.Series:
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    zscore = (series - rolling_mean) / rolling_std
    return zscore.abs() > threshold


def _build_window_features(series: pd.Series, window: int = 24) -> pd.DataFrame:
    rolling = series.rolling(window=window, min_periods=window)
    features = pd.DataFrame(
        {
            "mean": rolling.mean(),
            "std": rolling.std(),
            "min": rolling.min(),
            "max": rolling.max(),
            "median": rolling.median(),
        }
    )
    return features.dropna()


def train_main_model(
    series: pd.Series,
    *,
    window: int = 24,
    contamination: float = 0.01,
    random_state: int = 42,
) -> Tuple[IsolationForest, pd.Series]:
    features = _build_window_features(series, window=window)
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(features)
    preds = model.predict(features)
    anomaly_flags = pd.Series(preds == -1, index=features.index)
    return model, anomaly_flags


def _labels_to_mask(
    timestamps: pd.Series, windows: List[Tuple[pd.Timestamp, pd.Timestamp]]
) -> pd.Series:
    mask = pd.Series(False, index=timestamps.index)
    for start, end in windows:
        mask |= (timestamps >= start) & (timestamps <= end)
    return mask


def evaluate(
    timestamps: pd.Series,
    values: pd.Series,
    anomalies: pd.Series,
    label_windows: Optional[List[Tuple[pd.Timestamp, pd.Timestamp]]] = None,
) -> Tuple[Dict[str, float], plt.Figure]:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(timestamps, values, label="value", linewidth=1)
    ax.scatter(
        timestamps[anomalies.index][anomalies.values],
        values[anomalies.index][anomalies.values],
        color="red",
        s=12,
        label="anomaly",
    )
    ax.set_title("Anomaly Detection")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()

    metrics: Dict[str, float] = {
        "anomaly_count": float(anomalies.sum()),
    }

    if label_windows:
        labels_mask = _labels_to_mask(timestamps, label_windows)
        aligned_labels = labels_mask.loc[anomalies.index]
        precision = precision_score(aligned_labels, anomalies)
        recall = recall_score(aligned_labels, anomalies)
        metrics.update({"precision": float(precision), "recall": float(recall)})

    return metrics, fig


def export_artifacts(
    model: Optional[IsolationForest],
    metrics: Dict[str, float],
    fig: plt.Figure,
    *,
    run_dir: Optional[Path] = None,
) -> Path:
    if run_dir is None:
        run_dir = new_run_dir("anomaly")
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics.json"
    plot_path = run_dir / "anomaly_plot.png"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fig.savefig(plot_path, dpi=150)

    if model is not None:
        model_path = run_dir / "model.joblib"
        dump(model, model_path)

    return run_dir
