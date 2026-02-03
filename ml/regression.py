from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.common import DATA_DIR, new_run_dir


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series


def load_data(
    cache_path: Path | None = None,
    *,
    fast_mode: bool = False,
    sample_frac: float = 0.2,
    random_state: int = 42,
) -> Dataset:
    if cache_path is None:
        cache_path = DATA_DIR / "california_housing.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        df = pd.read_csv(cache_path)
    else:
        housing = fetch_california_housing(as_frame=True)
        df = housing.frame.copy()
        df.to_csv(cache_path, index=False)

    if "MedHouseVal" not in df.columns:
        raise ValueError("Expected target column 'MedHouseVal' not found in dataset.")

    if df["MedHouseVal"].isna().any():
        raise ValueError("Target column contains missing values.")

    if fast_mode:
        df = df.sample(frac=sample_frac, random_state=random_state)

    y = df["MedHouseVal"]
    X = df.drop(columns=["MedHouseVal"])

    non_numeric = X.select_dtypes(exclude=["number"])
    if not non_numeric.empty:
        raise ValueError("Feature matrix must be numeric for this regression task.")

    return Dataset(X=X, y=y)


def train_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    return pipeline.fit(X_train, y_train)


def train_main_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("model", HistGradientBoostingRegressor(random_state=42)),
        ]
    )
    return pipeline.fit(X_train, y_train)


def evaluate(
    model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, float]:
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    return {"mae": float(mae), "rmse": float(rmse)}


def plot_residuals(
    model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> plt.Figure:
    preds = model.predict(X_test)
    residuals = y_test - preds
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(preds, residuals, alpha=0.3)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title("Residual Plot")
    fig.tight_layout()
    return fig


def export_artifacts(
    model: Pipeline,
    metrics: Dict[str, float],
    fig: plt.Figure,
    *,
    run_dir: Optional[Path] = None,
) -> Path:
    if run_dir is None:
        run_dir = new_run_dir("regression")
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "model.joblib"
    metrics_path = run_dir / "metrics.json"
    plot_path = run_dir / "residuals.png"

    dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fig.savefig(plot_path, dpi=150)

    return run_dir


def train_test_split_data(
    dataset: Dataset, *, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        dataset.X, dataset.y, test_size=test_size, random_state=random_state
    )
