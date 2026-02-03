from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

from ml.common import DATA_DIR, new_run_dir


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_data(
    cache_path: Path | None = None,
    *,
    fast_mode: bool = False,
    sample_frac: float = 0.2,
    random_state: int = 42,
) -> Dataset:
    if cache_path is None:
        cache_path = DATA_DIR / "adult.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        df = pd.read_csv(cache_path)
    else:
        openml = fetch_openml(name="adult", version=2, as_frame=True)
        df = openml.frame.copy()
        df.to_csv(cache_path, index=False)

    if "class" not in df.columns:
        raise ValueError("Expected target column 'class' not found in Adult dataset.")

    if fast_mode:
        df = df.sample(frac=sample_frac, random_state=random_state)

    y = df["class"]
    X = df.drop(columns=["class"])

    if y.nunique() != 2:
        raise ValueError("Adult dataset target must have exactly 2 classes.")

    return Dataset(X=X, y=y)


def _build_preprocessor() -> ColumnTransformer:
    categorical_selector = make_column_selector(dtype_include=["object", "category", "bool"])
    numeric_selector = make_column_selector(dtype_include=["number"])

    categorical = Pipeline(
        steps=[
            ("onehot", _make_one_hot_encoder()),
        ]
    )
    numeric = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("cat", categorical, categorical_selector),
            ("num", numeric, numeric_selector),
        ],
        remainder="drop",
    )


def train_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    preprocessor = _build_preprocessor()
    model = LogisticRegression(max_iter=1000, n_jobs=None)
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    return pipeline.fit(X_train, y_train)


def train_main_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    preprocessor = _build_preprocessor()
    model = HistGradientBoostingClassifier(random_state=42)
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    return pipeline.fit(X_train, y_train)


def evaluate(
    model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[Dict[str, float], np.ndarray]:
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
    }
    cm = confusion_matrix(y_test, y_pred)
    return metrics, cm


def plot_confusion_matrix(cm: np.ndarray, class_labels: Tuple[str, str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
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
        run_dir = new_run_dir("classification")
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "model.joblib"
    metrics_path = run_dir / "metrics.json"
    plot_path = run_dir / "confusion_matrix.png"

    dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fig.savefig(plot_path, dpi=150)

    return run_dir


def train_test_split_data(
    dataset: Dataset, *, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        dataset.X, dataset.y, test_size=test_size, random_state=random_state, stratify=dataset.y
    )
