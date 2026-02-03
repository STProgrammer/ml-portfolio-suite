from __future__ import annotations

from datetime import datetime
from pathlib import Path
import random
import traceback
from typing import Optional

import numpy as np
import requests


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # Torch is optional; ignore if not installed or any init errors.
        pass


def new_run_dir(task_name: str) -> Path:
    ensure_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ARTIFACTS_DIR / task_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def download_with_cache(url: str, filepath: Path, timeout: int = 30) -> Path:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if filepath.exists():
        return filepath
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    filepath.write_bytes(response.content)
    return filepath


def log_exception(run_dir: Path, exc: BaseException) -> Optional[Path]:
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "traceback.txt"
    trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    log_path.write_text(trace, encoding="utf-8")
    return log_path
