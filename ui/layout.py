from __future__ import annotations

from pathlib import Path
import shutil

import streamlit as st


APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = APP_ROOT / "data"
ARTIFACTS_DIR = APP_ROOT / "artifacts"


def reset_cache() -> None:
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def health_check() -> dict:
    checks = {}
    try:
        import torch  # noqa: F401
        import sklearn  # noqa: F401

        checks["imports"] = "ok"
    except Exception as exc:
        checks["imports"] = f"fail: {exc}"

    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        test_file = DATA_DIR / "write_test.tmp"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
        checks["data_writable"] = "ok"
    except Exception as exc:
        checks["data_writable"] = f"fail: {exc}"

    try:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        test_file = ARTIFACTS_DIR / "write_test.tmp"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
        checks["artifacts_writable"] = "ok"
    except Exception as exc:
        checks["artifacts_writable"] = f"fail: {exc}"

    return checks


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Utilities")
        if st.button("Reset cache (data/)"):
            reset_cache()
            st.success("Cache cleared.")
        if st.button("Health check"):
            st.json(health_check())
