from __future__ import annotations

from pathlib import Path
import shutil

import streamlit as st


APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = APP_ROOT / "data"
ARTIFACTS_DIR = APP_ROOT / "artifacts"


def apply_global_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:opsz@8..144&family=Manrope:wght@300;400;500;600;700&display=swap');

        :root {
            --bg: #f6f4ef;
            --bg-soft: #fbfaf7;
            --ink: #151515;
            --ink-soft: #3c3c3c;
            --muted: #6b6b6b;
            --accent: #1f6f78;
            --accent-2: #d5a24c;
            --card: #ffffff;
            --border: #e7e1d9;
            --shadow: 0 10px 30px rgba(21, 21, 21, 0.08);
        }

        html, body, [class*="css"]  {
            font-family: "Manrope", system-ui, -apple-system, "Segoe UI", sans-serif;
            color: var(--ink);
            font-size: 16.5px;
        }

        .stApp {
            background: radial-gradient(1200px 800px at 10% -10%, #e7f3f4 0%, transparent 60%),
                        radial-gradient(1000px 700px at 100% 0%, #f6ead2 0%, transparent 55%),
                        var(--bg);
        }

        h1, h2, h3, h4 {
            font-family: "Instrument Serif", "Times New Roman", serif;
            color: var(--ink);
            letter-spacing: 0.2px;
        }

        .block-container {
            padding-top: 2.5rem;
            padding-bottom: 7rem;
        }

        .hero {
            background: linear-gradient(120deg, rgba(255,255,255,0.9), rgba(250,246,236,0.9));
            border: 1px solid rgba(31, 111, 120, 0.15);
            border-radius: 18px;
            padding: 24px 28px;
            box-shadow: var(--shadow);
            margin-bottom: 1.75rem;
            animation: rise 0.6s ease-out;
            position: relative;
        }

        .hero-title {
            font-size: 2.4rem;
            font-weight: 600;
            margin-bottom: 0.35rem;
            color: var(--ink);
        }

        .hero-sub {
            font-size: 1.02rem;
            color: var(--ink-soft);
            margin-bottom: 0.75rem;
        }

        .section-card {
            background: transparent;
            border-left: 3px solid rgba(31, 111, 120, 0.28);
            border-radius: 8px;
            padding: 0.25rem 0 0.25rem 1rem;
            margin: 0.35rem 0 1.25rem 0;
            animation: rise 0.6s ease-out;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
            background: rgba(255, 255, 255, 0.6);
            border: 1px solid rgba(31, 111, 120, 0.18);
            padding: 0.35rem;
            border-radius: 999px;
            box-shadow: 0 6px 18px rgba(21, 21, 21, 0.06);
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(31, 111, 120, 0.2);
            border-radius: 999px;
            padding: 0.5rem 1.15rem;
            color: var(--ink-soft);
            font-weight: 700;
            box-shadow: 0 6px 16px rgba(21, 21, 21, 0.06);
            cursor: pointer;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: var(--ink);
            border-color: rgba(31,111,120,0.5);
            box-shadow: 0 8px 20px rgba(21, 21, 21, 0.12);
            transform: translateY(-1px);
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(120deg, rgba(31,111,120,0.2), rgba(213,162,76,0.18));
            color: var(--ink);
            border-color: rgba(31,111,120,0.6);
            box-shadow: 0 10px 24px rgba(21, 21, 21, 0.16);
        }

        .stButton > button {
            border-radius: 999px;
            padding: 0.5rem 1.15rem;
            font-weight: 700;
            border: 1px solid rgba(31,111,120,0.4);
            background: linear-gradient(120deg, rgba(31,111,120,0.18), rgba(213,162,76,0.2));
            color: #0f2c2f;
            margin-bottom: 0.75rem;
        }

        .stButton > button:hover {
            border-color: rgba(31,111,120,0.6);
            transform: translateY(-1px);
        }

        .stRadio > label, .stCheckbox > label, .stSlider > label {
            font-weight: 600;
            color: var(--ink);
        }

        .stSidebar {
            background: #10181a;
        }

        .stSidebar [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #10181a 0%, #0c1214 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }

        .stSidebar * {
            color: #f7f3ec;
        }

        .stSidebar h2, .stSidebar h3 {
            font-family: "Instrument Serif", "Times New Roman", serif;
            font-size: 1.55rem;
            letter-spacing: 0.4px;
            color: #f7f3ec;
        }

        .stSidebar .stButton > button {
            background: #1f6f78;
            color: #ffffff;
            border: 1px solid #1f6f78;
            border-radius: 12px;
            width: 100%;
            font-weight: 700;
            letter-spacing: 0.2px;
            margin-bottom: 0.5rem;
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.22);
        }

        .stSidebar .stButton > button:hover {
            background: #185d64;
            border-color: #185d64;
            transform: translateY(-1px);
        }

        .stJson, .stCodeBlock, .stMarkdown {
            font-size: 1rem;
            line-height: 1.7;
        }

        div[data-testid="stImage"],
        div[data-testid="stPlot"] {
            max-width: 820px;
            margin: 0.35rem auto 1.25rem auto;
        }

        div[data-testid="stImage"] img,
        div[data-testid="stPlot"] img {
            width: 100%;
            height: auto;
            display: block;
        }

        @media (max-width: 640px) {
            div[data-testid="stImage"],
            div[data-testid="stPlot"] {
                max-width: 100%;
            }
        }

        .stAlert {
            border-radius: 12px;
        }

        @keyframes rise {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">ML Portfolio Suite</div>
            <div class="hero-sub">
                A single Streamlit app showcasing four applied ML tasks with clean, repeatable runs.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
