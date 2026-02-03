# ML Portfolio Suite — Tech Stack

This stack is optimized for:
- **Reliability on Windows 11**
- **One `.venv`** for everything
- **Fast CPU defaults**, with **GPU auto-use** for the forecasting NN when available
- Minimal moving parts (no heavy infrastructure)

---

## 1) Runtime and environment

- **Python:** 3.10 or 3.11 (recommend **3.10** for widest library compatibility on Windows)
- **Virtual environment:** built-in `venv` (`.venv` at repo root)
- **Dependency management:** `requirements.txt` (single file, pinned major versions)

Why: simplest setup with the least tooling friction, matches your PRD.

---

## 2) App/UI layer

- **Streamlit** — single-page app with 4 tabs (classification, regression, anomaly, forecasting)

Why: fast demo UI, easy for clients to understand, minimal boilerplate.

---

## 3) Data + numerical stack

- **NumPy** — numeric arrays
- **Pandas** — tabular data handling
- **Requests** — robust dataset downloads (NAB/M4 assets when needed)
- **PyArrow (optional)** — faster IO if you want Parquet caching (not required)

Why: standard, stable tooling for data workflows.

---

## 4) Classical ML stack (classification, regression, anomaly)

- **scikit-learn**
  - Classification: `LogisticRegression`, `HistGradientBoostingClassifier`
  - Regression: `Ridge`, `HistGradientBoostingRegressor`
  - Anomaly detection: `IsolationForest`
  - Utilities: pipelines, preprocessing, metrics

- **Joblib** — model persistence for sklearn models (`.joblib`)

Why: scikit-learn is the most recognizable and trusted library for tabular ML, with strong baselines that run well on CPU.

---

## 5) Forecasting NN stack (N-BEATS-style)

- **PyTorch** — core NN framework
- **Custom lightweight model** — a small N-BEATS-style MLP for fast, reliable demos
  - Runs on GPU when `torch.cuda.is_available()` is true
  - Avoids extra dependency conflicts from heavy forecasting libraries

### GPU/CPU behavior
- The app checks `torch.cuda.is_available()`.
- If true, it selects a CUDA device for forecasting training.
- Otherwise, it trains on CPU.

### PyTorch installation note (Windows)
PyTorch CPU-only vs CUDA-enabled depends on how you install it.
In `README.md`, include two install options:
- **CPU-only install** (works everywhere)
- **CUDA install** (for NVIDIA GPUs, correct CUDA wheel/index URL)

This keeps the default setup reliable while still supporting GPU users.

---

## 6) Plotting and reporting

- **Matplotlib** — plots for:
  - confusion matrix
  - residuals
  - time series + anomaly markers
  - forecast vs actual

Why: widely supported, low dependency risk.

---

## 7) Project hygiene (small, optional but recommended)

These are optional because you asked for simplicity, but they reduce “random errors” during development:

- **pytest** — quick sanity checks for data loading and model training
- **ruff** — fast linting/formatting (optional)

If you want the absolute minimum dependency surface, you can skip these.

---

## 8) Recommended `requirements.txt` groups

**Core (always):**
- streamlit
- numpy
- pandas
- scikit-learn
- matplotlib
- joblib
- requests

**Forecasting NN:**
- torch

**Dev-only (optional):**
- pytest
- ruff

---

## 9) Why this stack fits your PRD

- One `.venv`, one install step
- Streamlit makes demos and screen recordings easy
- scikit-learn covers 3/4 tasks with strong, CPU-friendly models
- PyTorch gives you NN forecasting (N-BEATS-style) with GPU auto-use when available
- Minimal folder layout and minimal infrastructure
