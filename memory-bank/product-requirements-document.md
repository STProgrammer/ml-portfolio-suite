# ML Portfolio Suite — Product Requirements Document (PRD)

**Project repo name:** `ml_portfolio_suite`  
**Doc date:** 2026-02-01  
**Primary goal:** A single, simple project that demonstrates four applied ML tasks end-to-end (data → training → evaluation → demo UI) in one Python virtual environment (`.venv`) with reliable, repeatable runs.

---

## 1) Problem and goals

### Problem
You want a portfolio project that demonstrates practical ML skills across:
- Classification
- Regression
- Anomaly detection
- Forecasting (neural network approach, with GPU if available)

It must be easy to run locally, have a clean UI, and avoid fragile setup or confusing folder layouts.

### Goals
1. **One repo, one environment**: everything runs in the same `.venv`.
2. **One UI**: Streamlit app with four tabs (one per task).
3. **Smooth runs**: minimal “it broke” risk; predictable defaults; graceful errors.
4. **Local GPU optional** for forecasting: use GPU if available, otherwise CPU automatically.
5. **Simple structure**: no `src/ml_portfolio_suite/` nesting; code lives directly under the root in a small `ml/` folder.

### Success criteria
- A fresh clone + `.venv` setup can run `streamlit run app.py` without runtime errors.
- Each tab can:
  - download/load a dataset,
  - train a baseline + a main model,
  - show metrics and at least one plot,
  - save artifacts (model + metrics) locally.
- Forecasting tab trains on CPU by default and switches to GPU when available (no manual toggles required).

### Non-goals
- Production deployment, CI/CD, auth, multi-user accounts
- Full MLOps (monitoring, drift, feature stores)
- Heavy hyperparameter search that takes hours
- Perfect accuracy; the focus is correctness + demonstration + clarity

---

## 2) Target user and usage

### Target user
- Upwork clients reviewing your portfolio
- You (as the developer) during demos and screen recordings

### Demo style
- “Load data → Train → Evaluate → See plots → Export artifacts”
- Each tab includes a short explanation of what the model does and why the evaluation is valid.

---

## 3) Core UX (Streamlit)

### Navigation
A single Streamlit app: `app.py` with **4 tabs**:
1. **Classification**
2. **Regression**
3. **Anomaly Detection**
4. **Forecasting (NN)**

### Common UI elements (all tabs)
- Dataset section: source + “Load” button
- Model section: baseline + main model selector (default set)
- Train button: runs training with progress indicator
- Results section: metrics + plot(s)
- Export section: saves artifacts to `artifacts/` and shows file paths

### Error handling UX
- If dataset download fails, show a clear message and steps to fix (network, proxy, manual download).
- If GPU is not available, show: “GPU not detected; using CPU.”
- If a training run fails, show a short error summary and write a full traceback to `artifacts/logs/`.

---

## 4) Datasets and models (defaults)

> The defaults below are chosen for reliability and speed on CPU, while still looking professional.

### 4.1 Classification
**Dataset:** Adult (Census Income) via OpenML (downloaded automatically and cached locally).  
**Task:** Predict income class (binary).  
**Baseline model:** Logistic Regression.  
**Main model:** `HistGradientBoostingClassifier` (fast, strong on tabular data).  
**Metrics:** ROC-AUC, F1, precision/recall; confusion matrix plot.

### 4.2 Regression
**Dataset:** California Housing via scikit-learn fetch (auto-download + cache).  
**Task:** Predict median house value.  
**Baseline model:** Ridge Regression.  
**Main model:** `HistGradientBoostingRegressor`.  
**Metrics:** MAE, RMSE; residual plot.

### 4.3 Anomaly detection
**Dataset:** NAB (Numenta Anomaly Benchmark) — download a small subset automatically and cache it.  
**Task:** Detect anomalies in a time series.  
**Baseline:** simple rolling z-score threshold.  
**Main model:** `IsolationForest` on engineered window features OR residual-based method (forecast baseline + threshold residual).  
**Metrics/outputs:** precision/recall against labeled anomaly windows (where available), plus a time-series plot with anomaly markers.

### 4.4 Forecasting (NN)
**Dataset:** M4 dataset (use a small subset for fast runs, auto-download + cache).  
**Task:** Forecast future values for a selected series.  
**Model:** N-BEATS (neural forecasting).  
**Framework:** PyTorch (custom lightweight N-BEATS-style model for stability).  
**Device:** Auto-select:
- GPU if `torch.cuda.is_available()`
- otherwise CPU

**Baseline forecast:** seasonal naive or ETS/ARIMA (fast) for comparison.  
**Evaluation:** walk-forward / rolling evaluation on a holdout window.  
**Metrics:** MAE, sMAPE (or MASE if included).  
**Plots:** history + forecast with confidence band (optional), plus error over horizon.

---

## 5) Functional requirements

### FR-1: Single environment setup
- Project runs with `.venv` only.
- A single dependency file controls installs (either `requirements.txt` or `pyproject.toml`).
- All tasks run without needing a separate environment per tab.

### FR-2: One-command demo run
- Running `streamlit run app.py` starts the UI.
- UI exposes all 4 tasks.

### FR-3: Data download + caching
- On first load, each dataset downloads into `data/` (or `data/cache/`) and is reused on subsequent runs.
- Provide a “Reset cache” button (optional) to clear dataset cache.

### FR-4: Training and evaluation per task
Each tab must support:
- Baseline training
- Main model training
- Evaluation with appropriate metrics
- At least one plot
- Export artifacts

### FR-5: Artifact export
For each run, save:
- `metrics.json`
- model file (joblib/pickle for sklearn; `.pt` for PyTorch)
- plots (PNG)
- a short `run_summary.md` (optional)

Use a run folder like:
`artifacts/<task>/<YYYYMMDD_HHMMSS>/...`

### FR-6: GPU fallback for forecasting
- Forecasting tab uses GPU if available, otherwise CPU.
- UI shows which device is used.

### FR-7: Deterministic-ish runs
- Set global seeds for numpy + random + torch where feasible.
- Make it obvious when full determinism is not guaranteed (GPU kernels).

---

## 6) Non-functional requirements

### NFR-1: Reliability
- No unhandled exceptions in normal runs.
- Input validation for UI fields (horizon, epochs, etc.).

### NFR-2: Performance
- Default runs should finish fast enough for a live demo:
  - Classification/regression: < 30–60 seconds typical on laptop CPU
  - Anomaly: < 60 seconds
  - Forecasting NN: < 2–5 minutes on CPU using a small subset and modest epochs

### NFR-3: Simplicity
- Minimal modules; avoid unnecessary abstractions.
- Only add configuration files that are used.

### NFR-4: Portability
- Works on Windows and Linux.
- Avoid OS-specific paths; use `pathlib`.

---

## 7) Project structure (simple)

Root folder: `ml_portfolio_suite/`

Recommended layout:

- `app.py` — Streamlit UI
- `ml/`
  - `classification.py`
  - `regression.py`
  - `anomaly.py`
  - `forecasting.py`
  - `common.py` (shared utils: seeding, caching, plotting helpers)
- `data/` (auto-downloaded datasets, cached)
- `artifacts/` (models, metrics, plots, logs)
- `requirements.txt` (or `pyproject.toml`)
- `README.md`
- `PRD.md`

Notes:
- No `src/` nesting.
- The `ml/` folder uses simple imports (not packaged distribution install).

---

## 8) Environment and dependencies

### Python
- Python 3.10+ recommended (choose one and pin it in README).

### Core libraries
- `streamlit`
- `pandas`, `numpy`
- `scikit-learn`
- `matplotlib`
- `joblib`
- Forecasting NN stack:
  - `torch`

### Dependency pinning
- Pin major versions in `requirements.txt` to reduce breakage.
- Add a short “known good” set of versions in README.

---

## 9) Run instructions (must be in README, primarily built for Windows [Windows 11]).

Windows PowerShell example:
```bash
cd ml_portfolio_suite
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

macOS/Linux example:
```bash
cd ml_portfolio_suite
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## 10) Acceptance checklist

- [ ] Repo runs in one `.venv` with one install step
- [ ] Streamlit app opens with 4 tabs
- [ ] Each tab loads data (auto-download + cache)
- [ ] Each tab trains baseline + main model successfully
- [ ] Each tab shows metrics + at least one plot
- [ ] Each tab exports artifacts to `artifacts/`
- [ ] Forecasting tab uses GPU if detected, otherwise CPU, with clear UI message
- [ ] No code is nested under `src/ml_portfolio_suite/`

---

## 11) Risks and mitigations (practical)

1. **Dataset links change / download fails**
   - Mitigation: cache downloads; provide manual download instructions in UI and README.

2. **Forecasting NN dependencies are heavy**
   - Mitigation: keep the NN stack minimal (PyTorch + one forecasting lib), use a small M4 subset, default epochs low.

3. **Windows CUDA setup issues**
   - Mitigation: forecasting runs fine on CPU; GPU use is optional. Detect GPU automatically and fall back cleanly.

4. **Long run times on CPU**
   - Mitigation: default to small subsets, modest epochs, and provide “Fast mode” toggle (optional).

---

## 12) Deliverables (what must exist in the repo)
- `app.py` Streamlit UI with 4 tabs
- `ml/` modules implementing train/eval per task
- `requirements.txt` (or `pyproject.toml`) and `.venv` guidance
- dataset auto-download + cache in `data/`
- artifact exports into `artifacts/`
- `README.md` with setup and demo steps
- `PRD.md` (this file)
