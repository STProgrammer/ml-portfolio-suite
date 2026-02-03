# ML Portfolio Suite

Single‑repo Streamlit app that demonstrates four applied ML tasks end‑to‑end:
classification, regression, anomaly detection, and forecasting.

## Setup (Windows-first)

1) Create a virtual environment at the repo root:

```powershell
py -m venv .venv
```

2) Activate the environment:

```powershell
.venv\\Scripts\\Activate.ps1
```

3) Install dependencies:

```powershell
pip install -r requirements.txt
```

4) Run the Streamlit app:

```powershell
streamlit run app.py
```

## Using the app

- Open the app and pick a tab: Classification, Regression, Anomaly Detection, or Forecasting.
- For each tab:
  1) Click **Load data**
  2) Choose **Baseline** or **Main** model
  3) Click **Train**
  4) Review metrics + plots
  5) Check the success message for the artifacts folder path

## Artifacts and caching

- Datasets are cached under `data/`.
- Artifacts (models, metrics, plots, logs) are saved under:
  `artifacts/<task>/<YYYYMMDD_HHMMSS>/`
- Use **Reset cache** in the sidebar if you want to re‑download datasets.

## PyTorch install note (CPU vs GPU)

- CPU-only installs work everywhere and are the default for most users.
- For NVIDIA GPU support, you must install a CUDA-enabled PyTorch build.
- Use the official PyTorch installation guide to pick the correct CUDA version and installation command for your system.

## Troubleshooting

- **Dataset download fails (offline / blocked):** ensure network access or use the cached `data/` files.
- **Forecasting data error:** the app auto‑converts M4 TSF to CSV on first download; if the zip is corrupted, delete `data/` and re‑download.
- **CUDA not detected:** the forecasting tab will fall back to CPU automatically.
- **Long installs:** torch and scipy can take time on Windows; let pip finish or use a faster mirror.

## Portfolio notes

- **Classification:** predicts income class from census features (ROC‑AUC, F1, precision, recall).
- **Regression:** predicts median housing value (MAE, RMSE).
- **Anomaly detection:** flags time‑series anomalies with a baseline z‑score and IsolationForest (precision/recall when labels exist).
- **Forecasting:** seasonal‑naive baseline vs a lightweight PyTorch N‑BEATS‑style model (MAE, sMAPE).

## Acceptance checklist

- [ ] Repo runs in one `.venv` with one install step.
- [ ] Streamlit app opens with 4 tabs.
- [ ] Each tab loads data (auto‑download + cache).
- [ ] Each tab trains baseline + main model successfully.
- [ ] Each tab shows metrics + at least one plot.
- [ ] Each tab exports artifacts to `artifacts/`.
- [ ] Forecasting tab uses GPU if detected, otherwise CPU, with clear UI message.
- [ ] No code is nested under `src/ml_portfolio_suite/`.
