## Architecture overview (current)

- `app.py`: Streamlit UI with four tabs; each tab calls the corresponding `ml/` module to load data, train models, evaluate, plot, and export artifacts. Includes health check and error logging.
- `ml/common.py`: shared utilities for folder paths, seeding, cached downloads, run folder creation, and traceback logging.
- `ml/classification.py`: Adult dataset classification pipeline (preprocess → baseline/main → metrics → confusion matrix → export).
- `ml/regression.py`: California Housing regression pipeline (baseline/main → MAE/RMSE → residual plot → export).
- `ml/anomaly.py`: NAB anomaly detection pipeline (download → load series → z‑score baseline / IsolationForest → metrics/plot → export).
- `ml/anomaly.py` handles NAB label windows and single timestamp labels when present.
- `ml/forecasting.py`: M4 subset forecasting with seasonal‑naive baseline and a small PyTorch N‑BEATS‑style model (train → MAE/sMAPE → plot → export).
- `data/`: dataset cache. Clearing `data/` forces re‑download on next load.
- `artifacts/`: run outputs saved under `artifacts/<task>/<timestamp>/` (models, metrics, plots, logs).
- `requirements.txt` / `README.md`: dependency list and Windows‑first setup/run instructions.
- `memory-bank/`: PRD, tech stack, plan, and progress notes.
