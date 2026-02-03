# ML Portfolio Suite — Implementation Plan (10 Steps)

> Audience: ChatGPT VSCode extension (AI developer).  
> Constraints: **No code in this plan**. Steps must be **small**, **specific**, and include a **test**.  
> Workflow rule: Each step is a standalone prompt. After each step’s tests pass, update `/memory-bank/progress.md` and `/memory-bank/architecture.md`.

---

## Step 1 — Project skeleton + memory-bank docs

**Prompt to AI developer**  
Read all the documents in `/memory-bank`, and proceed with Step 1 of the implementation plan. I will run the tests. Do not start Step 2 until I validate the tests. Once I validate them, open `progress.md` and document what you did for future developers. Then add any architectural insights to `architecture.md` to explain what each file does.

**Step 1 tasks**
1. In repo root (`ml_portfolio_suite/`), ensure the basic folders exist:
   - `ml/`
   - `data/`
   - `artifacts/`
   - `/memory-bank/` (if not already present)
2. In `/memory-bank/`, ensure these files exist (even if empty to start):
   - `progress.md`
   - `architecture.md`
3. Move/ensure the PRD and tech stack docs are stored under `/memory-bank/` as reference:
   - `product-requirements-document.md`
   - `tech-stack.md`
4. Ensure there is **no** `src/` nesting (specifically: no `src/ml_portfolio_suite/`).
5. Add placeholder files in repo root (empty content is fine for now):
   - `app.py`
   - `requirements.txt`
   - `README.md`

**Test (you run)**
- Confirm the following paths exist:
  - `ml_portfolio_suite/app.py`
  - `ml_portfolio_suite/ml/`
  - `ml_portfolio_suite/data/`
  - `ml_portfolio_suite/artifacts/`
  - `ml_portfolio_suite/requirements.txt`
  - `ml_portfolio_suite/README.md`
  - `ml_portfolio_suite/memory-bank/product-requirements-document.md`
  - `ml_portfolio_suite/memory-bank/tech-stack.md`
  - `ml_portfolio_suite/memory-bank/progress.md`
  - `ml_portfolio_suite/memory-bank/architecture.md`
- Confirm there is **no** `ml_portfolio_suite/src/` folder.

---

## Step 2 — Dependency plan + environment smoke check

**Prompt to AI developer**  
Read all the documents in `/memory-bank`, and proceed with Step 2 of the implementation plan. I will run the tests. Do not start Step 3 until I validate the tests. Once I validate them, open `progress.md` and document what you did for future developers. Then add any architectural insights to `architecture.md` to explain what each file does.

**Step 2 tasks**
1. Populate `requirements.txt` with the stack from `tech-stack.md`, organized into sections:
   - Core (Streamlit, numpy, pandas, scikit-learn, matplotlib, joblib, requests)
   - Forecasting NN (torch, darts)
   - Dev-only (pytest, ruff) as optional (may be included but not required to run)
2. Add a brief, Windows-first `.venv` setup section to `README.md` including:
   - how to create and activate `.venv`
   - how to install dependencies
   - how to run Streamlit
3. In `README.md`, add a “PyTorch install note” section:
   - CPU-only works for everyone
   - GPU requires CUDA-enabled PyTorch installation (do not include code; just instructions and where to look)

**Test (you run)**
1. In a fresh `.venv`, run `pip install -r requirements.txt` successfully.
2. Run `python -c "import streamlit, sklearn, torch; import darts"` with **no import errors**.
3. Run `python -c "import torch; print('cuda', torch.cuda.is_available())"` and confirm it prints a boolean (either is fine).

---

## Step 3 — Shared utilities: paths, seeding, caching, artifact runs

**Prompt to AI developer**  
Read all the documents in `/memory-bank`, and proceed with Step 3 of the implementation plan. I will run the tests. Do not start Step 4 until I validate the tests. Once I validate them, open `progress.md` and document what you did for future developers. Then add any architectural insights to `architecture.md` to explain what each file does.

**Step 3 tasks**
1. Add `ml/common.py` with small, focused utilities:
   - Ensure consistent folder paths (root, `data/`, `artifacts/`)
   - Set global random seeds for `random`, `numpy`, and `torch` (if installed)
   - A helper that creates a new timestamped artifact run directory per task:
     `artifacts/<task>/<YYYYMMDD_HHMMSS>/`
   - A small helper for “download with cache”:
     - given a URL and local filepath, download only if file not present
2. Add minimal logging:
   - If an exception occurs in training functions (later steps), they can write a traceback to `artifacts/logs/` under the run folder.

**Test (you run)**
1. Run `python -c "from ml.common import *; print('ok')"` with no errors.
2. Verify a run directory can be generated for a sample task (e.g., `classification`) and appears under `artifacts/`.
3. Verify the cached-download helper does **not** re-download if the file already exists (you can test by downloading a small file twice and confirming timestamps do not change).

---

## Step 4 — Classification module (Adult via OpenML) + CLI test

**Prompt to AI developer**  
Read all the documents in `/memory-bank`, and proceed with Step 4 of the implementation plan. I will run the tests. Do not start Step 5 until I validate the tests. Once I validate them, open `progress.md` and document what you did for future developers. Then add any architectural insights to `architecture.md` to explain what each file does.

**Step 4 tasks**
1. Implement `ml/classification.py` with clear public functions:
   - `load_data(...)`: pulls Adult dataset via OpenML and caches in `data/`
   - `train_baseline(...)`: logistic regression in a pipeline
   - `train_main_model(...)`: HistGradientBoostingClassifier in a pipeline
   - `evaluate(...)`: ROC-AUC, F1, precision, recall, confusion matrix
   - `plot_confusion_matrix(...)`: produces one plot object suitable for Streamlit
   - `export_artifacts(...)`: writes model + metrics + plot to a run folder
2. Include safeguards:
   - Handle categorical features properly (explicit encoding)
   - Validate that target has 2 classes; otherwise fail with a clear message
3. Keep runtime reasonable:
   - Include a “fast mode” behavior (e.g., downsample option) default off; Streamlit can toggle later.

**Test (you run)**
1. Run a small local script (or Python REPL) that:
   - loads data
   - trains baseline and main model
   - prints metrics
2. Confirm it completes without exceptions.
3. Confirm artifacts were written to `artifacts/classification/<timestamp>/`:
   - a model file
   - `metrics.json`
   - a confusion matrix PNG

---

## Step 5 — Regression module (California Housing) + artifact validation

**Prompt to AI developer**  
Read all the documents in `/memory-bank`, and proceed with Step 5 of the implementation plan. I will run the tests. Do not start Step 6 until I validate the tests. Once I validate them, open `progress.md` and document what you did for future developers. Then add any architectural insights to `architecture.md` to explain what each file does.

**Step 5 tasks**
1. Implement `ml/regression.py` with public functions similar to classification:
   - `load_data(...)`: uses scikit’s California housing fetch and caches
   - `train_baseline(...)`: Ridge regression pipeline
   - `train_main_model(...)`: HistGradientBoostingRegressor pipeline
   - `evaluate(...)`: MAE, RMSE
   - `plot_residuals(...)`: residual plot
   - `export_artifacts(...)`: model + metrics + plot
2. Add data validation:
   - Ensure no missing target values
   - Ensure feature matrix is numeric (handle if needed)
3. Keep defaults fast (no heavy tuning).

**Test (you run)**
1. Execute a small script that trains baseline + main regression model and prints MAE/RMSE.
2. Confirm artifacts appear under `artifacts/regression/<timestamp>/`:
   - model file
   - `metrics.json`
   - residual plot PNG

---

## Step 6 — Anomaly module (NAB subset) + visual validation

**Prompt to AI developer**  
Read all the documents in `/memory-bank`, and proceed with Step 6 of the implementation plan. I will run the tests. Do not start Step 7 until I validate the tests. Once I validate them, open `progress.md` and document what you did for future developers. Then add any architectural insights to `architecture.md` to explain what each file does.

**Step 6 tasks**
1. Implement `ml/anomaly.py` with:
   - `download_nab_subset(...)`: downloads a small NAB subset into `data/`
   - `load_series(...)`: reads one NAB CSV series and its labels/windows (if provided)
   - `baseline_zscore(...)`: rolling z-score anomaly baseline
   - `train_main_model(...)`: IsolationForest approach
     - define a simple feature engineering method (window statistics) suitable for IsolationForest
   - `evaluate(...)`: precision/recall (when labels exist) and a time-series overlay plot
   - `export_artifacts(...)`
2. Provide a stable “default series” for the demo (small and quick).
3. Ensure the module can run even if label windows are missing:
   - If labels are unavailable for that series, show unsupervised results without precision/recall, but still export plots and scores.

**Test (you run)**
1. Run a script that:
   - downloads data (first time)
   - loads the default series
   - runs z-score baseline + IsolationForest
   - produces and saves the time-series plot with anomaly markers
2. Confirm artifacts under `artifacts/anomaly/<timestamp>/` including a plot PNG.
3. If labels exist, confirm precision/recall are present in `metrics.json`.

---

## Step 7 — Forecasting module (M4 subset + N-BEATS) with GPU fallback

**Prompt to AI developer**  
Read all the documents in `/memory-bank`, and proceed with Step 7 of the implementation plan. I will run the tests. Do not start Step 8 until I validate the tests. Once I validate them, open `progress.md` and document what you did for future developers. Then add any architectural insights to `architecture.md` to explain what each file does.

**Step 7 tasks**
1. Implement `ml/forecasting.py` with:
   - `download_m4_subset(...)`: downloads a small subset (monthly or daily) and caches in `data/`
   - `load_series_list(...)`: lists available series IDs
   - `train_baseline(...)`: seasonal naive (or a simple ETS/ARIMA baseline if stable)
   - `train_nbeats(...)`: uses Darts N-BEATS model
   - `get_device(...)`: returns `"cuda"` if available otherwise `"cpu"` and surfaces this to UI
   - `evaluate(...)`: MAE + sMAPE on a holdout window; optional rolling evaluation if simple
   - `plot_forecast(...)`: history + forecast plot
   - `export_artifacts(...)`: save model + metrics + plot
2. Keep defaults fast:
   - small subset
   - small epochs
   - modest model size
3. Implement clean fallback:
   - If CUDA not available, proceed on CPU with a UI-visible message later.

**Test (you run)**
1. Run a script that:
   - loads a single default series
   - trains baseline
   - trains N-BEATS for a small number of epochs
   - prints device used and metrics
2. Confirm artifacts under `artifacts/forecasting/<timestamp>/` including:
   - saved NN model artifact
   - `metrics.json`
   - forecast plot PNG

---

## Step 8 — Build Streamlit app with four tabs + integration tests

**Prompt to AI developer**  
Read all the documents in `/memory-bank`, and proceed with Step 8 of the implementation plan. I will run the tests. Do not start Step 9 until I validate the tests. Once I validate them, open `progress.md` and document what you did for future developers. Then add any architectural insights to `architecture.md` to explain what each file does.

**Step 8 tasks**
1. Implement `app.py` Streamlit UI:
   - Title and short overview
   - Four tabs matching PRD
2. Each tab includes:
   - “Load data” button
   - model selection (baseline vs main)
   - inputs with defaults (fast, safe)
   - “Train” button
   - metrics display
   - plot display
   - artifact export confirmation with file paths
3. Forecasting tab must display:
   - `torch.cuda.is_available()` result and selected device
4. Add “Reset cache” UI controls (optional, but small and useful).

**Test (you run)**
1. Run `streamlit run app.py` and confirm app opens.
2. For each tab:
   - Load dataset
   - Train baseline
   - Train main model
   - Confirm no unhandled exceptions
3. Confirm artifact folders populate correctly for each task.

---

## Step 9 — Hardening pass: common failure modes on Windows

**Prompt to AI developer**  
Read all the documents in `/memory-bank`, and proceed with Step 9 of the implementation plan. I will run the tests. Do not start Step 10 until I validate the tests. Once I validate them, open `progress.md` and document what you did for future developers. Then add any architectural insights to `architecture.md` to explain what each file does.

**Step 9 tasks**
1. Add robust exception handling around:
   - dataset downloads
   - model training
   - plotting
   - artifact writing
2. Ensure every exception:
   - shows a concise error in Streamlit
   - writes a full traceback log into the run folder
3. Verify path handling uses `pathlib` everywhere.
4. Ensure matplotlib works headless under Streamlit without requiring special backends.
5. Add a small “health check” section in the UI (or in a separate module) that confirms:
   - imports work
   - data dirs writable
   - artifact dirs writable

**Test (you run)**
1. Disconnect network and attempt to load a dataset:
   - Confirm you get a clear error message and no crash.
2. Make `artifacts/` read-only and attempt export:
   - Confirm the app shows a clear error and writes a log.
3. Re-enable normal permissions and confirm exports work again.

---

## Step 10 — Documentation polish + acceptance checklist completion

**Prompt to AI developer**  
Read all the documents in `/memory-bank`, and proceed with Step 10 of the implementation plan. I will run the tests. Do not start any new steps after this. Once I validate the tests, open `progress.md` and document what you did for future developers. Then add any architectural insights to `architecture.md` to explain what each file does.

**Step 10 tasks**
1. Finalize `README.md`:
   - Windows-first setup instructions
   - Common troubleshooting (CUDA, long installs, dataset download)
   - How to run each task via UI
   - Where artifacts are saved
2. Add a “Portfolio notes” section:
   - what each tab demonstrates
   - what metrics mean (short, client-friendly)
3. Add or finalize an acceptance checklist section:
   - copy from PRD and ensure each item is verifiable
4. Ensure `memory-bank/architecture.md` explains:
   - each file in `ml/`
   - how `app.py` calls each module
   - how caching and artifacts work

**Test (you run)**
1. Fresh clone simulation (or delete `.venv`, delete `data/` and `artifacts/`):
   - recreate `.venv`
   - install deps
   - run Streamlit
2. Validate all PRD acceptance checklist items are satisfied.
3. Confirm README is sufficient for a new user to run the project without extra help.

---

# Output file checklist
This plan expects these files to exist by the end:
- `app.py`
- `ml/common.py`
- `ml/classification.py`
- `ml/regression.py`
- `ml/anomaly.py`
- `ml/forecasting.py`
- `requirements.txt`
- `README.md`
- `memory-bank/progress.md`
- `memory-bank/architecture.md`
- `memory-bank/product-requirements-document.md`
- `memory-bank/tech-stack.md`
