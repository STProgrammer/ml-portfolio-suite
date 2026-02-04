from __future__ import annotations

import streamlit as st
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
)

from ml import classification as clf
from ml import regression as reg
from ml import anomaly as anom
from ml import forecasting as fc
from ml.common import new_run_dir, log_exception


def start_card() -> None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)


def end_card() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def safe_classification_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    pos_label = sorted(y_test.unique())[-1]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        classes = list(model.classes_)
        pos_index = classes.index(pos_label)
        y_score = proba[:, pos_index]
    else:
        y_score = model.decision_function(X_test)
    y_test_bin = (y_test == pos_label).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test_bin, y_score)),
        "f1": float(f1_score(y_test, y_pred, pos_label=pos_label)),
        "precision": float(precision_score(y_test, y_pred, pos_label=pos_label)),
        "recall": float(recall_score(y_test, y_pred, pos_label=pos_label)),
    }
    cm = confusion_matrix(y_test, y_pred, labels=sorted(y_test.unique()))
    return metrics, cm


def safe_regression_metrics(model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    return {"mae": float(mae), "rmse": float(rmse)}


def render_classification_tab() -> None:
    st.subheader("Classification — Adult Income")
    start_card()
    st.markdown(
        """
        **What this data is about**
        - A census-style dataset about people and jobs.
        - Each row is one person with features like age, education, occupation, and hours worked.
        - The goal is to predict if income is **above $50K** or **$50K or less**.
        - This is a classic **binary classification** problem (only two possible answers).

        **What this tab does**
        - Trains a model to guess the income group from the features.
        - Lets you compare a simple model vs a stronger one.
        - Shows how well the model separates the two groups using multiple metrics.
        """
    )
    end_card()
    with st.expander("How to read the results (simple language)"):
        st.markdown(
            """
            **Metrics**
            - **ROC‑AUC**: how well the model can separate the two income groups.
              1.0 is perfect, 0.5 is like guessing.
            - **F1**: one score that balances “how many it finds” and “how many are correct.”
            - **Precision**: if the model says “>50K”, how often that is correct.
            - **Recall**: out of all real “>50K” people, how many the model found.
            - Tip: precision focuses on **not over‑calling** >50K, recall focuses on **not missing** >50K.

            **Confusion matrix (the square table)**
            - The table has 4 boxes (2 classes × 2 predictions).
            - The **diagonal** boxes are correct predictions:
              - Top‑left = correctly predicted “<=50K”
              - Bottom‑right = correctly predicted “>50K”
            - The **off‑diagonal** boxes are mistakes:
              - Top‑right = predicted “>50K” but actually “<=50K”
              - Bottom‑left = predicted “<=50K” but actually “>50K”
            - If off‑diagonal numbers are large, the model is confusing the classes.
            - If the diagonal numbers are large, the model is doing well.
            """
        )
    fast_mode = st.checkbox("Fast mode (downsample)", value=False, key="clf_fast")
    if st.button("Load data", key="clf_load"):
        try:
            st.session_state["clf_data"] = clf.load_data(fast_mode=fast_mode)
            st.success("Classification dataset loaded.")
        except Exception as exc:
            run_dir = new_run_dir("classification")
            log_path = log_exception(run_dir, exc)
            st.error(f"Load failed: {exc}")
            st.caption(f"Traceback saved to: {log_path}")

    model_choice = st.radio(
        "Model", ["Baseline (Logistic Regression)", "Main (HistGradientBoosting)"], key="clf_model"
    )

    if st.button("Train", key="clf_train"):
        data = st.session_state.get("clf_data")
        if data is None:
            st.warning("Load data first.")
        else:
            run_dir = new_run_dir("classification")
            try:
                X_train, X_test, y_train, y_test = clf.train_test_split_data(data)
                if model_choice.startswith("Baseline"):
                    model = clf.train_baseline(X_train, y_train)
                else:
                    model = clf.train_main_model(X_train, y_train)
                try:
                    metrics, cm = clf.evaluate(model, X_test, y_test)
                except Exception:
                    metrics, cm = safe_classification_metrics(model, X_test, y_test)
                labels = sorted(y_test.unique())
                fig = clf.plot_confusion_matrix(cm, tuple(labels))
                run_dir = clf.export_artifacts(model, metrics, fig, run_dir=run_dir)
                st.json(metrics)
                st.pyplot(fig, width="stretch")
                st.success(f"Artifacts saved to: {run_dir}")
            except Exception as exc:
                log_path = log_exception(run_dir, exc)
                st.error(f"Training failed: {exc}")
                st.caption(f"Traceback saved to: {log_path}")


def render_regression_tab() -> None:
    st.subheader("Regression — California Housing")
    start_card()
    st.markdown(
        """
        **What this data is about**
        - California Housing data about neighborhoods.
        - Each row is a neighborhood with features like income, rooms, and population.
        - The goal is to predict the **median house price** (a number, not a class).
        - This is a **regression** problem.

        **What this tab does**
        - Trains a model to predict a number.
        - Lets you compare a simple model vs a stronger one.
        - Shows how far predictions are from the real prices.
        """
    )
    end_card()
    with st.expander("How to read the results (simple language)"):
        st.markdown(
            """
            **Metrics**
            - **MAE**: the average size of the error. Smaller is better.
            - **RMSE**: like MAE but punishes big mistakes more.
            - If RMSE is much larger than MAE, the model sometimes makes big errors.

            **Residual plot**
            - Each dot is one prediction.
            - **Up** means the model predicted too low.
            - **Down** means the model predicted too high.
            - A good model looks like a random cloud around 0.
            - A curve or funnel shape means the model is missing a pattern.
            """
        )
    fast_mode = st.checkbox("Fast mode (downsample)", value=False, key="reg_fast")
    if st.button("Load data", key="reg_load"):
        try:
            st.session_state["reg_data"] = reg.load_data(fast_mode=fast_mode)
            st.success("Regression dataset loaded.")
        except Exception as exc:
            run_dir = new_run_dir("regression")
            log_path = log_exception(run_dir, exc)
            st.error(f"Load failed: {exc}")
            st.caption(f"Traceback saved to: {log_path}")

    model_choice = st.radio(
        "Model", ["Baseline (Ridge)", "Main (HistGradientBoosting)"], key="reg_model"
    )

    if st.button("Train", key="reg_train"):
        data = st.session_state.get("reg_data")
        if data is None:
            st.warning("Load data first.")
        else:
            run_dir = new_run_dir("regression")
            try:
                X_train, X_test, y_train, y_test = reg.train_test_split_data(data)
                if model_choice.startswith("Baseline"):
                    model = reg.train_baseline(X_train, y_train)
                else:
                    model = reg.train_main_model(X_train, y_train)
                try:
                    metrics = reg.evaluate(model, X_test, y_test)
                except Exception:
                    metrics = safe_regression_metrics(model, X_test, y_test)
                fig = reg.plot_residuals(model, X_test, y_test)
                run_dir = reg.export_artifacts(model, metrics, fig, run_dir=run_dir)
                st.json(metrics)
                st.pyplot(fig, width="stretch")
                st.success(f"Artifacts saved to: {run_dir}")
            except Exception as exc:
                log_path = log_exception(run_dir, exc)
                st.error(f"Training failed: {exc}")
                st.caption(f"Traceback saved to: {log_path}")


def render_anomaly_tab() -> None:
    st.subheader("Anomaly Detection — NAB Subset")
    start_card()
    st.markdown(
        """
        **What this data is about**
        - A time series (values over time).
        - We try to find points that look **unusual** compared to recent behavior.
        - Think of sudden spikes, drops, or changes in pattern.

        **What this tab does**
        - Baseline: flags points far from recent average (simple rule).
        - Main model: learns what “normal” windows look like (machine learning).
        - Both are **unsupervised** unless labels are provided.
        """
    )
    end_card()
    with st.expander("How to read the results (simple language)"):
        st.markdown(
            """
            **Plot**
            - The line shows the values over time.
            - Red dots are points the model thinks look unusual.
            - If red dots appear in a sudden spike or drop, that is a clearer anomaly.
            - If red dots are scattered randomly, the detector may be too sensitive.

            **Metrics**
            - If labels exist: precision/recall tell how correct the red dots are.
            - If labels are missing: we only show how many dots were flagged.
            - Fewer dots is not always better — you want **meaningful** dots.
            """
        )
    use_labels = st.checkbox("Use label windows if available", value=False, key="anom_labels")
    if st.button("Load data", key="anom_load"):
        try:
            csv_path, labels_path = anom.download_nab_subset()
            label_path = labels_path if use_labels else None
            st.session_state["anom_data"] = anom.load_series(
                csv_path, label_path, series_key=anom.DEFAULT_SERIES
            )
            st.success("Anomaly series loaded.")
        except Exception as exc:
            run_dir = new_run_dir("anomaly")
            log_path = log_exception(run_dir, exc)
            st.error(f"Load failed: {exc}")
            st.caption(f"Traceback saved to: {log_path}")

    model_choice = st.radio("Model", ["Baseline (Z-score)", "Main (IsolationForest)"], key="anom_model")

    if st.button("Train", key="anom_train"):
        data = st.session_state.get("anom_data")
        if data is None:
            st.warning("Load data first.")
        else:
            run_dir = new_run_dir("anomaly")
            try:
                if model_choice.startswith("Baseline"):
                    flags = anom.baseline_zscore(data.values)
                    metrics, fig = anom.evaluate(
                        data.timestamps, data.values, flags.loc[flags.index], data.labels
                    )
                    run_dir = anom.export_artifacts(None, metrics, fig, run_dir=run_dir)
                else:
                    model, flags = anom.train_main_model(data.values)
                    metrics, fig = anom.evaluate(data.timestamps, data.values, flags, data.labels)
                    run_dir = anom.export_artifacts(model, metrics, fig, run_dir=run_dir)
                st.json(metrics)
                st.pyplot(fig, width="stretch")
                st.success(f"Artifacts saved to: {run_dir}")
            except Exception as exc:
                log_path = log_exception(run_dir, exc)
                st.error(f"Training failed: {exc}")
                st.caption(f"Traceback saved to: {log_path}")


def render_forecasting_tab() -> None:
    st.subheader("Forecasting — M4 Subset (PyTorch)")
    start_card()
    st.markdown(
        """
        **What this data is about**
        - A monthly time series (numbers over time).
        - We try to **predict future values** based on the past.
        - This is the classic forecasting task.

        **What this tab does**
        - Baseline: repeats the last seasonal pattern (simple and fast).
        - Main model: a small PyTorch neural net (more flexible).
        - Compares how close predictions are to the real future.
        """
    )
    end_card()
    with st.expander("How to read the results (simple language)"):
        st.markdown(
            """
            **Metrics**
            - **MAE**: average error size. Smaller is better.
            - **sMAPE**: percent‑style error. Smaller is better.
            - If MAE and sMAPE are both low, the forecast is close to the truth.

            **Forecast plot**
            - **History** = past data the model learned from.
            - **Actual** = the real future values.
            - **Forecast** = what the model predicted.
            - If forecast is close to actual, the model is good.
            - If forecast is flat but actual goes up/down, the model is missing the trend.
            """
        )
    st.write(f"CUDA available: {fc.get_device() == 'cuda'} | Device: {fc.get_device()}")
    if st.button("Prepare data", key="fc_prepare"):
        try:
            fc.download_m4_subset()
            st.success("Forecasting data ready.")
        except Exception as exc:
            run_dir = new_run_dir("forecasting")
            log_path = log_exception(run_dir, exc)
            st.error(f"Download failed: {exc}")
            st.caption(f"Traceback saved to: {log_path}")

    series_id = fc.DEFAULT_SERIES_ID

    epochs = st.slider("Epochs (N-BEATS)", min_value=1, max_value=25, value=8, step=1)
    horizon = st.slider("Forecast horizon (steps)", min_value=6, max_value=60, value=24, step=1)

    model_choice = st.radio("Model", ["Baseline (Seasonal Naive)", "Main (N-BEATS)"], key="fc_model")

    if st.button("Train", key="fc_train"):
        try:
            run_dir = new_run_dir("forecasting")
            bundle = fc.load_series(series_id)
            if model_choice.startswith("Baseline"):
                model = fc.train_baseline(bundle.series)
            else:
                model = fc.train_nbeats(bundle.series, epochs=epochs)
            metrics = fc.evaluate(model, bundle.series, horizon=horizon)
            fig = fc.plot_forecast(model, bundle.series, horizon=horizon)
            run_dir = fc.export_artifacts(model, metrics, fig, run_dir=run_dir)
            st.json(metrics)
            st.pyplot(fig, width="stretch")
            st.success(f"Artifacts saved to: {run_dir}")
        except Exception as exc:
            run_dir = new_run_dir("forecasting")
            log_path = log_exception(run_dir, exc)
            st.error(f"Training failed: {exc}")
            st.caption(f"Traceback saved to: {log_path}")
