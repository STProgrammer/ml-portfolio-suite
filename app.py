from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import streamlit as st

from ui.layout import render_sidebar
from ui.tabs import (
    render_anomaly_tab,
    render_classification_tab,
    render_forecasting_tab,
    render_regression_tab,
)


st.set_page_config(page_title="ML Portfolio Suite", layout="wide")
st.title("ML Portfolio Suite")
st.write(
    "A single Streamlit app showcasing four applied ML tasks: classification, regression, "
    "anomaly detection, and forecasting."
)

render_sidebar()

(tab_class, tab_reg, tab_anom, tab_forecast) = st.tabs(
    ["Classification", "Regression", "Anomaly Detection", "Forecasting (NN)"]
)

with tab_class:
    render_classification_tab()

with tab_reg:
    render_regression_tab()

with tab_anom:
    render_anomaly_tab()

with tab_forecast:
    render_forecasting_tab()
