"""Local UI: metrics, actual vs predicted on holdout, per-crypto latest prediction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.data_prep import build_modeling_dataframe
from src.paths import METRICS_PATH, MODEL_PATH
from src.predict import load_bundle
from src.train import time_ordered_split

st.set_page_config(page_title="Crypto Volatility", layout="wide")
st.title("Cryptocurrency volatility prediction")
st.caption(
    "Model predicts forward 5-day realized volatility (std of next five daily log returns)."
)


@st.cache_resource
def bundle():
    if not MODEL_PATH.exists():
        return None
    return load_bundle(MODEL_PATH)


@st.cache_data
def modeling_df():
    df, _le = build_modeling_dataframe()
    return df


def main():
    b = bundle()
    if b is None:
        st.error(
            f"No trained model at `{MODEL_PATH}`. Run: `python -m src.train` from the project folder."
        )
        return

    df = modeling_df()
    feat_cols = b["feature_columns"]
    X = df[feat_cols].to_numpy(dtype=float)
    y = df["target_vol"].to_numpy(dtype=float)
    train_idx, test_idx = time_ordered_split(df, test_frac=0.15)

    pred_all = b["model"].predict(X)
    y_test = y[test_idx]
    p_test = pred_all[test_idx]

    if METRICS_PATH.exists():
        with open(METRICS_PATH, encoding="utf-8") as f:
            metrics = json.load(f)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RMSE (holdout)", f"{metrics['rmse']:.6f}")
        c2.metric("MAE (holdout)", f"{metrics['mae']:.6f}")
        c3.metric("R² (holdout)", f"{metrics['r2']:.4f}")
        c4.metric("Train / test rows", f"{metrics['n_train']:,} / {metrics['n_test']:,}")

    st.subheader("Actual vs predicted (holdout sample)")
    sample = min(5000, len(test_idx))
    rng = np.random.default_rng(42)
    pick = rng.choice(len(test_idx), size=sample, replace=False)
    sc = pd.DataFrame({"actual": y_test[pick], "predicted": p_test[pick]})
    st.scatter_chart(sc, x="actual", y="predicted")

    st.subheader("Predict from latest row (by crypto)")
    names = sorted(df["crypto_name"].unique())
    choice = st.selectbox("Cryptocurrency", names)
    row = df[df["crypto_name"] == choice].sort_values("date").iloc[-1]
    x = np.array([[row[c] for c in feat_cols]], dtype=float)
    vol_hat = float(b["model"].predict(x)[0])
    c1, c2 = st.columns(2)
    c1.metric("Predicted forward 5d vol", f"{vol_hat:.6f}")
    with c2:
        st.write("As of date (UTC):", str(row["date"])[:10])
        st.write("Last close:", f"{row['close']:.6f}")


main()
