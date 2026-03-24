"""Load saved model and predict on prepared rows."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.paths import MODEL_PATH


def load_bundle(path: Path = MODEL_PATH):
    return joblib.load(path)


def predict_row(bundle: dict, feature_row: dict) -> float:
    cols = bundle["feature_columns"]
    x = np.array([[feature_row[c] for c in cols]], dtype=float)
    return float(bundle["model"].predict(x)[0])


def latest_features_for_crypto(
    df_features: pd.DataFrame, crypto: str
) -> pd.Series | None:
    sub = df_features[df_features["crypto_name"] == crypto].sort_values("date")
    if sub.empty:
        return None
    return sub.iloc[-1]
