"""Train volatility regressor, save model and metrics."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_prep import build_modeling_dataframe
from src.features import feature_columns
from src.paths import FEATURE_NAMES_PATH, METRICS_PATH, MODEL_PATH


def time_ordered_split(
    df, date_col: str = "date", test_frac: float = 0.15
) -> tuple[np.ndarray, np.ndarray]:
    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    cut = int(n * (1.0 - test_frac))
    idx = np.arange(n)
    return idx[:cut], idx[cut:]


def train_and_save(
    test_frac: float = 0.15,
    random_state: int = 42,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
):
    df, le = build_modeling_dataframe()
    feat_cols = feature_columns() + ["crypto_id"]
    X = df[feat_cols].to_numpy(dtype=float)
    y = df["target_vol"].to_numpy(dtype=float)

    train_idx, test_idx = time_ordered_split(df, test_frac=test_frac)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = HistGradientBoostingRegressor(
        max_depth=8,
        learning_rate=0.06,
        max_iter=200,
        min_samples_leaf=40,
        l2_regularization=1e-3,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mae = float(mean_absolute_error(y_test, pred))
    r2 = float(r2_score(y_test, pred))

    joblib.dump({"model": model, "label_encoder": le, "feature_columns": feat_cols}, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "target": "forward_5d_realized_volatility (std of next 5 daily log returns)",
            },
            f,
            indent=2,
        )
    with open(FEATURE_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump({"features": feat_cols}, f, indent=2)

    return {"rmse": rmse, "mae": mae, "r2": r2}


if __name__ == "__main__":
    m = train_and_save()
    print(json.dumps(m, indent=2))
