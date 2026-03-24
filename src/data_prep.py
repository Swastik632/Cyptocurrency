"""Load CSV, clean, encode crypto, build modeling frame."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.features import add_features, feature_columns
from src.paths import DATA_CSV


def load_raw(csv_path=DATA_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    for col in ("open", "high", "low", "close", "volume", "marketCap"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "crypto_name", "open", "high", "low", "close"])
    df = df[df["close"] > 0]
    df = df.sort_values(["crypto_name", "date"]).reset_index(drop=True)
    return df


def build_modeling_dataframe(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, LabelEncoder]:
    if df is None:
        df = load_raw()
    df = add_features(df)
    feat_cols = feature_columns()
    df = df.dropna(subset=feat_cols + ["target_vol"])

    le = LabelEncoder()
    df = df.copy()
    df["crypto_id"] = le.fit_transform(df["crypto_name"].astype(str))

    return df, le
