"""Feature engineering and target: forward realized volatility."""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def forward_realized_vol(
    rets: pd.Series, window: int = 5, min_periods: int | None = None
) -> pd.Series:
    """Std of daily log returns over the next `window` days (excludes current day)."""
    if min_periods is None:
        min_periods = window
    r = rets.to_numpy(dtype=float)
    n = len(r)
    out = np.full(n, np.nan, dtype=float)
    for i in range(n - window):
        chunk = r[i + 1 : i + 1 + window]
        if np.isnan(chunk).any():
            continue
        out[i] = float(np.std(chunk, ddof=0))
    return pd.Series(out, index=rets.index)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-crypto OHLCV features using only past/current rows (no lookahead)."""
    out = []
    for name, g in df.groupby("crypto_name", sort=False):
        g = g.sort_values("date").copy()
        c, h, l, o = g["close"], g["high"], g["low"], g["open"]
        v, m = g["volume"], g["marketCap"]

        lr = log_returns(c)
        g["log_ret"] = lr
        g["roll_vol_5"] = lr.rolling(5, min_periods=3).std()
        g["roll_vol_10"] = lr.rolling(10, min_periods=5).std()
        g["roll_vol_20"] = lr.rolling(20, min_periods=10).std()

        ma7 = c.rolling(7, min_periods=3).mean()
        ma20 = c.rolling(20, min_periods=5).mean()
        g["close_ma7_ratio"] = c / ma7 - 1.0
        g["close_ma20_ratio"] = c / ma20 - 1.0

        std20 = c.rolling(20, min_periods=5).std()
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        bb_mid = ma20.replace(0, np.nan)
        g["bb_width"] = (upper - lower) / bb_mid

        g["hl_range_pct"] = (h - l) / c.replace(0, np.nan)
        g["co_range_pct"] = (c - o).abs() / o.replace(0, np.nan)

        m_safe = m.replace(0, np.nan)
        g["vol_mcap_ratio"] = v / m_safe
        g["log_mcap"] = np.log1p(m.clip(lower=0))

        g["target_vol"] = forward_realized_vol(lr, window=5)

        out.append(g)

    return pd.concat(out, ignore_index=True)


def feature_columns() -> list[str]:
    return [
        "log_ret",
        "roll_vol_5",
        "roll_vol_10",
        "roll_vol_20",
        "close_ma7_ratio",
        "close_ma20_ratio",
        "bb_width",
        "hl_range_pct",
        "co_range_pct",
        "vol_mcap_ratio",
        "log_mcap",
    ]
