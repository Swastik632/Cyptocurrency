"""EDA: summary stats and figures under reports/figures/. Run: python -m src.eda"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data_prep import load_raw
from src.features import add_features, feature_columns
from src.paths import FIGURES


def main():
    sns.set_theme(style="whitegrid")
    raw = load_raw()
    df = add_features(raw)
    feat_cols = feature_columns()

    summary = df[feat_cols + ["target_vol"]].describe()
    summary.to_csv(FIGURES.parent / "eda_summary_stats.csv")

    # Sample for correlation heatmap (full matrix is fine for ~12 cols)
    corr = df[feat_cols + ["target_vol"]].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="vlag", center=0)
    plt.title("Feature / target correlation")
    plt.tight_layout()
    plt.savefig(FIGURES / "correlation_heatmap.png", dpi=120)
    plt.close()

    top = (
        raw.groupby("crypto_name")["date"]
        .count()
        .sort_values(ascending=False)
        .head(12)
        .index.tolist()
    )
    sub = df[df["crypto_name"].isin(top)].copy()
    plt.figure(figsize=(11, 5))
    for name in top[:4]:
        g = sub[sub["crypto_name"] == name].sort_values("date")
        plt.plot(g["date"], g["close"], label=name, alpha=0.85)
    plt.legend()
    plt.title("Close price (sample cryptos)")
    plt.ylabel("Close")
    plt.tight_layout()
    plt.savefig(FIGURES / "close_trends_sample.png", dpi=120)
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.histplot(df["target_vol"].dropna(), bins=60, kde=True)
    plt.title("Distribution of target: forward 5d realized volatility")
    plt.tight_layout()
    plt.savefig(FIGURES / "target_distribution.png", dpi=120)
    plt.close()

    print("Wrote:", FIGURES / "correlation_heatmap.png")
    print("Wrote:", FIGURES / "close_trends_sample.png")
    print("Wrote:", FIGURES / "target_distribution.png")
    print("Wrote:", FIGURES.parent / "eda_summary_stats.csv")


if __name__ == "__main__":
    main()
