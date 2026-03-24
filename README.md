# Cryptocurrency volatility prediction

ML pipeline on daily OHLCV + market cap data: predicts **forward 5-day realized volatility** (standard deviation of the next five daily log returns) per coin.

## Setup

```bash
pip install -r requirements.txt
```

Place `dataset.csv` in the project root (columns: `open`, `high`, `low`, `close`, `volume`, `marketCap`, `timestamp`, `crypto_name`, `date`).

## Run

```bash
python -m src.train
```

Writes `artifacts/volatility_model.joblib`, `artifacts/metrics.json`, and `artifacts/feature_names.json`.

```bash
python -m src.eda
```

Writes summary stats and plots under `reports/`.

```bash
streamlit run streamlit_app.py
```

Shows holdout RMSE / MAE / R², an actual-vs-predicted scatter sample, and a per-crypto “latest row” prediction.

## Features (engineered)

Rolling log-return volatility (5/10/20d), MA ratios, Bollinger bandwidth, high–low range vs close, candle body vs open, volume-to-market-cap ratio, log market cap, plus `crypto_id` from label encoding.

## Model

`HistGradientBoostingRegressor` on time-ordered split (last 15% of rows by date as test). Tune hyperparameters in `src/train.py` if needed.

## Word documentation (HLD / LLD / pipeline / EDA / final report)

```bash
pip install python-docx
python scripts/build_word_docs.py
```

Creates `.docx` files under `docs/` (open in Microsoft Word or compatible editors).
