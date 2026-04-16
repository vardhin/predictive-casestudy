# Case Study 16 — Rossmann Store Sales Forecasting

**22BDS0114 — GSVARDHIN**

End-to-end retail-sales forecasting on the **Rossmann Store Sales** dataset
(Kaggle): EDA, decomposition, stationarity testing, five forecasting models,
walk-forward evaluation, and an interactive Streamlit UI.

---

## Why this case study

The original AirPassengers (144 monthly points) is a textbook toy. Real retail
forecasting is *messier* — multiple stores, daily granularity, promos,
holidays, store-type heterogeneity, and zero-sales days when stores close.
Rossmann gives us all of that on **1,115 stores × 942 daily observations**
(~1M rows), so we can compare classical and ML models on a realistic problem.

---

## Dataset

| Detail | Value |
|---|---|
| Source | [Rossmann Store Sales — Kaggle](https://www.kaggle.com/c/rossmann-store-sales) |
| Files used | `train.csv` (sales), `store.csv` (store metadata) |
| Stores | 1,115 |
| Daily observations | 1,017,209 rows |
| Period | 2013-01-01 → 2015-07-31 |
| Target | `Sales` (€) |
| Exogenous | `Promo`, `SchoolHoliday`, `StateHoliday`, `Promo2`, `CompetitionDistance`, `StoreType`, `Assortment` |

**The CSVs are not committed.** Download from Kaggle and place `train.csv`,
`store.csv`, (optionally `test.csv`) in the project root.

---

## Architecture

```
predictive-casestudy/
├── main.py              # CLI: runs full headless pipeline, saves PNG/CSV to outputs/
├── app.py               # Streamlit multi-page UI
├── src/
│   ├── data.py          # Load, merge, feature engineering (cached as parquet)
│   ├── eda.py           # Plotly figures + ADF stationarity test
│   └── models.py        # Naive, Holt-Winters, SARIMA, Prophet, XGBoost + walk-forward CV
├── pyproject.toml
├── train.csv  store.csv  test.csv      # (gitignored — fetched from Kaggle)
├── cache/                              # parquet cache of engineered data
└── outputs/                            # generated charts + leaderboard CSVs
```

---

## Models compared

| Model | Type | Notes |
|---|---|---|
| **Naive (seasonal-7)** | Baseline | `y_t = y_{t-7}` — must be beaten |
| **Holt-Winters** | Classical ETS | Triple exponential smoothing, weekly seasonality |
| **SARIMA(1,1,1)(1,1,1,7)** | Classical | Seasonal ARIMA on daily data |
| **Prophet** | Decomposable | Trend + weekly + yearly + Promo/School regressors |
| **XGBoost** | ML | Lag (1/7/14/28) + rolling stats + calendar/promo features, recursive forecast |

All models share a common `fit(train) → predict(future_index)` interface in
[src/models.py](src/models.py), so adding a new model is one class.

### Metrics
MAE · RMSE · MAPE · **RMSPE** (Rossmann competition's official metric).
Walk-forward evaluation across `n` folds is available via
`models.walk_forward_eval()`.

---

## How to run

```bash
# Install deps (uv handles venv automatically)
uv sync

# Headless analysis on the network-wide aggregate (writes to ./outputs/)
uv run main.py

# Or analyse a single store:
uv run main.py 1 28        # store_id=1, horizon=28 days

# Interactive Streamlit UI
uv run streamlit run app.py
```

The Streamlit app has six pages — Overview, EDA, Decomposition & Stationarity,
Forecasting, Model Comparison, Future Forecast — and a sidebar to switch
between **Network-wide** and **Single-store** scope.

---

## Limitations

- Holt-Winters & SARIMA assume regular sampling and can be slow on long daily
  series — we cap CV folds and horizons accordingly.
- XGBoost recursive forecasting compounds errors over long horizons; for >30 days
  it loses to Holt-Winters on the network-wide series.
- Exogenous regressors (`Promo`, `SchoolHoliday`) need to be known in advance for
  out-of-sample prediction; we substitute day-of-week medians for the future.
- We do not model store closures during the Rossmann refurbishment in 2014; the
  raw zeros are filtered out by `only_open=True` in the per-store path.
