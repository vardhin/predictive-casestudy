"""
Case Study 16 — Rossmann Store Sales Forecasting
22BDS0114 — GSVARDHIN

CLI entry point. For the interactive UI, run:

    uv run streamlit run app.py

This script runs the headless analysis end-to-end on the network-wide aggregate
series and saves the figures + a metrics table to ./outputs/.
"""

from __future__ import annotations

import sys
from pathlib import Path
import warnings

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import data as data_mod
from src import eda as eda_mod
from src import models as models_mod

warnings.filterwarnings("ignore")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


def banner(title: str) -> None:
    bar = "=" * 64
    print(f"\n{bar}\n  {title}\n{bar}")


def save_plotly(fig, name: str) -> None:
    try:
        fig.write_image(OUT / f"{name}.png", scale=2)
    except Exception:
        # kaleido may not be installed; skip silently
        pass
    fig.write_html(OUT / f"{name}.html")


def main(store_id: int | None = None, horizon: int = 28) -> None:
    banner("Loading Rossmann data")
    df = data_mod.load_full()
    print(data_mod.quick_stats(df))

    if store_id is None:
        agg = data_mod.aggregate_total(df, freq="D")
        series = agg[["Sales"]].rename(columns={"Sales": "Sales"})
        # Add network-level exogenous proxies
        promo_share = df[df["Open"] == 1].groupby("Date")["Promo"].mean()
        series["Promo"] = promo_share.reindex(series.index).fillna(0)
        series["SchoolHoliday"] = df.groupby("Date")["SchoolHoliday"].mean().reindex(series.index).fillna(0)
        series["StateHolidayFlag"] = df.groupby("Date")["StateHolidayFlag"].mean().reindex(series.index).fillna(0)
        series["Promo2Active"] = df.groupby("Date")["Promo2Active"].mean().reindex(series.index).fillna(0)
        label = "network"
    else:
        sub = data_mod.daily_series(df, store_id, only_open=True)
        series = sub[["Sales", "Promo", "SchoolHoliday", "StateHolidayFlag", "Promo2Active"]].copy()
        label = f"store_{store_id}"

    series = series.asfreq("D").interpolate()

    banner("EDA → outputs/")
    save_plotly(eda_mod.fig_total_sales(series), "01_total_sales")
    save_plotly(eda_mod.fig_rolling(series, window=30), "02_rolling")
    save_plotly(eda_mod.fig_dow_box(df), "03_dow_box")
    save_plotly(eda_mod.fig_month_box(df), "04_month_box")
    save_plotly(eda_mod.fig_promo_effect(df), "05_promo")
    save_plotly(eda_mod.fig_storetype(df), "06_storetype")
    save_plotly(eda_mod.fig_decomposition(series["Sales"], period=7), "07_decomposition")

    banner("Stationarity (ADF)")
    print(eda_mod.adf_summary(series["Sales"]).to_string(index=False))

    banner(f"Forecasting — horizon={horizon} days  ({label})")
    train, test = models_mod.train_test_split(series, horizon)
    print(f"train: {train.index[0].date()} → {train.index[-1].date()}  ({len(train)})")
    print(f"test : {test.index[0].date()} → {test.index[-1].date()}  ({len(test)})")

    rows = []
    forecasts: dict[str, pd.Series] = {}
    for name, cls in models_mod.MODEL_REGISTRY.items():
        print(f"\n→ {name}")
        try:
            mdl = cls().fit(train)
            fc = mdl.predict(test.index)
            m = models_mod.metrics(test["Sales"], fc)
            m["Model"] = name
            rows.append(m)
            forecasts[name] = fc
            print({k: round(v, 2) for k, v in m.items() if k != "Model"})
        except Exception as e:
            print(f"  [skip] {name}: {e}")

    leaderboard = pd.DataFrame(rows).set_index("Model").sort_values("RMSPE")
    print("\n─── Leaderboard ───")
    print(leaderboard.round(2).to_string())
    leaderboard.to_csv(OUT / f"leaderboard_{label}.csv")

    # Comparison plot
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(train["Sales"].iloc[-90:], label="Train (last 90d)", color="steelblue", alpha=0.6)
    ax.plot(test["Sales"], label="Actual", color="black", linewidth=2)
    for name, fc in forecasts.items():
        ax.plot(fc, label=name, linestyle="--", alpha=0.85)
    ax.set_title(f"Forecast vs Actual — {label}")
    ax.set_ylabel("Sales")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / f"forecasts_{label}.png", dpi=130)
    plt.close()

    print(f"\n✅ Done. Outputs in {OUT.resolve()}")


if __name__ == "__main__":
    args = sys.argv[1:]
    sid = int(args[0]) if args else None
    h = int(args[1]) if len(args) > 1 else 28
    main(store_id=sid, horizon=h)
