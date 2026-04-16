"""Rossmann data loading, cleaning, and feature engineering."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def _read_train() -> pd.DataFrame:
    df = pd.read_csv(
        ROOT / "train.csv",
        parse_dates=["Date"],
        dtype={"StateHoliday": str},
        low_memory=False,
    )
    return df


def _read_store() -> pd.DataFrame:
    return pd.read_csv(ROOT / "store.csv")


def _engineer(train: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    df = train.merge(store, on="Store", how="left")

    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"] = df["Date"].dt.quarter
    df["IsWeekend"] = (df["DayOfWeek"] >= 6).astype(int)
    df["IsMonthStart"] = df["Date"].dt.is_month_start.astype(int)
    df["IsMonthEnd"] = df["Date"].dt.is_month_end.astype(int)

    df["StateHoliday"] = df["StateHoliday"].fillna("0").astype(str)
    df["StateHolidayFlag"] = (df["StateHoliday"] != "0").astype(int)

    # Competition open duration in months (clipped at 0)
    comp_year = df["CompetitionOpenSinceYear"].fillna(df["Year"])
    comp_month = df["CompetitionOpenSinceMonth"].fillna(df["Month"])
    df["CompetitionOpenMonths"] = (
        12 * (df["Year"] - comp_year) + (df["Month"] - comp_month)
    ).clip(lower=0)

    # Promo2 active flag — store enrolled AND current month is in PromoInterval
    month_to_str = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    df["MonthStr"] = df["Month"].map(month_to_str)
    interval = df["PromoInterval"].fillna("")
    in_interval = np.array(
        [m in iv.split(",") for m, iv in zip(df["MonthStr"], interval)],
        dtype=bool,
    )
    df["Promo2Active"] = ((df["Promo2"] == 1).to_numpy() & in_interval).astype(int)
    df = df.drop(columns=["MonthStr"])

    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(
        df["CompetitionDistance"].median()
    )

    return df


def load_full(use_cache: bool = True) -> pd.DataFrame:
    """Return the merged train+store dataframe with engineered features."""
    cache = CACHE_DIR / "full.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    df = _engineer(_read_train(), _read_store())
    try:
        df.to_parquet(cache, index=False)
    except Exception:
        pass
    return df


def load_store(meta_only: bool = False) -> pd.DataFrame:
    return _read_store()


def list_stores(df: pd.DataFrame | None = None) -> list[int]:
    if df is None:
        df = load_full()
    return sorted(df["Store"].unique().tolist())


def daily_series(df: pd.DataFrame, store_id: int, only_open: bool = True) -> pd.DataFrame:
    """Per-store daily sales series."""
    sub = df[df["Store"] == store_id].copy()
    if only_open:
        sub = sub[sub["Open"] == 1]
    sub = sub.set_index("Date").sort_index()
    return sub


def aggregate_total(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """Network-wide aggregate sales at given frequency (D/W/M)."""
    open_only = df[df["Open"] == 1]
    agg = open_only.groupby("Date").agg(
        Sales=("Sales", "sum"),
        Customers=("Customers", "sum"),
        StoresOpen=("Store", "nunique"),
    )
    if freq != "D":
        agg = agg.resample(freq).sum()
    return agg


def quick_stats(df: pd.DataFrame) -> dict:
    open_df = df[df["Open"] == 1]
    return {
        "rows": len(df),
        "stores": df["Store"].nunique(),
        "date_min": str(df["Date"].min().date()),
        "date_max": str(df["Date"].max().date()),
        "days": df["Date"].nunique(),
        "total_sales": float(open_df["Sales"].sum()),
        "avg_daily_sales_per_store": float(open_df["Sales"].mean()),
        "promo_share": float((open_df["Promo"] == 1).mean()),
    }
