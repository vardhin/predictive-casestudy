"""EDA helpers — return Plotly figures for Streamlit + summary tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller


def fig_total_sales(agg: pd.DataFrame) -> go.Figure:
    fig = px.line(agg.reset_index(), x="Date", y="Sales",
                  title="Network-wide Daily Sales")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def fig_rolling(agg: pd.DataFrame, window: int = 30) -> go.Figure:
    s = agg["Sales"]
    df = pd.DataFrame({
        "Sales": s,
        f"Rolling mean ({window}d)": s.rolling(window).mean(),
        f"Rolling std ({window}d)": s.rolling(window).std(),
    }).reset_index()
    fig = px.line(df, x="Date", y=df.columns[1:].tolist(),
                  title=f"Rolling Statistics ({window}-day)")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def fig_dow_box(df: pd.DataFrame) -> go.Figure:
    sub = df[df["Open"] == 1]
    fig = px.box(sub, x="DayOfWeek", y="Sales", points=False,
                 title="Sales by Day of Week (1=Mon … 7=Sun)")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def fig_month_box(df: pd.DataFrame) -> go.Figure:
    sub = df[df["Open"] == 1]
    fig = px.box(sub, x="Month", y="Sales", points=False,
                 title="Sales by Month")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def fig_promo_effect(df: pd.DataFrame) -> go.Figure:
    sub = df[df["Open"] == 1].copy()
    sub["Promo"] = sub["Promo"].map({0: "No Promo", 1: "Promo"})
    fig = px.box(sub, x="Promo", y="Sales",
                 title="Promo vs Non-Promo Day Sales", points=False)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def fig_storetype(df: pd.DataFrame) -> go.Figure:
    sub = df[df["Open"] == 1]
    grp = sub.groupby(["StoreType", "Assortment"])["Sales"].mean().reset_index()
    fig = px.bar(grp, x="StoreType", y="Sales", color="Assortment",
                 barmode="group", title="Avg Daily Sales by StoreType × Assortment")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def fig_sales_vs_customers(df: pd.DataFrame, sample: int = 5000) -> go.Figure:
    sub = df[df["Open"] == 1].sample(min(sample, len(df)), random_state=0)
    fig = px.scatter(sub, x="Customers", y="Sales", color="Promo",
                     opacity=0.4, title="Sales vs Customers (sampled)")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def fig_decomposition(series: pd.Series, period: int = 7) -> go.Figure:
    series = series.asfreq("D").interpolate()
    result = seasonal_decompose(series, model="additive", period=period)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, name="Observed"))
    fig.add_trace(go.Scatter(x=series.index, y=result.trend, name="Trend"))
    fig.add_trace(go.Scatter(x=series.index, y=result.seasonal, name="Seasonal"))
    fig.add_trace(go.Scatter(x=series.index, y=result.resid, name="Residual"))
    fig.update_layout(title=f"Additive Decomposition (period={period})",
                      height=480, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def adf_summary(series: pd.Series) -> pd.DataFrame:
    rows = []
    for label, s in [
        ("Original", series),
        ("1st diff", series.diff().dropna()),
        ("log + 1st diff", np.log(series.replace(0, np.nan)).dropna().diff().dropna()),
    ]:
        if len(s) < 20:
            continue
        r = adfuller(s, autolag="AIC")
        rows.append({
            "Series": label,
            "ADF stat": round(r[0], 4),
            "p-value": round(r[1], 6),
            "Lags": r[2],
            "Stationary (5%)": "yes" if r[1] < 0.05 else "no",
        })
    return pd.DataFrame(rows)
