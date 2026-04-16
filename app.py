"""Streamlit UI — Rossmann Store Sales Forecasting case study."""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import streamlit as st

from src import data as data_mod
from src import eda as eda_mod
from src import models as models_mod

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Rossmann Sales Forecasting",
    page_icon="📈",
    layout="wide",
)


# ──────────────────────────────────────────────
# Cached data loaders
# ──────────────────────────────────────────────
@st.cache_data(show_spinner="Loading & engineering Rossmann data…")
def load_data():
    return data_mod.load_full()


@st.cache_data(show_spinner=False)
def get_stats(df_hash: int):
    return data_mod.quick_stats(load_data())


@st.cache_data(show_spinner=False)
def get_aggregate(freq: str):
    return data_mod.aggregate_total(load_data(), freq=freq)


@st.cache_data(show_spinner=False)
def get_store_series(store_id: int):
    df = load_data()
    sub = data_mod.daily_series(df, store_id, only_open=True)
    keep = ["Sales", "Promo", "SchoolHoliday", "StateHolidayFlag", "Promo2Active"]
    return sub[keep].asfreq("D").interpolate()


@st.cache_data(show_spinner=False)
def get_network_series():
    df = load_data()
    agg = data_mod.aggregate_total(df, freq="D")[["Sales"]]
    open_only = df[df["Open"] == 1]
    agg["Promo"] = open_only.groupby("Date")["Promo"].mean().reindex(agg.index).fillna(0)
    agg["SchoolHoliday"] = df.groupby("Date")["SchoolHoliday"].mean().reindex(agg.index).fillna(0)
    agg["StateHolidayFlag"] = df.groupby("Date")["StateHolidayFlag"].mean().reindex(agg.index).fillna(0)
    agg["Promo2Active"] = df.groupby("Date")["Promo2Active"].mean().reindex(agg.index).fillna(0)
    return agg.asfreq("D").interpolate()


@st.cache_resource(show_spinner=False)
def fit_and_forecast(scope: str, store_id: int | None, model_name: str, horizon: int):
    series = get_network_series() if scope == "Network" else get_store_series(store_id)
    train, test = models_mod.train_test_split(series, horizon)
    cls = models_mod.MODEL_REGISTRY[model_name]
    mdl = cls().fit(train)
    fc_test = mdl.predict(test.index)
    metrics = models_mod.metrics(test["Sales"], fc_test)
    # Refit on full data for true future forecast
    full_mdl = cls().fit(series)
    future_idx = pd.date_range(series.index[-1] + pd.Timedelta(days=1),
                               periods=horizon, freq="D")
    fc_future = full_mdl.predict(future_idx)
    return {
        "series": series, "train": train, "test": test,
        "fc_test": fc_test, "fc_future": fc_future, "metrics": metrics,
    }


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
st.sidebar.title("⚙️ Controls")
page = st.sidebar.radio(
    "Page",
    ["🏠 Overview", "🔍 EDA", "📉 Decomposition & Stationarity",
     "🤖 Forecasting", "🏆 Model Comparison", "🔮 Future Forecast"],
)

scope = st.sidebar.radio("Scope", ["Network", "Single store"], index=0)
store_id: int | None = None
if scope == "Single store":
    df_for_list = load_data()
    stores = data_mod.list_stores(df_for_list)
    store_id = st.sidebar.selectbox("Store", stores, index=0)

st.sidebar.markdown("---")
st.sidebar.caption("**Case Study 16** · 22BDS0114 · GSVARDHIN")
st.sidebar.caption("Dataset: Rossmann Store Sales (Kaggle)")


# ──────────────────────────────────────────────
# Pages
# ──────────────────────────────────────────────
df = load_data()

if page == "🏠 Overview":
    st.title("📈 Rossmann Store Sales — Forecasting Case Study")
    st.markdown(
        "Multi-model time-series forecasting on **1,115 stores × 942 days** of "
        "real retail sales data from the Rossmann drugstore chain (Kaggle)."
    )
    stats = data_mod.quick_stats(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stores", f"{stats['stores']:,}")
    c2.metric("Days observed", f"{stats['days']:,}")
    c3.metric("Total sales", f"€{stats['total_sales']/1e6:,.1f}M")
    c4.metric("Promo share", f"{stats['promo_share']*100:.1f}%")
    st.caption(f"Date range: {stats['date_min']} → {stats['date_max']}  ·  "
               f"{stats['rows']:,} rows  ·  avg daily sales/store: "
               f"€{stats['avg_daily_sales_per_store']:,.0f}")

    st.subheader("Network-wide daily sales")
    agg = get_aggregate("D")
    st.plotly_chart(eda_mod.fig_total_sales(agg), use_container_width=True)

    st.subheader("Pipeline")
    st.markdown(
        "1. **Load & engineer** — merge `train.csv` + `store.csv`, derive calendar / promo features.\n"
        "2. **EDA** — distributional & seasonal cuts, promo & holiday effects, store-type comparisons.\n"
        "3. **Decomposition + ADF** — separate trend, weekly seasonality, residual; test stationarity.\n"
        "4. **Forecasting** — train Naive (baseline), Holt-Winters, SARIMA, Prophet, XGBoost.\n"
        "5. **Compare** — per-fold walk-forward metrics: MAE, RMSE, MAPE, RMSPE.\n"
        "6. **Future forecast** — refit best model on full data and project ahead."
    )

elif page == "🔍 EDA":
    st.title("🔍 Exploratory Data Analysis")
    agg = get_aggregate("D")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(eda_mod.fig_total_sales(agg), use_container_width=True)
        st.plotly_chart(eda_mod.fig_dow_box(df), use_container_width=True)
        st.plotly_chart(eda_mod.fig_promo_effect(df), use_container_width=True)
    with c2:
        window = st.slider("Rolling window (days)", 7, 90, 30, key="roll_w")
        st.plotly_chart(eda_mod.fig_rolling(agg, window=window), use_container_width=True)
        st.plotly_chart(eda_mod.fig_month_box(df), use_container_width=True)
        st.plotly_chart(eda_mod.fig_storetype(df), use_container_width=True)

    st.subheader("Sales vs Customers (sample)")
    st.plotly_chart(eda_mod.fig_sales_vs_customers(df), use_container_width=True)

elif page == "📉 Decomposition & Stationarity":
    st.title("📉 Decomposition & Stationarity")
    if scope == "Network":
        series = get_network_series()
    else:
        series = get_store_series(store_id)

    period = st.select_slider("Seasonal period", options=[7, 14, 30, 90, 365], value=7)
    st.plotly_chart(eda_mod.fig_decomposition(series["Sales"], period=period),
                    use_container_width=True)

    st.subheader("Augmented Dickey-Fuller Test")
    adf = eda_mod.adf_summary(series["Sales"])
    st.dataframe(adf, use_container_width=True)
    st.caption("p < 0.05 → reject the null of a unit root → series is stationary.")

elif page == "🤖 Forecasting":
    st.title("🤖 Single-Model Forecast")
    series = get_network_series() if scope == "Network" else get_store_series(store_id)

    c1, c2 = st.columns([1, 1])
    model_name = c1.selectbox("Model", list(models_mod.MODEL_REGISTRY.keys()), index=1)
    horizon = c2.slider("Test horizon (days)", 7, 90, 28)

    if st.button("▶ Train & evaluate", type="primary"):
        with st.spinner(f"Training {model_name}…"):
            res = fit_and_forecast(scope, store_id, model_name, horizon)
        m = res["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE", f"{m['MAE']:,.0f}")
        c2.metric("RMSE", f"{m['RMSE']:,.0f}")
        c3.metric("MAPE", f"{m['MAPE']:.2f}%")
        c4.metric("RMSPE", f"{m['RMSPE']:.2f}%")

        import plotly.graph_objects as go
        fig = go.Figure()
        tail = res["train"]["Sales"].iloc[-90:]
        fig.add_trace(go.Scatter(x=tail.index, y=tail.values,
                                 name="Train (last 90d)", line=dict(color="steelblue")))
        fig.add_trace(go.Scatter(x=res["test"].index, y=res["test"]["Sales"],
                                 name="Actual", line=dict(color="black", width=2)))
        fig.add_trace(go.Scatter(x=res["fc_test"].index, y=res["fc_test"].values,
                                 name=f"{model_name} forecast",
                                 line=dict(color="crimson", dash="dash")))
        fig.update_layout(height=460, title=f"{model_name} — backtest",
                          margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Pick a model and horizon, then click **Train & evaluate**.")

elif page == "🏆 Model Comparison":
    st.title("🏆 Model Comparison")
    series = get_network_series() if scope == "Network" else get_store_series(store_id)

    horizon = st.slider("Test horizon (days)", 7, 90, 28, key="cmp_h")
    chosen = st.multiselect(
        "Models to compare",
        list(models_mod.MODEL_REGISTRY.keys()),
        default=["Naive (seasonal-7)", "Holt-Winters", "SARIMA", "XGBoost"],
    )

    if st.button("▶ Run comparison", type="primary"):
        train, test = models_mod.train_test_split(series, horizon)
        rows, fcs = [], {}
        prog = st.progress(0.0, text="Training models…")
        for i, name in enumerate(chosen, 1):
            try:
                mdl = models_mod.MODEL_REGISTRY[name]().fit(train)
                fc = mdl.predict(test.index)
                m = models_mod.metrics(test["Sales"], fc)
                m["Model"] = name
                rows.append(m)
                fcs[name] = fc
            except Exception as e:
                st.warning(f"{name} failed: {e}")
            prog.progress(i / len(chosen), text=f"Done: {name}")
        prog.empty()

        if rows:
            lb = pd.DataFrame(rows).set_index("Model").sort_values("RMSPE")
            st.subheader("Leaderboard (lower = better)")
            st.dataframe(lb.style.format("{:.2f}").background_gradient(
                cmap="RdYlGn_r", subset=["MAE", "RMSE", "MAPE", "RMSPE"]),
                use_container_width=True)

            import plotly.graph_objects as go
            fig = go.Figure()
            tail = train["Sales"].iloc[-90:]
            fig.add_trace(go.Scatter(x=tail.index, y=tail.values,
                                     name="Train", line=dict(color="lightgray")))
            fig.add_trace(go.Scatter(x=test.index, y=test["Sales"],
                                     name="Actual", line=dict(color="black", width=2)))
            for name, fc in fcs.items():
                fig.add_trace(go.Scatter(x=fc.index, y=fc.values,
                                         name=name, line=dict(dash="dash")))
            fig.update_layout(height=500, title="Forecasts vs Actual",
                              margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Pick the models, then click **Run comparison**.")

elif page == "🔮 Future Forecast":
    st.title("🔮 Future Forecast")
    series = get_network_series() if scope == "Network" else get_store_series(store_id)

    c1, c2 = st.columns(2)
    model_name = c1.selectbox("Model", list(models_mod.MODEL_REGISTRY.keys()), index=1)
    horizon = c2.slider("Forecast horizon (days)", 7, 180, 60)

    if st.button("▶ Forecast", type="primary"):
        with st.spinner(f"Fitting {model_name} on full history…"):
            cls = models_mod.MODEL_REGISTRY[model_name]
            mdl = cls().fit(series)
            future_idx = pd.date_range(series.index[-1] + pd.Timedelta(days=1),
                                       periods=horizon, freq="D")
            fc = mdl.predict(future_idx)

        import plotly.graph_objects as go
        fig = go.Figure()
        hist = series["Sales"].iloc[-180:]
        fig.add_trace(go.Scatter(x=hist.index, y=hist.values,
                                 name="History (last 180d)", line=dict(color="steelblue")))
        fig.add_trace(go.Scatter(x=fc.index, y=fc.values,
                                 name="Forecast", line=dict(color="crimson", dash="dash")))
        fig.add_trace(go.Scatter(
            x=list(fc.index) + list(fc.index[::-1]),
            y=list(fc.values * 1.10) + list(fc.values[::-1] * 0.90),
            fill="toself", fillcolor="rgba(220,20,60,0.10)",
            line=dict(color="rgba(0,0,0,0)"), name="±10% band", showlegend=True,
        ))
        fig.update_layout(height=480,
                          title=f"{model_name} — next {horizon} days",
                          margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Forecast values")
        st.dataframe(fc.to_frame("Forecast").round(0), use_container_width=True)
        st.download_button(
            "⬇ Download CSV",
            fc.to_frame("Forecast").to_csv().encode("utf-8"),
            file_name=f"forecast_{model_name}_{horizon}d.csv",
        )
    else:
        st.info("Pick a model and horizon, then click **Forecast**.")
