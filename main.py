"""
Case Study 16: Retail Sales Time Series Analysis
22BDS0114 – GSVARDHIN

Dataset : AirPassengers (Box & Jenkins, 1976)
          Monthly totals of international airline passengers, 1949–1960.
          Built-in dataset from the statsmodels library — no download needed.

Tasks:
  1. Perform trend and seasonality analysis
  2. Build forecasting models (ARIMA & Holt-Winters)
  3. Discuss limitations of time series prediction
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Save PNGs without GUI
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["figure.dpi"] = 120


# ──────────────────────────────────────────────
# 1. LOAD THE AIRPASSENGERS DATASET
# ──────────────────────────────────────────────
def load_data():
    """
    Load the AirPassengers dataset (Box & Jenkins, 1976).
    144 monthly observations of international airline passenger totals,
    January 1949 – December 1960.  Sourced from statsmodels built-in data.
    """
    raw = sm.datasets.get_rdataset("AirPassengers", "datasets").data

    # The 'time' column is fractional years (1949.000, 1949.083, …).
    # Convert to a proper DatetimeIndex.
    start = pd.Timestamp("1949-01-01")
    dates = pd.date_range(start=start, periods=len(raw), freq="MS")

    df = pd.DataFrame({"Passengers": raw["value"].values}, index=dates)
    df.index.name = "Date"
    return df


# ──────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ──────────────────────────────────────────────
def eda(df):
    """Perform exploratory data analysis on the AirPassengers data."""
    print("=" * 60)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    series = df["Passengers"]

    print(f"\n📊 Dataset Shape : {df.shape}")
    print(f"   Period         : {df.index[0].date()} to {df.index[-1].date()}")
    print(f"   Frequency      : Monthly")
    print("\n📋 First 5 rows:")
    print(df.head())
    print("\n📈 Descriptive Statistics:")
    print(df.describe().round(2))
    print(f"\n🔍 Missing Values : {series.isnull().sum()}")

    # ── Plot 1: Overall trend ──
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(series, color="steelblue", linewidth=1.5)
    ax.set_title("Monthly International Airline Passengers (1949 – 1960)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Passengers (thousands)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("01_eda_passenger_trend.png", bbox_inches="tight")
    plt.close()

    # ── Plot 2: Box-plot by month ──
    df_tmp = df.copy()
    df_tmp["Month"] = df_tmp.index.month
    df_tmp["Year"] = df_tmp.index.year

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(x="Month", y="Passengers", data=df_tmp, palette="coolwarm", ax=ax)
    ax.set_title("Passenger Distribution by Month", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Passengers (thousands)")
    plt.tight_layout()
    plt.savefig("02_eda_monthly_boxplot.png", bbox_inches="tight")
    plt.close()

    # ── Plot 3: Year-over-Year comparison ──
    fig, ax = plt.subplots(figsize=(12, 5))
    for year in sorted(df_tmp["Year"].unique()):
        yearly = df_tmp[df_tmp["Year"] == year]
        ax.plot(yearly["Month"], yearly["Passengers"], marker="o", markersize=4, label=str(year))
    ax.set_title("Year-over-Year Passenger Comparison", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Passengers (thousands)")
    ax.set_xticks(range(1, 13))
    ax.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.savefig("03_eda_yoy_comparison.png", bbox_inches="tight")
    plt.close()

    # ── Plot 4: Rolling statistics ──
    rolling_mean = series.rolling(window=12).mean()
    rolling_std = series.rolling(window=12).std()

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(series, label="Original", color="steelblue", alpha=0.7)
    ax.plot(rolling_mean, label="12-month Rolling Mean", color="red", linewidth=2)
    ax.plot(rolling_std, label="12-month Rolling Std", color="orange", linewidth=2)
    ax.set_title("Rolling Mean & Standard Deviation", fontsize=13, fontweight="bold")
    ax.set_ylabel("Passengers (thousands)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("04_eda_rolling_stats.png", bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────
# 3. TIME SERIES DECOMPOSITION
# ──────────────────────────────────────────────
def decompose(series):
    """Decompose the series into trend, seasonal, and residual (multiplicative)."""
    print("\n" + "=" * 60)
    print("  TIME SERIES DECOMPOSITION")
    print("=" * 60)

    # Multiplicative model suits this data (seasonal amplitude grows with level)
    result = seasonal_decompose(series, model="multiplicative", period=12)

    fig, axes = plt.subplots(4, 1, figsize=(13, 11))
    fig.suptitle("Multiplicative Decomposition – AirPassengers", fontsize=14, fontweight="bold")

    components = [
        (series,          "Observed",  "steelblue"),
        (result.trend,    "Trend",     "darkorange"),
        (result.seasonal, "Seasonal",  "seagreen"),
        (result.resid,    "Residual",  "crimson"),
    ]
    for ax, (data, title, color) in zip(axes, components):
        ax.plot(data, color=color, linewidth=1.2)
        ax.set_title(title)
        ax.set_ylabel(title)

    plt.tight_layout()
    plt.savefig("05_decomposition.png", bbox_inches="tight")
    plt.close()

    print("\n✅ Multiplicative decomposition complete.")
    print(f"   Trend range      : {result.trend.dropna().min():.1f} – {result.trend.dropna().max():.1f}")
    print(f"   Seasonal range   : {result.seasonal.min():.3f} – {result.seasonal.max():.3f}")
    print(f"   Residual std     : {result.resid.dropna().std():.4f}")
    return result


# ──────────────────────────────────────────────
# 4. STATIONARITY TEST (ADF)
# ──────────────────────────────────────────────
def stationarity_test(series):
    """Run the Augmented Dickey-Fuller test on the original and differenced series."""
    print("\n" + "=" * 60)
    print("  STATIONARITY TEST (Augmented Dickey-Fuller)")
    print("=" * 60)

    def _run_adf(s, label):
        r = adfuller(s.dropna(), autolag="AIC")
        print(f"\n   {label}")
        print(f"   ADF Statistic  : {r[0]:.4f}")
        print(f"   p-value        : {r[1]:.6f}")
        print(f"   Lags Used      : {r[2]}")
        for k, v in r[4].items():
            print(f"      Critical {k}: {v:.4f}")
        stationary = r[1] < 0.05
        tag = "✅ Stationary" if stationary else "❌ Non-stationary"
        print(f"   Result         : {tag}")
        return stationary

    _run_adf(series, "Original series")

    diff = series.diff().dropna()
    _run_adf(diff, "After 1st differencing")

    log_diff = np.log(series).diff().dropna()
    _run_adf(log_diff, "After log + 1st differencing")

    return diff


# ──────────────────────────────────────────────
# 5. ACF & PACF PLOTS
# ──────────────────────────────────────────────
def plot_acf_pacf(series):
    """Plot ACF and PACF on log-differenced series for parameter selection."""
    print("\n" + "=" * 60)
    print("  ACF & PACF PLOTS")
    print("=" * 60)

    diff = np.log(series).diff().dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ACF & PACF of Log-Differenced Passengers", fontsize=13, fontweight="bold")

    plot_acf(diff, lags=30, ax=axes[0])
    axes[0].set_title("ACF")

    plot_pacf(diff, lags=30, ax=axes[1], method="ywm")
    axes[1].set_title("PACF")

    plt.tight_layout()
    plt.savefig("06_acf_pacf.png", bbox_inches="tight")
    plt.close()

    print("   Saved → 06_acf_pacf.png")


# ──────────────────────────────────────────────
# 6. ARIMA FORECAST
# ──────────────────────────────────────────────
def arima_forecast(series):
    """Fit ARIMA on the data (last 24 months held out for testing)."""
    print("\n" + "=" * 60)
    print("  ARIMA FORECASTING MODEL")
    print("=" * 60)

    train, test = series[:-24], series[-24:]
    print(f"\n   Train : {train.index[0].date()} → {train.index[-1].date()}  ({len(train)} months)")
    print(f"   Test  : {test.index[0].date()} → {test.index[-1].date()}  ({len(test)} months)")

    model = ARIMA(train, order=(2, 1, 2))
    fitted = model.fit()
    print(f"\n   ARIMA(2,1,2)  AIC={fitted.aic:.1f}  BIC={fitted.bic:.1f}")

    fc = fitted.forecast(steps=len(test))

    mae  = mean_absolute_error(test, fc)
    rmse = np.sqrt(mean_squared_error(test, fc))
    mape = np.mean(np.abs((test - fc) / test)) * 100

    print(f"\n   📊 Performance on test set:")
    print(f"   MAE  = {mae:.2f}")
    print(f"   RMSE = {rmse:.2f}")
    print(f"   MAPE = {mape:.2f}%")

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(train, label="Train", color="steelblue")
    ax.plot(test, label="Actual (Test)", color="seagreen", marker="o", markersize=4)
    ax.plot(fc, label="ARIMA Forecast", color="crimson", linestyle="--", marker="x", markersize=5)
    ax.set_title("ARIMA(2,1,2) – Forecast vs Actual", fontsize=13, fontweight="bold")
    ax.set_ylabel("Passengers (thousands)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("07_arima_forecast.png", bbox_inches="tight")
    plt.close()

    return {"Model": "ARIMA(2,1,2)", "MAE": mae, "RMSE": rmse, "MAPE": mape}


# ──────────────────────────────────────────────
# 7. HOLT-WINTERS EXPONENTIAL SMOOTHING
# ──────────────────────────────────────────────
def holtwinters_forecast(series):
    """Fit Holt-Winters triple exponential smoothing (multiplicative)."""
    print("\n" + "=" * 60)
    print("  HOLT-WINTERS EXPONENTIAL SMOOTHING")
    print("=" * 60)

    train, test = series[:-24], series[-24:]

    model = ExponentialSmoothing(
        train, trend="mul", seasonal="mul", seasonal_periods=12
    ).fit(optimized=True)

    fc = model.forecast(steps=len(test))

    mae  = mean_absolute_error(test, fc)
    rmse = np.sqrt(mean_squared_error(test, fc))
    mape = np.mean(np.abs((test - fc) / test)) * 100

    print(f"\n   📊 Performance on test set:")
    print(f"   MAE  = {mae:.2f}")
    print(f"   RMSE = {rmse:.2f}")
    print(f"   MAPE = {mape:.2f}%")

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(train, label="Train", color="steelblue")
    ax.plot(test, label="Actual (Test)", color="seagreen", marker="o", markersize=4)
    ax.plot(fc, label="Holt-Winters Forecast", color="darkorange", linestyle="--", marker="s", markersize=4)
    ax.set_title("Holt-Winters (Multiplicative) – Forecast vs Actual", fontsize=13, fontweight="bold")
    ax.set_ylabel("Passengers (thousands)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("08_holtwinters_forecast.png", bbox_inches="tight")
    plt.close()

    return {"Model": "Holt-Winters (mul)", "MAE": mae, "RMSE": rmse, "MAPE": mape}


# ──────────────────────────────────────────────
# 8. MODEL COMPARISON
# ──────────────────────────────────────────────
def compare_models(results):
    """Side-by-side comparison of ARIMA vs Holt-Winters."""
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON")
    print("=" * 60)

    comp = pd.DataFrame(results).set_index("Model")
    print("\n", comp.to_string())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Model Comparison – ARIMA vs Holt-Winters", fontsize=14, fontweight="bold")

    colors = ["steelblue", "darkorange"]
    for i, metric in enumerate(["MAE", "RMSE", "MAPE"]):
        bars = axes[i].bar(comp.index, comp[metric], color=colors)
        axes[i].set_title(metric)
        axes[i].set_ylabel(metric + (" (%)" if metric == "MAPE" else ""))
        for bar, val in zip(bars, comp[metric]):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f"{val:.1f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig("09_model_comparison.png", bbox_inches="tight")
    plt.close()

    best = comp["MAPE"].idxmin()
    print(f"\n   🏆 Best Model : {best}  (lowest MAPE = {comp.loc[best, 'MAPE']:.2f}%)")


# ──────────────────────────────────────────────
# 9. FUTURE FORECAST
# ──────────────────────────────────────────────
def future_forecast(series, months=24):
    """Forecast into the future using Holt-Winters."""
    print("\n" + "=" * 60)
    print(f"  FUTURE FORECAST  (next {months} months)")
    print("=" * 60)

    model = ExponentialSmoothing(
        series, trend="mul", seasonal="mul", seasonal_periods=12
    ).fit(optimized=True)

    future = model.forecast(steps=months)
    future.index = pd.date_range(
        start=series.index[-1] + pd.DateOffset(months=1), periods=months, freq="MS"
    )

    print("\n   Predicted passengers:")
    for dt, val in future.items():
        print(f"   {dt.strftime('%b %Y')} : {val:,.0f} thousand")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(series, label="Historical", color="steelblue", linewidth=1.5)
    ax.plot(future, label="Forecast", color="crimson", linestyle="--", marker="o", markersize=4)
    ax.axvline(x=series.index[-1], color="gray", linestyle=":", alpha=0.6, label="Forecast start")
    ax.fill_between(future.index, future * 0.90, future * 1.10, alpha=0.12, color="crimson",
                    label="±10% band")
    ax.set_title(f"Passenger Forecast – Next {months} Months (Holt-Winters)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Passengers (thousands)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("10_future_forecast.png", bbox_inches="tight")
    plt.close()



# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print("╔" + "═" * 58 + "╗")
    print("║  Case Study 16 : Retail Sales Time Series Analysis      ║")
    print("║  Dataset        : AirPassengers (Box & Jenkins, 1976)   ║")
    print("║  22BDS0114 – GSVARDHIN                                  ║")
    print("╚" + "═" * 58 + "╝\n")

    # 1 – Load data
    df = load_data()
    series = df["Passengers"]

    # 2 – EDA
    eda(df)

    # 3 – Decomposition
    decompose(series)

    # 4 – Stationarity
    stationarity_test(series)

    # 5 – ACF / PACF
    plot_acf_pacf(series)

    # 6 – ARIMA
    res_arima = arima_forecast(series)

    # 7 – Holt-Winters
    res_hw = holtwinters_forecast(series)

    # 8 – Compare
    compare_models([res_arima, res_hw])

    # 9 – Future forecast
    future_forecast(series, months=24)

    print("✅ Done — all plots saved as PNG files in the project directory.\n")


if __name__ == "__main__":
    main()
    #3rd module 2 questions
    #4th module 2 questions
    #5th module 1 question (till non linear classification models (only classification models))