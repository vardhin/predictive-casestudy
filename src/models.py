"""Forecasting models: Naive, Holt-Winters, SARIMA, Prophet, XGBoost.

All models share a common interface:
    fit(train_df) -> self
    predict(future_index) -> pd.Series indexed by future_index

`train_df` is a pandas DataFrame with a DatetimeIndex and at least a 'Sales'
column. Exogenous models also use 'Promo', 'SchoolHoliday', 'StateHolidayFlag',
and calendar features.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# Feature engineering for ML model
# ──────────────────────────────────────────────
LAGS = [1, 7, 14, 28]
ROLLS = [7, 14, 28]
EXOG_COLS = ["Promo", "SchoolHoliday", "StateHolidayFlag", "DayOfWeek",
             "Month", "WeekOfYear", "Day", "IsWeekend", "Promo2Active"]


def _calendar_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame(index=idx)
    df["DayOfWeek"] = idx.dayofweek + 1
    df["Month"] = idx.month
    df["WeekOfYear"] = idx.isocalendar().week.astype(int).values
    df["Day"] = idx.day
    df["IsWeekend"] = (df["DayOfWeek"] >= 6).astype(int)
    return df


def _build_supervised(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag/rolling features for supervised ML."""
    out = df.copy()
    for lag in LAGS:
        out[f"lag_{lag}"] = out["Sales"].shift(lag)
    for w in ROLLS:
        out[f"roll_mean_{w}"] = out["Sales"].shift(1).rolling(w).mean()
        out[f"roll_std_{w}"] = out["Sales"].shift(1).rolling(w).std()
    return out


# ──────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────
class NaiveSeasonalModel:
    """Predict y_t = y_{t-7} (last week same weekday)."""
    name = "Naive (seasonal-7)"

    def fit(self, train: pd.DataFrame):
        self.history = train["Sales"].copy()
        return self

    def predict(self, future_index: pd.DatetimeIndex) -> pd.Series:
        h = self.history.copy()
        preds = []
        for t in future_index:
            ref = t - pd.Timedelta(days=7)
            preds.append(h.get(ref, h.iloc[-1]))
            h.loc[t] = preds[-1]
        return pd.Series(preds, index=future_index, name="forecast")


class HoltWintersModel:
    name = "Holt-Winters"

    def __init__(self, seasonal_periods: int = 7):
        self.seasonal_periods = seasonal_periods

    def fit(self, train: pd.DataFrame):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        y = train["Sales"].asfreq("D").interpolate()
        # Sales can be 0 → use additive seasonality
        self.model_ = ExponentialSmoothing(
            y, trend="add", seasonal="add",
            seasonal_periods=self.seasonal_periods,
            initialization_method="estimated",
        ).fit(optimized=True)
        self.last_ = y.index[-1]
        return self

    def predict(self, future_index: pd.DatetimeIndex) -> pd.Series:
        steps = len(future_index)
        fc = self.model_.forecast(steps=steps)
        fc.index = future_index
        return fc.clip(lower=0).rename("forecast")


class SARIMAModel:
    name = "SARIMA"

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
        self.order = order
        self.seasonal_order = seasonal_order

    def fit(self, train: pd.DataFrame):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        y = train["Sales"].asfreq("D").interpolate()
        self.model_ = SARIMAX(
            y, order=self.order, seasonal_order=self.seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)
        return self

    def predict(self, future_index: pd.DatetimeIndex) -> pd.Series:
        fc = self.model_.forecast(steps=len(future_index))
        fc.index = future_index
        return fc.clip(lower=0).rename("forecast")


class ProphetModel:
    name = "Prophet"

    def fit(self, train: pd.DataFrame):
        from prophet import Prophet
        df = pd.DataFrame({"ds": train.index, "y": train["Sales"].values})
        regressors = []
        if "Promo" in train.columns:
            df["promo"] = train["Promo"].values
            regressors.append("promo")
        if "SchoolHoliday" in train.columns:
            df["school"] = train["SchoolHoliday"].values
            regressors.append("school")
        self.regressors_ = regressors
        m = Prophet(weekly_seasonality=True, yearly_seasonality=True,
                    daily_seasonality=False)
        for r in regressors:
            m.add_regressor(r)
        m.fit(df)
        self.model_ = m
        # Stash typical regressor values keyed by (dow, month) for future
        self.train_ = train
        return self

    def predict(self, future_index: pd.DatetimeIndex) -> pd.Series:
        future = pd.DataFrame({"ds": future_index})
        # Fill regressors using last-known same-dow median from training
        if self.regressors_:
            tr = self.train_.copy()
            tr["dow"] = tr.index.dayofweek
            for r, col in [("promo", "Promo"), ("school", "SchoolHoliday")]:
                if r not in self.regressors_:
                    continue
                medians = tr.groupby("dow")[col].median()
                future[r] = future["ds"].dt.dayofweek.map(medians).fillna(0).values
        fc = self.model_.predict(future)
        out = pd.Series(fc["yhat"].values, index=future_index, name="forecast")
        return out.clip(lower=0)


class XGBModel:
    name = "XGBoost"

    def __init__(self, n_estimators: int = 400, max_depth: int = 6,
                 learning_rate: float = 0.05):
        self.params = dict(n_estimators=n_estimators, max_depth=max_depth,
                           learning_rate=learning_rate, n_jobs=-1,
                           objective="reg:squarederror", random_state=42)

    def fit(self, train: pd.DataFrame):
        from xgboost import XGBRegressor
        sup = _build_supervised(train).dropna()
        feat_cols = [c for c in sup.columns if c.startswith(("lag_", "roll_"))] + [
            c for c in EXOG_COLS if c in sup.columns
        ]
        X = sup[feat_cols]
        y = sup["Sales"]
        self.feat_cols_ = feat_cols
        self.model_ = XGBRegressor(**self.params).fit(X, y)
        self.history_ = train.copy()
        return self

    def predict(self, future_index: pd.DatetimeIndex) -> pd.Series:
        history = self.history_.copy()
        cal = _calendar_features(future_index)
        # Carry forward last-known exog values via dow medians
        for c in ["Promo", "SchoolHoliday", "StateHolidayFlag", "Promo2Active"]:
            if c in history.columns:
                medians = history.groupby(history.index.dayofweek)[c].median()
                cal[c] = cal["DayOfWeek"].sub(1).map(medians).fillna(0).values

        preds = []
        for t in future_index:
            row = pd.DataFrame(index=[t])
            for col in cal.columns:
                row[col] = cal.loc[t, col]
            for lag in LAGS:
                ref = t - pd.Timedelta(days=lag)
                row[f"lag_{lag}"] = history["Sales"].get(ref, history["Sales"].iloc[-1])
            for w in ROLLS:
                window = history["Sales"].iloc[-w:]
                row[f"roll_mean_{w}"] = window.mean()
                row[f"roll_std_{w}"] = window.std()
            X = row.reindex(columns=self.feat_cols_, fill_value=0)
            yhat = float(self.model_.predict(X)[0])
            preds.append(max(yhat, 0))
            # Append prediction back into history so next-step lags are populated
            new_row = {c: row[c].iloc[0] for c in row.columns}
            new_row["Sales"] = yhat
            history.loc[t] = pd.Series(new_row).reindex(history.columns).fillna(0)
        return pd.Series(preds, index=future_index, name="forecast")


MODEL_REGISTRY = {
    "Naive (seasonal-7)": NaiveSeasonalModel,
    "Holt-Winters": HoltWintersModel,
    "SARIMA": SARIMAModel,
    "Prophet": ProphetModel,
    "XGBoost": XGBModel,
}


# ──────────────────────────────────────────────
# Metrics & evaluation
# ──────────────────────────────────────────────
def metrics(actual: pd.Series, pred: pd.Series) -> dict:
    a, p = actual.align(pred, join="inner")
    a, p = a.values.astype(float), p.values.astype(float)
    mask = a != 0
    mae = float(np.mean(np.abs(a - p)))
    rmse = float(np.sqrt(np.mean((a - p) ** 2)))
    mape = float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100) if mask.any() else np.nan
    # Rossmann competition metric
    rmspe = float(np.sqrt(np.mean(((a[mask] - p[mask]) / a[mask]) ** 2)) * 100) if mask.any() else np.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "RMSPE": rmspe}


def train_test_split(series: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    return series.iloc[:-horizon], series.iloc[-horizon:]


def walk_forward_eval(
    series: pd.DataFrame,
    model_name: str,
    horizon: int = 28,
    n_folds: int = 3,
    **model_kwargs,
) -> pd.DataFrame:
    """Walk-forward CV. Returns per-fold metrics."""
    cls = MODEL_REGISTRY[model_name]
    rows = []
    for fold in range(n_folds):
        cutoff = len(series) - horizon * (n_folds - fold)
        if cutoff <= horizon * 4:
            continue
        train = series.iloc[:cutoff]
        test = series.iloc[cutoff:cutoff + horizon]
        try:
            mdl = cls(**model_kwargs).fit(train)
            fc = mdl.predict(test.index)
            m = metrics(test["Sales"], fc)
        except Exception as e:
            m = {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "RMSPE": np.nan,
                 "error": str(e)}
        m["fold"] = fold
        m["model"] = model_name
        rows.append(m)
    return pd.DataFrame(rows)
