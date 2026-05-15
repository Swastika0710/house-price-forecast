"""
predict.py - Generate future house price forecasts from trained models.
"""
import numpy as np
import pandas as pd


def forecast_xgboost(model, last_known_row, feature_cols, steps=12):
    predictions = []
    row = last_known_row[feature_cols].values.copy().astype(float)
    feature_list = list(feature_cols)
    for _ in range(steps):
        pred = model.predict(row.reshape(1, -1))[0]
        predictions.append(float(pred))
        lag_cols = sorted(
            [c for c in feature_cols if "_lag_" in c],
            key=lambda x: int(x.split("_lag_")[-1]),
            reverse=True
        )
        for col in lag_cols:
            lag_n = int(col.split("_lag_")[-1])
            idx = feature_list.index(col)
            if lag_n == 1:
                row[idx] = pred
            else:
                prev_col = col.replace(f"_lag_{lag_n}", f"_lag_{lag_n - 1}")
                if prev_col in feature_list:
                    row[idx] = row[feature_list.index(prev_col)]
    return predictions


def forecast_sarima(sarima_result, steps=12):
    fc = sarima_result.get_forecast(steps=steps)
    return fc.predicted_mean, fc.conf_int()


def build_forecast_dates(last_date, steps=12, freq="MS"):
    return pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=steps, freq=freq
    )


def make_forecast_df(dates, predictions, lower=None, upper=None):
    df = pd.DataFrame({"date": dates, "forecast": predictions})
    if lower is not None:
        df["lower_95"] = lower.values
    if upper is not None:
        df["upper_95"] = upper.values
    return df
