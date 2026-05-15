"""
features.py - Feature engineering pipeline for house price forecasting.
"""
import pandas as pd
import numpy as np


def create_lag_features(df, target_col="avg_price", lags=None):
    if lags is None:
        lags = [1, 2, 3, 6, 12]
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    return df


def create_rolling_features(df, target_col="avg_price", windows=None):
    if windows is None:
        windows = [3, 6, 12]
    df = df.copy()
    for w in windows:
        df[f"{target_col}_roll_mean_{w}"] = df[target_col].shift(1).rolling(w).mean()
        df[f"{target_col}_roll_std_{w}"] = df[target_col].shift(1).rolling(w).std()
    return df


def create_date_features(df, date_col="date"):
    df = df.copy()
    df["month"] = df[date_col].dt.month
    df["quarter"] = df[date_col].dt.quarter
    df["year"] = df[date_col].dt.year
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def create_macro_features(df):
    df = df.copy()
    df["bank_rate_change"] = df["bank_rate"].diff()
    df["bank_rate_lag_3"] = df["bank_rate"].shift(3)
    return df


def create_pct_change_features(df, target_col="avg_price"):
    df = df.copy()
    df["mom_change"] = df[target_col].pct_change(1)
    df["yoy_change"] = df[target_col].pct_change(12)
    return df


def build_feature_matrix(df, target_col="avg_price"):
    df = create_lag_features(df, target_col)
    df = create_rolling_features(df, target_col)
    df = create_date_features(df)
    df = create_macro_features(df)
    df = create_pct_change_features(df, target_col)
    return df.dropna().reset_index(drop=True)


def get_feature_columns(df, target_col="avg_price", exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = [target_col, "date"]
    return [c for c in df.columns if c not in exclude_cols]
