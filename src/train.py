"""
train.py - Train baseline, SARIMA, and XGBoost models.
"""
import os
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def train_test_split_temporal(df, test_size=0.2):
    n = len(df)
    split = int(n * (1 - test_size))
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def train_baseline(train_df, target_col="avg_price"):
    last_value = train_df[target_col].iloc[-1]
    return {"model": "baseline", "last_value": last_value}


def predict_baseline(model_dict, n_steps):
    return np.full(n_steps, model_dict["last_value"])


def train_sarima(train_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    print("Fitting SARIMA...")
    model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    return model.fit(disp=False)


def train_xgboost(X_train, y_train):
    print("Training XGBoost...")
    model = XGBRegressor(n_estimators=500, learning_rate=0.05,
                         max_depth=4, subsample=0.8,
                         colsample_bytree=0.8, random_state=42)
    model.fit(X_train, y_train, verbose=False)
    return model


def evaluate_model(y_true, y_pred, name="model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true))) * 100
    print(f"{name} | MAE={mae:,.0f}  RMSE={rmse:,.0f}  MAPE={mape:.2f}%")
    return {"model": name, "mae": mae, "rmse": rmse, "mape": mape}


def save_model(model, name):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))


def load_model(name):
    return joblib.load(os.path.join(MODEL_DIR, f"{name}.pkl"))
