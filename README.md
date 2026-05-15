# House Price Forecast

> Predict future UK house prices using XGBoost, SARIMA, and a naive baseline model. Includes a Streamlit dashboard for interactive forecasting.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

This project builds an end-to-end machine learning pipeline to forecast UK average house prices up to 24 months ahead. It uses:

- **XGBoost** with lag, rolling, date, and macroeconomic features
- **SARIMA** for classical time-series forecasting
- **Naive baseline** (last observed value) for benchmarking
- **Streamlit dashboard** for interactive visualisation and forecasting

Data is sourced from the ONS UK House Price Index and Bank of England base rate history.

---

## Project Structure

```
house-price-forecast/
+-- data/                  # Raw and processed CSV data
+-- notebooks/             # Exploratory analysis notebooks
+-- src/
|   +-- data_ingest.py     # Download and merge HPI + BoE rate data
|   +-- features.py        # Feature engineering pipeline
|   +-- train.py           # Train baseline, SARIMA, XGBoost models
|   +-- predict.py         # Generate future forecasts
+-- app/
|   +-- streamlit_app.py   # Interactive Streamlit dashboard
+-- models/                # Saved model files (.pkl)
+-- reports/               # Charts and evaluation outputs
+-- requirements.txt
+-- README.md
```

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/Swastika0710/house-price-forecast.git
cd house-price-forecast
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download data

```bash
python src/data_ingest.py
```

This downloads UK HPI data from the ONS API and Bank of England base rate history, saving them to `data/`.

### 4. Train models

```bash
python src/train.py
```

Trains baseline, XGBoost, and SARIMA models and prints MAE, RMSE, MAPE on the test set.

### 5. Run the dashboard

```bash
streamlit run app/streamlit_app.py
```

Opens an interactive app where you can select forecast horizon (3-24 months), test set size, and view predictions with confidence intervals.

---

## Data Sources

| Dataset | Source | Update frequency |
|---|---|---|
| UK House Price Index | ONS / HM Land Registry | Monthly |
| Bank of England Base Rate | Bank of England | As changed |

---

## Features Used

- Lagged prices (1, 2, 3, 6, 12 months)
- Rolling mean and standard deviation (3, 6, 12 month windows)
- Month and quarter cyclical encodings
- Bank of England base rate and rate change
- Month-on-month and year-on-year price change

---

## Model Comparison

| Model | Description |
|---|---|
| Baseline | Naive last-value repeat |
| SARIMA | Classical time-series with seasonal component |
| XGBoost | Gradient boosting with lag and macro features |

---

## Author

**Swastika Dey** - PhD Researcher, University College London

Built as a portfolio project for ML engineering applications.
