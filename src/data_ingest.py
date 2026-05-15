"""
data_ingest.py
Downloads UK House Price Index data from ONS and Bank of England base rate.
Uses the ONS bulk download CSV (reliable, no API key needed).
"""

import os
import io
import requests
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ONS UK HPI full dataset - direct CSV download (updated monthly)
HPI_CSV_URL = "https://www.ons.gov.uk/generator?format=csv&uri=/economy/inflationandpriceindices/bulletins/housepriceindex/latest"

# Alternative: HM Land Registry open data (very reliable)
HMLR_URL = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-complete.csv"

# Simpler ONS time series download
ONS_TIMESERIES_URL = "https://www.ons.gov.uk/generator?format=csv&uri=/economy/inflationandpriceindices/timeseries/czeq/hpi"


def download_uk_hpi(save_path=None):
    """
    Download UK average house price time series from ONS.
    Returns a DataFrame with columns: date, avg_price
    """
    if save_path is None:
        save_path = os.path.join(DATA_DIR, "uk_hpi_raw.csv")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    headers = {"User-Agent": "Mozilla/5.0 (compatible; house-price-forecast/1.0)"}

    # Try ONS time series for UK average house price (CZEQ series)
    urls_to_try = [
        "https://www.ons.gov.uk/generator?format=csv&uri=/economy/inflationandpriceindices/timeseries/czeq/hpi",
        "https://www.ons.gov.uk/generator?format=csv&uri=/economy/inflationandpriceindices/timeseries/czeh/hpi",
    ]

    for url in urls_to_try:
        try:
            print(f"Trying: {url}")
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            lines = resp.text.strip().split("\n")
            # Find where the monthly data starts (lines with format YYYY MMM)
            data_rows = []
            for line in lines:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    try:
                        # ONS format: "2024 JAN", "123456.78"
                        date = pd.to_datetime(parts[0].strip(), format="%Y %b")
                        price = float(parts[1].strip())
                        data_rows.append({"date": date, "avg_price": price})
                    except Exception:
                        pass
            if len(data_rows) > 12:
                df = pd.DataFrame(data_rows).sort_values("date").reset_index(drop=True)
                df.to_csv(save_path, index=False)
                print(f"Downloaded {len(df)} months of HPI data -> {save_path}")
                return df
        except Exception as e:
            print(f"  Failed: {e}")

    # Fallback: generate synthetic data for testing if all downloads fail
    print("All ONS downloads failed. Generating synthetic data for demo purposes.")
    return _generate_synthetic_hpi(save_path)


def _generate_synthetic_hpi(save_path):
    """Generate realistic synthetic UK house price data for demo/testing."""
    import numpy as np
    np.random.seed(42)
    dates = pd.date_range(start="2005-01-01", end="2025-12-01", freq="MS")
    # Start at ~180k, grow to ~290k with realistic noise and 2008/2020 dips
    n = len(dates)
    trend = np.linspace(180000, 295000, n)
    seasonal = 3000 * np.sin(2 * np.pi * np.arange(n) / 12)
    noise = np.random.normal(0, 2000, n)
    prices = trend + seasonal + noise
    # 2008 crash
    crash_mask = (dates >= "2008-06-01") & (dates <= "2010-06-01")
    prices[crash_mask] *= 0.92
    # 2020 dip then boom
    dip_mask = (dates >= "2020-03-01") & (dates <= "2020-08-01")
    prices[dip_mask] *= 0.97
    boom_mask = (dates >= "2020-09-01") & (dates <= "2022-12-01")
    prices[boom_mask] *= 1.08
    df = pd.DataFrame({"date": dates, "avg_price": prices.round(0)})
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved synthetic HPI data ({len(df)} rows) -> {save_path}")
    return df


def download_boe_rate(save_path=None):
    """
    Download Bank of England base rate history.
    Returns a DataFrame with columns: date, bank_rate
    """
    if save_path is None:
        save_path = os.path.join(DATA_DIR, "boe_rate_raw.csv")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # BoE published CSV
    url = "https://www.bankofengland.co.uk/boeapps/database/fromshowcolumns.asp?Travel=NIxSUx&FromSeries=1&ToSeries=50&DAT=RNG&FD=1&FM=Jan&FY=1975&TD=31&TM=Dec&TY=2030&VFD=N&html.x=66&html.y=26&C=C05&Filter=N"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        print("Downloading BoE base rate...")
        tables = pd.read_html(url)
        df = tables[0].copy()
        df.columns = ["date", "bank_rate"]
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        df["bank_rate"] = pd.to_numeric(df["bank_rate"], errors="coerce")
        df = df.dropna().sort_values("date").reset_index(drop=True)
        df.to_csv(save_path, index=False)
        print(f"Saved BoE rate data ({len(df)} rows) -> {save_path}")
        return df
    except Exception as e:
        print(f"BoE download failed: {e}. Using synthetic rate data.")
        return _generate_synthetic_boe(save_path)


def _generate_synthetic_boe(save_path):
    """Generate synthetic BoE rate history for demo/testing."""
    dates = pd.date_range(start="2005-01-01", end="2025-12-01", freq="MS")
    # Approximate real BoE rate history
    rates = []
    for d in dates:
        if d < pd.Timestamp("2008-10-01"):
            rates.append(5.0)
        elif d < pd.Timestamp("2009-03-01"):
            rates.append(1.0)
        elif d < pd.Timestamp("2021-12-01"):
            rates.append(0.1)
        elif d < pd.Timestamp("2023-08-01"):
            rates.append(4.0)
        else:
            rates.append(5.25)
    df = pd.DataFrame({"date": dates, "bank_rate": rates})
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved synthetic BoE data ({len(df)} rows) -> {save_path}")
    return df


def load_clean_data(hpi_path=None, boe_path=None):
    """
    Merge HPI and BoE rate data on a monthly date index.
    Returns a DataFrame ready for feature engineering.
    """
    if hpi_path is None:
        hpi_path = os.path.join(DATA_DIR, "uk_hpi_raw.csv")
    if boe_path is None:
        boe_path = os.path.join(DATA_DIR, "boe_rate_raw.csv")

    if not os.path.exists(hpi_path):
        print("HPI data not found. Downloading...")
        download_uk_hpi(hpi_path)
    if not os.path.exists(boe_path):
        print("BoE data not found. Downloading...")
        download_boe_rate(boe_path)

    hpi = pd.read_csv(hpi_path, parse_dates=["date"])
    boe = pd.read_csv(boe_path, parse_dates=["date"])

    boe = boe.set_index("date").resample("MS").last().ffill().reset_index()

    merged = pd.merge_asof(
        hpi.sort_values("date"),
        boe.sort_values("date"),
        on="date",
        direction="backward",
    )
    return merged


if __name__ == "__main__":
    download_uk_hpi()
    download_boe_rate()
    df = load_clean_data()
    print(df.tail(10))
    print(f"\nTotal rows: {len(df)}")
