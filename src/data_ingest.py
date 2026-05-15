"""
data_ingest.py
Downloads UK House Price Index data from ONS and Bank of England base rate data.
"""

import os
import requests
import pandas as pd

HPI_URL = (
    "https://www.ons.gov.uk/generator?format=csv&uri=/economy/inflationandpriceindices"
    "/bulletins/housepriceindex/latest"
)
BOE_RATE_URL = "https://www.bankofengland.co.uk/boeapps/database/fromshowcolumns.asp"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def download_uk_hpi(save_path: str = None) -> pd.DataFrame:
    """
    Download UK House Price Index (HPI) data from the ONS open data API.
    Falls back to a local CSV if the download fails.

    Returns
    -------
    pd.DataFrame with columns: date, avg_price, index
    """
    if save_path is None:
        save_path = os.path.join(DATA_DIR, "uk_hpi_raw.csv")

    # ONS open data — UK average house price
    url = (
        "https://api.beta.ons.gov.uk/v1/datasets/house-prices-local-authority"
        "/editions/time-series/versions/4/observations"
        "?area_type=country&area=K02000001"
    )
    try:
        print("Downloading UK HPI data from ONS...")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        records = [
            {"date": obs["dimensions"]["Time"]["id"], "avg_price": float(obs["observation"])}
            for obs in data.get("observations", [])
        ]
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m")
        df = df.sort_values("date").reset_index(drop=True)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Saved HPI data to {save_path}")
        return df
    except Exception as e:
        print(f"Download failed: {e}")
        if os.path.exists(save_path):
            print("Loading cached data...")
            return pd.read_csv(save_path, parse_dates=["date"])
        raise


def download_boe_rate(save_path: str = None) -> pd.DataFrame:
    """
    Download Bank of England official bank rate history.

    Returns
    -------
    pd.DataFrame with columns: date, bank_rate
    """
    if save_path is None:
        save_path = os.path.join(DATA_DIR, "boe_rate_raw.csv")

    url = (
        "https://www.bankofengland.co.uk/boeapps/database/fromshowcolumns.asp"
        "?Travel=NIxSUx&FromSeries=1&ToSeries=50&DAT=RNG"
        "&FD=1&FM=Jan&FY=1975&TD=31&TM=Dec&TY=2030"
        "&VFD=N&html.x=66&html.y=26&C=C05&Filter=N"
    )
    try:
        print("Downloading BoE rate data...")
        tables = pd.read_html(url)
        df = tables[0].copy()
        df.columns = ["date", "bank_rate"]
        df["date"] = pd.to_datetime(df["date"], dayfirst=True)
        df["bank_rate"] = pd.to_numeric(df["bank_rate"], errors="coerce")
        df = df.dropna().sort_values("date").reset_index(drop=True)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Saved BoE rate data to {save_path}")
        return df
    except Exception as e:
        print(f"Download failed: {e}")
        if os.path.exists(save_path):
            return pd.read_csv(save_path, parse_dates=["date"])
        raise


def load_clean_data(hpi_path: str = None, boe_path: str = None) -> pd.DataFrame:
    """
    Merge HPI and BoE rate data on a monthly date index.

    Returns
    -------
    pd.DataFrame ready for feature engineering.
    """
    if hpi_path is None:
        hpi_path = os.path.join(DATA_DIR, "uk_hpi_raw.csv")
    if boe_path is None:
        boe_path = os.path.join(DATA_DIR, "boe_rate_raw.csv")

    hpi = pd.read_csv(hpi_path, parse_dates=["date"])
    boe = pd.read_csv(boe_path, parse_dates=["date"])

    # Resample BoE rate to monthly (last value per month)
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
    print(df.tail())
