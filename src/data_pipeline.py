import os
import pandas as pd
from fredapi import Fred
import config


# ==============================
# INITIALIZE FRED
# ==============================

FRED_API_KEY = os.getenv("FRED_API_KEY")

if not FRED_API_KEY:
    raise ValueError("Please set FRED_API_KEY in your terminal.")

fred = Fred(api_key=FRED_API_KEY)


# ==============================
# DOWNLOAD DATA (FRED ONLY)
# ==============================

def download_data():
    series_codes = {
        "US_GDP": "GDP",
        "US_Unemployment": "UNRATE",
        "US_CPI": "CPIAUCSL",
        "Fed_Rate": "FEDFUNDS",
        "US_10Y": "DGS10",
        "Credit_Spread": "BAA10Y",
        "Dollar_Index": "DTWEXBGS",
        "Oil_Price": "DCOILWTICO",
        "Industrial_Prod": "INDPRO",
        "SP500": "SP500",
        "VIX": "VIXCLS"
    }

    df = pd.DataFrame()

    for name, code in series_codes.items():
        print(f"Downloading {name}...")
        df[name] = fred.get_series(code)

    return df


# ==============================
# FEATURE ENGINEERING
# ==============================

def engineer_features(df):
    df["SP500_Return"] = df["SP500"].pct_change(fill_method=None)
    df["SP500_Volatility"] = df["SP500_Return"].rolling(6).std()
    return df


# ==============================
# LABEL CRISIS PERIODS
# ==============================

def apply_crisis_labels(df):
    df["Crisis"] = 0
    for start, end in config.CRISIS_PERIODS:
        df.loc[start:end, "Crisis"] = 1
    return df


# ==============================
# MAIN PIPELINE
# ==============================

def main():
    print("Downloading data from FRED...")
    data = download_data()

    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    data = data.loc[config.START_DATE:]

    # Monthly resample
    data = data.resample(config.FREQUENCY).last()

    # Forward fill
    data = data.ffill()

    # Feature engineering
    data = engineer_features(data)

    # Label crises
    data = apply_crisis_labels(data)

    # Drop NA
    data = data.dropna()

    # Save
    os.makedirs("data/final", exist_ok=True)
    data.to_csv("data/final/global_dataset.csv")

    print("\n✅ Dataset created successfully!")
    print("Total rows:", len(data))


if __name__ == "__main__":
    main()