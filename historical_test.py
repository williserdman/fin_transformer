import yfinance as yf
import pandas as pd

# The Yahoo tickers for continuous futures usually end in '=F'
tickers = ["ES", "NQ", "YM", "CL", "NG", "GC", "SI", "ZN", "ZB", "6E", "6J"]

for symbol in tickers:
    print(f"Downloading {symbol}...")
    # Yahoo allows ~60 days of 1m data, or 2y of 1h data.
    # For 20 years, you MUST use '1d' (daily) or '1h' (hourly).
    df = yf.download(f"{symbol}=F", start="2000-01-01", interval="1d")

    # Clean for Nautilus
    df.reset_index(inplace=True)
    df.rename(
        columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        },
        inplace=True,
    )

    # Save to Parquet
    df.to_parquet(f"data/futures/{symbol}.parquet")
