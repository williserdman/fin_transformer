import requests
import time
import pandas as pd
import os
from environment import aa_api_key

API_KEY = aa_api_key
BASE_URL = "https://www.alphavantage.co/query"
output_dir = "alpha_vantage_data"
os.makedirs(output_dir, exist_ok=True)

# fmt: off
russell_2000_sectors = {
    #"Technology": ["AVNW", "TTMI", "ACMR", "CAMT", "PDFS", "SANM", "POWI", "ONTO", "AEIS", "SCSC"],
    #"Healthcare": ["FATE", "INVA", "CORT", "RGEN", "AGIO", "EXEL", "GKOS", "HZNP", "ITCI", "RARE"],
    #"Industrials": ["AIMC", "BDC", "CRS", "DCO", "EAF", "ESAB",  ... replace ...] api key has 25/day limit
    "Industrials": ["HAYN", "KAI", "LECO", "MTZ"],
    "Consumer Discretionary": ["BLMN", "CROX", "LCII", "MHO", "PZZA", "SCVL", "THO", "UFPI", "VIRC", "WSM"],
    "Financials": ["BANF", "CBSH", "EWBC", "FFBC", "FIBK", "GBCI", "HBNC", "IBOC", "LKFN", "RNST"],
    "Energy": ["AROC", "BKR", "CLB", "DNR", "ENSV", "HP", "LBRT", "PUMP", "RES", "VTNR"],
    "Materials": ["CMC", "FUL", "KWR", "IOSP", "NEU", "VMC", "EXP", "MLM", "NGVT", "USLM"],
    "Utilities": ["ALE", "AVA", "BKH", "CWT", "EIX", "IDA", "MGEE", "NWE", "OTTR", "SJW"],
    "Consumer Staples": ["THS", "POST", "CENT", "JBSS", "HAIN", "CALM", "FLO", "VGR", "BGS", "UNFI"],
    "Real Estate": ["ELS", "EXR", "INVH", "NSA", "PSA", "SBRA", "STAG", "UDR", "VICI", "WELL"]
}
# fmt: on


def download_daily_adjusted(ticker):
    params = {
        "function": "TIME_SERIES_WEEKLY",
        "symbol": ticker,
        "outputsize": "full",
        "apikey": API_KEY,
        "datatype": "csv",
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        filename = os.path.join(output_dir, f"{ticker}.csv")
        with open(filename, "w") as f:
            f.write(response.text)
        print(f"Saved data for {ticker}")
    else:
        print(f"Failed to get data for {ticker}, status {response.status_code}")


tickers = [ticker for sector in russell_2000_sectors.values() for ticker in sector]

for i, ticker in enumerate(tickers):
    download_daily_adjusted(ticker)
    if (i + 1) % 5 == 0:  # Alpha Vantage free tier limit: 5 calls per minute
        print("Sleeping for 60 seconds to respect API rate limits...")
        time.sleep(60)
