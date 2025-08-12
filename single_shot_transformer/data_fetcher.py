from environment import aa_api_key

import requests
import os
from time import sleep

# Settings
symbol = "IBM"
interval = "1min"
outputsize = "full"
datatype = "csv"
year_start = 2020
year_end_exclusive = 2025
save_dir = "alpha_vantage_weekly_data"

# Create directory to store data
os.makedirs(save_dir, exist_ok=True)

for year in range(year_start, year_end_exclusive):
    # Fetch data for each month
    for month in range(1, 13):
        month_str = f"{year}-{month:02d}"
        print(f"Fetching data for {symbol} - {month_str}...")

        """ url = (
            "https://www.alphavantage.co/query"
            f"?function=TIME_SERIES_INTRADAY"
            f"&symbol={symbol}"
            f"&interval={interval}"
            f"&month={month_str}"
            f"&outputsize={outputsize}"
            f"&datatype={datatype}"
            f"&apikey={aa_api_key}"
        ) """

        url = (
            "https://www.alphavantage.co/query"
            f"?function=TIME_SERIES_WEEKLY"
            f"&symbol={symbol}"
            f"&apikey={aa_api_key}"
        )

        response = requests.get(url)

        if response.status_code == 200:
            file_path = os.path.join(save_dir, f"{symbol}_{month_str}.csv")
            with open(file_path, "w") as f:
                f.write(response.text)
            print(f"Saved to {file_path}")
        else:
            print(
                f"Failed to fetch data for {month_str}: {response.status_code} {response.text}"
            )

        sleep(12)

print("All done.")
