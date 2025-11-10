# yfinance = allows you to download real stock market data from Yahoo Finance
# pandas = used for handling and manipulating data in table form
import yfinance as yf
import pandas as pd

# Define tickers and date range
tickers = ["AAPL", "MSFT", "TSLA"]
start_date = "2023-01-01"
end_date = "2024-11-30"

# Download data from Yahoo Finance
data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')

rows = []

# Process each ticker separately
for ticker in tickers:
    # Extract the data for the current ticker and reset the index
    df = data[ticker].reset_index()
    # Add a new column so we know which company each row belongs to
    df["Ticker"] = ticker
    # Create a new column 'UpDown':
    # 1 = if tomorrow's Close price is higher than today's
    # 0 = if tomorrow's Close price is lower than today's
    df["UpDown"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df[:-1]  # Drop last row (no next day)
    rows.append(df)

# Combine all tickers into one DataFrame
all_data = pd.concat(rows)
all_data.to_csv("real_stocks.csv", index=False)

print("Saved dataset: real_stocks.csv")
print(all_data.head())
