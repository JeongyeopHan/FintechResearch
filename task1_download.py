__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
from sec_edgar_downloader import Downloader

def download_10k_filings(ticker, start_year=1995, end_year=2023):
    dl = Downloader("Jeong", "20150613rke3@gmail.com", ".")
    for year in range(start_year, end_year + 1):
        try:
            dl.get("10-K", ticker, after="2018-11-01", before="2023-12-31")
        except Exception as e:
            print(f"Failed to download 10-K for {ticker} for year {year}: {e}")

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL"]  # Example tickers
    for ticker in tickers:
        download_10k_filings(ticker)
