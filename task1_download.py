import os
from sec_edgar_downloader import Downloader

def download_10k_filings(ticker, start_year=1994, end_year=2023):
    dl = Downloader("Jeong", "20150613rke3@gmail.com", '.')
    for year in range(start_year, end_year + 1):
        try:
            dl.get("10-K", ticker, after=f"{year}-01-01", before=f"{year}-12-31")
        except Exception as e:
            print(f"Failed to download 10-K for {ticker} for year {year}: {e}")

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL"]  # Example tickers
    for ticker in tickers:
        download_10k_filings(ticker)
