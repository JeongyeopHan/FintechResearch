import os
from typing import List
from sec_edgar_downloader import Downloader

def download_10k_filings(ticker, emailaddress, download_path):
    dl = Downloader(ticker, email_address, download_path)
    try:
        print(f"Downloading 10-K for {ticker}")
        dl.get("10-K", ticker, after="1995-12-31", before="2023-01-01")
    except Exception as e:
        print(f"An error occurred for {ticker}: {e}")
    print("Download complete.")

if __name__ == "__main__":
    ticker1 = "AAPL"
    email_address = "20150613rke3@gmail.com"
    download_path1 = "C:/Users/20150/Dropbox (GaTech)/Georgia Tech/FintechResearch/fillings"
    download_10k_filings(ticker1, email_address, download_path1)
    
    ticker2 = "MSFT"
    email_address = "20150613rke3@gmail.com"
    download_path2 = "C:/Users/20150/Dropbox (GaTech)/Georgia Tech/FintechResearch/fillings"
    download_10k_filings(ticker2, email_address, download_path2)

    ticker3 = "TSLA"
    email_address = "20150613rke3@gmail.com"
    download_path3 = "C:/Users/20150/Dropbox (GaTech)/Georgia Tech/FintechResearch/fillings"
    download_10k_filings(ticker3, email_address, download_path3)