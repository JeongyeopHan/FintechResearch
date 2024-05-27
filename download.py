import os
from typing import List
from sec_edgar_downloader import Downloader

def download(self, tickers: str, emailaddress: str, download_path: str):


    down = Downloader(ticker, emailaddress, download_path)
    try:
        print(f"Downloading 10-k for {ticker} for the year {year}")
        dl.get("10-k", ticker, after=f"1995-12-31", before=f"2023-01-")
    except Exception as e:
        print(f"An error occured for {ticker} in {year}: {e}")
    print("Donwnload complete.")

if __name__ == "__main__":
    ticker1 = "AAPL"
    email_address = "20150613rke3@gmail.com"
    download_path = "C:/Users/20150/Dropbox (GaTech)/Georgia Tech/Research"
    download(ticker1, email_addres, download_path)
    
    ticker2 = "MSFT"
    email_address = "20150613rke3@gmail.com"
    download_path = "C:/Users/20150/Dropbox (GaTech)/Georgia Tech/Research"
    download(ticker2, email_address, download_path)

    ticker3 = "TSLA"
    email_address = "20150613rke3@gmail.com"
    download_path = "C:/Users/20150/Dropbox (GaTech)/Georgia Tech/Research"
    download(ticker3, email_address, download_path)