import os
from sec_edgar_downloader import Downloader

def download_filings(ticker):
    # Initialize Downloader
    dl = Downloader("Jeong", "20150613rke3@gmail.com", ".")

    # Download all 10-K filings for the ticker from 2023 onward
    dl.get("10-K", ticker, after="2018-11-01", before="2023-12-31")

    # Directory where filings are downloaded
    download_dir = os.path.join(".", "sec-edgar-filings", ticker, "10-K")

    return download_dir
