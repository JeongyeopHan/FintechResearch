import os
from sec_edgar_downloader import Downloader

def download_filings(ticker, download_dir="."):
    # Initialize Downloader
    dl = Downloader("Jeong", "20150613rke3@gmail.com", download_dir)

    # Download all 10-K filings for the ticker from 2023 onward
    dl.get("10-K", ticker, after="1994-12-31", before="2024-01-01")

    # Directory where filings are downloaded
    download_dir = os.path.join(download_dir, "sec-edgar-filings", ticker, "10-K")

    return download_dir
