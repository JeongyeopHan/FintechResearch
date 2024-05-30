import os
from sec_edgar_downloader import Downloader

def download_filings(ticker):
    dl = Downloader("Jeong", "20150613rke3@gmail.com", ".")
    dl.get("10-K", ticker, after="1994-12-31", before="2024-01-01")
    download_dir = os.path.join(".", "sec-edgar-filings", ticker, "10-K")
    return download_dir
