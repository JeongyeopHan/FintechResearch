import os
from sec_edgar_downloader import Downloader

def download_filings(ticker):
    download_dir = os.path.join(".", "sec-edgar-filings", ticker, "10-K")
    
    # Initialize Downloader
    dl = Downloader("Jeongyeop", "20150613rke3@gmail.com", ".")
    
    # Download all 10-K filings for the ticker from 1995 to 2023
    dl.get("10-K", ticker, after="1994-12-31", before="2024-01-01")
    
    return download_dir
