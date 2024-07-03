import os
from sec_edgar_downloader import Downloader

def download_filings(ticker):
    # Initialize Downloader
    dl = Downloader("Jeong", "20150613rke3@gmail.com", ".")
    
    # Download all 10-K filings for the ticker from 1994 onward
    dl.get("10-K", ticker, after="1994-12-31", before="2024-01-01")

    # Directory where filings are downloaded
    download_dir = os.path.join(".", "sec-edgar-filings", ticker, "10-K")
    
    # Ensure the download directory exists
    if not os.path.exists(download_dir):
        raise FileNotFoundError(f"Download directory {download_dir} does not exist.")
    
    return download_dir
