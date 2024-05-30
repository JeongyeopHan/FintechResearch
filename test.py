import os
from sec_edgar_downloader import Downloader

# Function to download 10-K filings
def get10Kfilings(name, email, path, companyname):
  d = Downloader(name, email, path)
  return d.get("10-K", companyname, after="1994-12-31", before="2024-01-01")

get10Kfilings("JHON", "jhondoe@gmail.com", ".", "MSFT")
