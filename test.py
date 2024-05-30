import streamlit as st
import os
from sec_edgar_downloader import Downloader

# Streamlit app layout
st.title("SEC EDGAR 10-K Filings Downloader")

ticker = st.text_input("Enter the company ticker:")
start_year = st.number_input("Enter the start year:", min_value=1995, max_value=2023, value=1995)
end_year = st.number_input("Enter the end year:", min_value=1995, max_value=2023, value=2023)

if st.button("Download 10-K Filings"):
    if ticker and start_year and end_year:
        # Initialize Downloader
        dl = Downloader("Jeong", "20150613rke3@gmail.com")

        # Convert years to strings
        after_date = f"{start_year}-01-01"
        before_date = f"{end_year}-12-31"

        st.write(f"Downloading 10-K filings for ticker: {ticker} from {start_year} to {end_year}")

        # Download all 10-K filings for the ticker from the specified year range
        dl.get("10-K", ticker, after=after_date, before=before_date)

        # Directory where filings are downloaded
        download_dir = f"./sec-edgar-filings/{ticker}/10-K/"
        st.write(f"Download directory: {download_dir}")

        # Check if files are downloaded
        if os.path.exists(download_dir):
            st.write("Files downloaded successfully. Here are the downloaded files:")
            for root, dirs, files in os.walk(download_dir):
                for file in files:
                    st.write(file)
        else:
            st.write("No filings found for the given ticker and year range.")
    else:
        st.write("Please enter a valid ticker and year range.")
