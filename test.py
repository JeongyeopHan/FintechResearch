import streamlit as st
import os
from sec_edgar_downloader import Downloader

# Function to download 10-K filings
def get10Kfilings(name, email, path, companyname):
    d = Downloader(name, email, path)
    return d.get("10-K", companyname, after="1994-12-31", before="2024-01-01")

# Streamlit app layout
st.title("Download SEC 10-K Filings")

# User input for the company ticker
ticker = st.text_input("Enter the company ticker (e.g., MSFT):")

# Default values
default_name = "JHON"
default_email = "jhondoe@gmail.com"
default_path = "."

# Display the current working directory
current_directory = os.getcwd()
st.write(f"Current working directory: {current_directory}")

# Download button
if st.button("Download 10-K Filings"):
    if ticker:
        # Ensure the download directory exists
        if not os.path.exists(default_path):
            os.makedirs(default_path)

        # Call the function to download 10-K filings
        try:
            get10Kfilings(default_name, default_email, default_path, ticker)

            # Check if files are downloaded
            download_dir = os.path.join(default_path, f"sec-edgar-filings/{ticker}/10-K/")
            if os.path.exists(download_dir) and os.listdir(download_dir):
                st.success(f"Downloaded 10-K filings for {ticker} to {download_dir}")
                st.write(f"Files downloaded to: {download_dir}")
                st.write("Downloaded files:")
                for file in os.listdir(download_dir):
                    st.write(file)
            else:
                st.warning("No files were downloaded. Please check the ticker symbol and try again.")
        except Exception as e:
            st.error(f"Error downloading filings: {e}")
    else:
        st.warning("Please enter the company ticker.")
