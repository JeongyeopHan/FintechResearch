import os
from sec_edgar_downloader import Downloader

# Function to download 10-K filings
def download_10k_filings(ticker, start_year, end_year):
    # Get the user's desktop directory
    desktop_dir = os.path.join(os.path.expanduser("~"), "바탕화면")
    download_dir = os.path.join(desktop_dir, "sec-edgar-filings", ticker, "10-K")

    # Initialize Downloader with the specified download directory
    dl = Downloader(download_dir)

    # Convert years to date strings
    after_date = f"{start_year}-01-01"
    before_date = f"{end_year}-12-31"

    print(f"Downloading 10-K filings for ticker: {ticker} from {start_year} to {end_year}")

    # Download all 10-K filings for the ticker from the specified year range
    dl.get("10-K", ticker, after=after_date, before=before_date)

    print(f"Download directory: {download_dir}")

    # Check if files are downloaded
    if os.path.exists(download_dir):
        print("Files downloaded successfully. Here are the downloaded files:")
        for root, dirs, files in os.walk(download_dir):
            for file in files:
                print(file)
    else:
        print("No filings found for the given ticker and year range.")

# Define the main function
def main():
    # Input parameters
    ticker = input("Enter the company ticker: ")
    start_year = int(input("Enter the start year (e.g., 1995): "))
    end_year = int(input("Enter the end year (e.g., 2023): "))

    # Download the filings
    download_10k_filings(ticker, start_year, end_year)

# Run the main function
if __name__ == "__main__":
    main()
