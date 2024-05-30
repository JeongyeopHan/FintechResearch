__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import tempfile
from langchain.schema import Document
from download_filings import download_10k_filings
from analyze_filings import extract_risk_factors, analyze_filings

# Get API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Ensure the API keys are not None
if not openai_api_key:
    st.error("API keys are not set properly. Please check your environment variables.")
    st.stop()

# Initialize OpenAI API
os.environ["OPENAI_API_KEY"] = openai_api_key

# Streamlit app layout
st.title("SEC Filings Analysis with ChatGPT")

ticker = st.text_input("Enter the company ticker:")
if st.button("Analyze"):
    if ticker:
        download_10k_filings(ticker)

        risk_factor_filings = []
        download_dir = os.path.join(".", "sec-edgar-filings", ticker, "10-K")

        # Ensure the download directory exists
        if not os.path.exists(download_dir):
            st.error(f"Download directory {download_dir} does not exist.")
            st.stop()

        st.write(f"Checking directory: {download_dir}")

        # Iterate over downloaded filings directories and extract sections
        for root, dirs, files in os.walk(download_dir):
            for subdir in dirs:
                subdir_path = os.path.join(root, subdir)
                st.write(f"Checking subdir: {subdir_path}")
                for file in os.listdir(subdir_path):
                    st.write(f"Found file: {file}")
                    if file == "full-submission.txt":
                        filepath = os.path.join(subdir_path, file)
                        st.write(f"Processing file: {filepath}")

                        # Extract Risk Factors
                        risk_factors_text = extract_risk_factors(filepath)
                        if risk_factors_text:
                            risk_factor_filings.append(Document(page_content=risk_factors_text, metadata={"source": filepath}))

        if risk_factor_filings:
            st.write(f"Found {len(risk_factor_filings)} filings with risk factors.")
            result = analyze_filings(risk_factor_filings)
            st.write("Risk Factors Analysis:")
            st.write(result)
        else:
            st.write("No filings found for the given ticker.")
    else:
        st.write("Please enter a ticker symbol.")
