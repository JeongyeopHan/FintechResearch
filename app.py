__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from task1_download import download_filings
from task2_analysis import extract_risk_factors, analyze_filings
from langchain.schema import Document
import os

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
        filings = []

        download_dir = download_filings(ticker)

        # Ensure the download directory exists
        if not os.path.exists(download_dir):
            st.error(f"Download directory {download_dir} does not exist.")
            st.stop()

        st.write(f"Checking directory: {download_dir}")

        # Iterate over downloaded filings directories and extract "Risk Factors"
        for root, dirs, files in os.walk(download_dir):
            for subdir in dirs:
                subdir_path = os.path.join(root, subdir)
                st.write(f"Checking subdir: {subdir_path}")
                for file in os.listdir(subdir_path):
                    st.write(f"Found file: {file}")
                    if file == "full-submission.txt":
                        filepath = os.path.join(subdir_path, file)
                        st.write(f"Processing file: {filepath}")
                        section_text = extract_risk_factors(filepath)
                        if section_text:
                            filings.append(Document(page_content=section_text, metadata={"source": filepath}))

        if filings:
            st.write(f"Found {len(filings)} filings with risk factors.")
            
            # Analyze filings
            result = analyze_filings(filings)
            st.write(result)
        else:
            st.write("No filings found for the given ticker.")
    else:
        st.write("Please enter a ticker symbol.")
