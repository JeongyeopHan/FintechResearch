__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from pydantic import BaseModel, Field

from task1_download import download_filings
from task2_analyze import analyze_filings

# Get API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Ensure the API keys are not None
if not openai_api_key:
    st.error("API keys are not set properly. Please check your environment variables.")
    st.stop()

# Streamlit app layout
st.title("SEC Filings Analysis with ChatGPT")

ticker = st.text_input("Enter the company ticker:")
if st.button("Analyze"):
    if ticker:
        try:
            download_dir = download_filings(ticker)

            # Ensure the download directory exists
            if not os.path.exists(download_dir):
                st.error(f"Download directory {download_dir} does not exist.")
                st.stop()

            risk_factor_filings, mdna_filings = get_filings(download_dir)

            if risk_factor_filings and mdna_filings:
                st.write(f"Found {len(risk_factor_filings)} filings with risk factors.")
                st.write(f"Found {len(mdna_filings)} filings with MD&A sections.")

                risk_agent, mdna_agent = analyze_filings(risk_factor_filings, mdna_filings, openai_api_key)

                # Define the questions
                risk_question = f"Identify five major risks identified by {ticker} in its 10-K filings. In English."
                mdna_question = "What are the key strategic initiatives outlined by the company for future growth, and how does the company plan to address any identified risks or challenges in the coming fiscal year?"

                # Get answers from the agents
                risk_response = risk_agent({"input": risk_question})
                mdna_response = mdna_agent({"input": mdna_question})

                st.write("Risk Factors Analysis:")
                st.write(risk_response["output"])

                st.write("MD&A Analysis:")
                st.write(mdna_response["output"])
            else:
                st.write("No filings found for the given ticker.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.write("Please enter a ticker symbol.")

