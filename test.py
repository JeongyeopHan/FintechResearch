import streamlit as st
import os
from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, AgentType, Tool
from pydantic import BaseModel, Field

# Get API keys from environment variables
##openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_key =  "sk-proj-mCXyTBtkG50GO2xQn6EDT3BlbkFJpIRJ2R3tfaea2S7Y6yhY"
# Ensure the API keys are not None
if not openai_api_key:
    st.error("API keys are not set properly. Please check your environment variables.")
    st.stop()

# Initialize OpenAI API
os.environ["OPENAI_API_KEY"] = openai_api_key
llm = OpenAI(temperature=0)

# Streamlit app layout
st.title("SEC Filings Analysis with ChatGPT")

ticker = st.text_input("Enter the company ticker:")
if st.button("Analyze"):
    if ticker:
        filings = []

        # Initialize Downloader
        dl = Downloader("Jeong", "20150613rke3@gmail.com")

        # Download all 10-K filings for the ticker from 1995 to 2023
        dl.get("10-K", ticker, after="1995-01-01", before="2023-12-31")

        # Directory where filings are downloaded
        download_dir = f"./sec-edgar-filings/{ticker}/10-K/"
