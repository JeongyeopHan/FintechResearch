__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import tempfile
from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup, FeatureNotFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, AgentType, Tool
from pydantic import BaseModel, Field
from langchain.schema import Document

# Get API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Ensure the API keys are not None
if not openai_api_key:
    st.error("API keys are not set properly. Please check your environment variables.")
    st.stop()

# Initialize OpenAI API
os.environ["OPENAI_API_KEY"] = openai_api_key
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

# Streamlit app layout
st.title("SEC Filings Analysis with ChatGPT")

ticker = st.text_input("Enter the company ticker:")
if st.button("Analyze"):
    if ticker:
        filings = []

        # Initialize Downloader
        dl = Downloader("Jeong", "20150613rke3@gmail.com", ".")

        # Download all 10-K filings for the ticker from 2023 onward
        dl.get("10-K", ticker, after="1994-12-31", before="2024-01-01")

        # Directory where filings are downloaded
        download_dir = os.path.join(".", "sec-edgar-filings", ticker, "10-K")

        # Ensure the download directory exists
        if not os.path.exists(download_dir):
            st.error(f"Download directory {download_dir} does not exist.")
            st.stop()

        st.write(f"Checking directory: {download_dir}")

        # Function to extract risk factors section
        def extract_risk_factors(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    soup = BeautifulSoup(file, 'lxml')
                    risk_factors_section = ""
                    risk_factors = False
                    for line in soup.get_text().splitlines():
                        if "Item 1A." in line:
                            risk_factors = True
                        if risk_factors:
                            risk_factors_section += line + "\n"
                            if "Item 1B." in line:
                                break
                    return risk_factors_section
            except FeatureNotFound:
                st.error("lxml parser not found. Please ensure it is installed.")
                st.stop()
            except Exception as e:
                st.error(f"Error processing file {filepath}: {e}")
                return ""

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
            
            # Process filings with Langchain
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_texts = text_splitter.split_documents(filings)

            embeddings = OpenAIEmbeddings()
            
            # Use a temporary directory for Chroma persistence
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    db = Chroma.from_documents(split_texts, embeddings, persist_directory=temp_dir)
                    db.persist()
                except Exception as e:
                    st.error(f"Error initializing Chroma: {e}")
                    st.stop()

            class DocumentInput(BaseModel):
                question: str = Field()

            tools = [
                Tool(
                    args_schema=DocumentInput,
                    name="document_tool",  # Ensure the name matches the required pattern
                    description="Useful for answering questions about the document",
                    func=RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever()),
                )
            ]

            agent = initialize_agent(agent=AgentType.OPENAI_FUNCTIONS, tools=tools, llm=llm, verbose=True)

            # Define the question
            question = f"Identify five major risks identified by {ticker} in its 10-K filings. In English."
            
            # Get answer from the agent
            response = agent({"input": question})
            st.write(response["output"])
        else:
            st.write("No filings found for the given ticker.")
    else:
        st.write("Please enter a ticker symbol.")
