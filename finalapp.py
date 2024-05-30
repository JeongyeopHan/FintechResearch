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
        risk_filings = []
        mda_filings = []

        # Initialize Downloader
        dl = Downloader("Jeong", "20150613rke3@gmail.com", ".")

        # Download all 10-K filings for the ticker from 2018 onward
        dl.get("10-K", ticker, after="2018-11-01", before="2023-12-31")

        # Directory where filings are downloaded
        download_dir = os.path.join(".", "sec-edgar-filings", ticker, "10-K")

        # Ensure the download directory exists
        if not os.path.exists(download_dir):
            st.error(f"Download directory {download_dir} does not exist.")
            st.stop()

        st.write(f"Checking directory: {download_dir}")

        # Function to extract sections
        def extract_section(filepath, start_marker, end_marker):
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    soup = BeautifulSoup(file, 'lxml')
                    section_text = ""
                    section_found = False
                    for line in soup.get_text().splitlines():
                        if start_marker in line:
                            section_found = True
                        if section_found:
                            section_text += line + "\n"
                            if end_marker in line:
                                break
                    return section_text
            except FeatureNotFound:
                st.error("lxml parser not found. Please ensure it is installed.")
                st.stop()
            except Exception as e:
                st.error(f"Error processing file {filepath}: {e}")
                return ""

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
                        risk_section_text = extract_section(filepath, "Item 1A.", "Item 1B.")
                        mda_section_text = extract_section(filepath, "Item 7.", "Item 7A.")
                        if risk_section_text:
                            risk_filings.append(Document(page_content=risk_section_text, metadata={"source": filepath}))
                        if mda_section_text:
                            mda_filings.append(Document(page_content=mda_section_text, metadata={"source": filepath}))

        if risk_filings:
            st.write(f"Found {len(risk_filings)} filings with risk factors.")
        else:
            st.write("No risk factors found for the given ticker.")

        if mda_filings:
            st.write(f"Found {len(mda_filings)} filings with MD&A sections.")
        else:
            st.write("No MD&A sections found for the given ticker.")

        if risk_filings or mda_filings:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            risk_texts = text_splitter.split_documents(risk_filings)
            mda_texts = text_splitter.split_documents(mda_filings)

            embeddings = OpenAIEmbeddings()
            
            # Use a temporary directory for Chroma persistence
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    risk_db = Chroma.from_documents(risk_texts, embeddings, persist_directory=os.path.join(temp_dir, "risk"))
                    risk_db.persist()
                    mda_db = Chroma.from_documents(mda_texts, embeddings, persist_directory=os.path.join(temp_dir, "mda"))
                    mda_db.persist()
                except Exception as e:
                    st.error(f"Error initializing Chroma: {e}")
                    st.stop()

            class DocumentInput(BaseModel):
                question: str = Field()

            risk_tool = Tool(
                args_schema=DocumentInput,
                name="Risk Factors Tool",
                description="Useful for answering questions about the risk factors in the document",
                func=RetrievalQA.from_chain_type(llm=llm, retriever=risk_db.as_retriever()),
            )

            mda_tool = Tool(
                args_schema=DocumentInput,
                name="MD&A Tool",
                description="Useful for answering questions about the MD&A sections in the document",
                func=RetrievalQA.from_chain_type(llm=llm, retriever=mda_db.as_retriever()),
            )

            risk_agent = initialize_agent(agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, tools=[risk_tool], llm=llm, verbose=True)
            mda_agent = initialize_agent(agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, tools=[mda_tool], llm=llm, verbose=True)

            # Define the questions
            risk_question = f"Summarize the main risks identified by {ticker} in its 10-K filings. In English."
            mda_insight_question = f"Provide an overview of the financial performance and key insights from the MD&A sections of {ticker}'s 10-K filings."

            # Get answers from the agent
            try:
                risk_response = risk_agent({"input": risk_question})
                st.write("Main Risks Identified:")
                st.write(risk_response["output"])
            except Exception as e:
                st.error(f"Error processing risk factors question: {e}")

            try:
                mda_insight_response = mda_agent({"input": mda_insight_question})
                st.write("MD&A Insights:")
                st.write(mda_insight_response["output"])
            except Exception as e:
                st.error(f"Error processing MD&A insights question: {e}")

        else:
            st.write("No filings found for the given ticker.")
    else:
        st.write("Please enter a ticker symbol.")
