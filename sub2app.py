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
import plotly.express as px
import pandas as pd

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
        risk_factor_filings = []
        financial_statements = []

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

        # Function to extract sections
        def extract_section(filepath, start_marker, end_marker):
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    soup = BeautifulSoup(file, 'lxml')
                    section_text = ""
                    in_section = False
                    for line in soup.get_text().splitlines():
                        if start_marker in line:
                            in_section = True
                        if in_section:
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

                        # Extract Risk Factors
                        risk_factors_text = extract_section(filepath, "Item 1A.", "Item 1B.")
                        if risk_factors_text:
                            risk_factor_filings.append(Document(page_content=risk_factors_text, metadata={"source": filepath}))

                        # Extract Financial Statements
                        financial_text = extract_section(filepath, "Item 8.", "Item 9.")
                        if financial_text:
                            financial_statements.append(Document(page_content=financial_text, metadata={"source": filepath}))

        if risk_factor_filings and financial_statements:
            st.write(f"Found {len(risk_factor_filings)} filings with risk factors.")
            st.write(f"Found {len(financial_statements)} filings with financial statements.")

            # Process risk factors with Langchain
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            risk_split_texts = text_splitter.split_documents(risk_factor_filings)

            embeddings = OpenAIEmbeddings()
            
            # Use a temporary directory for Chroma persistence
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    risk_db = Chroma.from_documents(risk_split_texts, embeddings, persist_directory=temp_dir)
                    risk_db.persist()
                except Exception as e:
                    st.error(f"Error initializing Chroma for risk factors: {e}")
                    st.stop()

            # Process financial statements with Langchain
            financial_split_texts = text_splitter.split_documents(financial_statements)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    financial_db = Chroma.from_documents(financial_split_texts, embeddings, persist_directory[temp_dir])
                    financial_db.persist()
                except Exception as e:
                    st.error(f"Error initializing Chroma for financial statements: {e}")
                    st.stop()

            class DocumentInput(BaseModel):
                question: str = Field()

            risk_tools = [
                Tool(
                    args_schema=DocumentInput,
                    name="risk_document_tool",  # Ensure the name matches the required pattern
                    description="Useful for answering questions about risk factors in the document",
                    func=RetrievalQA.from_chain_type(llm=llm, retriever=risk_db.as_retriever()),
                )
            ]

            financial_tools = [
                Tool(
                    args_schema=DocumentInput,
                    name="financial_document_tool",  # Ensure the name matches the required pattern
                    description="Useful for answering questions about financial statements in the document",
                    func=RetrievalQA.from_chain_type(llm=llm, retriever=financial_db.as_retriever()),
                )
            ]

            risk_agent = initialize_agent(agent=AgentType.OPENAI_FUNCTIONS, tools=risk_tools, llm=llm, verbose=True)
            financial_agent = initialize_agent(agent=AgentType.OPENAI_FUNCTIONS, tools=financial_tools, llm=llm, verbose=True)

            # Define the questions
            risk_question = f"Identify and rank the five major risks identified by {ticker} in its 10-K filings."
            financial_question = f"Summarize the financial statements of {ticker} over the recent years."

            # Get answers from the agents
            risk_response = risk_agent({"input": risk_question})
            financial_response = financial_agent({"input": financial_question})

            st.write("Risk Factors Analysis:")
            st.write(risk_response["output"])

            st.write("Financial Statements Summary:")
            st.write(financial_response["output"])

            # Example risk factors analysis output (replace with actual response)
            risk_factors_ranked = [
                {"Risk Factor": "Data Security Standards", "Importance": 5},
                {"Risk Factor": "Fluctuating Net Sales", "Importance": 4},
                {"Risk Factor": "Gross Margins Pressure", "Importance": 3},
                {"Risk Factor": "New Business Strategies", "Importance": 2},
                {"Risk Factor": "IT System Failures", "Importance": 1}
            ]

            # Example financial summary output (replace with actual response)
            financial_summary = [
                {"Year": 2020, "Revenue": 274515, "Net Income": 57411, "Total Assets": 323888},
                {"Year": 2021, "Revenue": 365817, "Net Income": 94680, "Total Assets": 351002},
                {"Year": 2022, "Revenue": 394328, "Net Income": 99983, "Total Assets": 351002}
            ]

            # Convert to DataFrame for visualization
            risk_factors_df = pd.DataFrame(risk_factors_ranked)
            financial_summary_df = pd.DataFrame(financial_summary)

            # Create bar charts for risk factors
            fig_risk = px.bar(risk_factors_df, x='Risk Factor', y='Importance', title='Major Risk Factors Ranked by Importance')

            # Create line charts for financial summary
            fig_financial = px.line(financial_summary_df, x='Year', y=['Revenue', 'Net Income', 'Total Assets'],
                                    title='Financial Summary Over Recent Years')

            st.plotly_chart(fig_risk)
            st.plotly_chart(fig_financial)

        else:
            st.write("No filings found for the given ticker.")
    else:
        st.write("Please enter a ticker symbol.")
