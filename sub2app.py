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
import plotly.graph_objects as go

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
        mdna_filings = []

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

                        # Extract MD&A
                        mdna_text = extract_section(filepath, "Item 7.", "Item 7A.")
                        if mdna_text:
                            mdna_filings.append(Document(page_content=mdna_text, metadata={"source": filepath}))

        if risk_factor_filings and mdna_filings:
            st.write(f"Found {len(risk_factor_filings)} filings with risk factors.")
            st.write(f"Found {len(mdna_filings)} filings with MD&A sections.")

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

            # Process MD&A with Langchain
            mdna_split_texts = text_splitter.split_documents(mdna_filings)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    mdna_db = Chroma.from_documents(mdna_split_texts, embeddings, persist_directory=temp_dir)
                    mdna_db.persist()
                except Exception as e:
                    st.error(f"Error initializing Chroma for MD&A: {e}")
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

            mdna_tools = [
                Tool(
                    args_schema=DocumentInput,
                    name="mdna_document_tool",  # Ensure the name matches the required pattern
                    description="Useful for answering questions about MD&A sections in the document",
                    func=RetrievalQA.from_chain_type(llm=llm, retriever=mdna_db.as_retriever()),
                )
            ]

            risk_agent = initialize_agent(agent=AgentType.OPENAI_FUNCTIONS, tools=risk_tools, llm=llm, verbose=True)
            mdna_agent = initialize_agent(agent=AgentType.OPENAI_FUNCTIONS, tools=mdna_tools, llm=llm, verbose=True)

            # Define the questions
            risk_question = f"What are the top five risk factors identified by {ticker} in its 10-K filings ranked by importance? In English."
            mdna_question = "What are the key strategic initiatives outlined by the company for future growth, and how does the company plan to address any identified risks or challenges in the coming fiscal year?"

            # Get answers from the agents
            risk_response = risk_agent({"input": risk_question})
            mdna_response = mdna_agent({"input": mdna_question})

            st.write("Risk Factors Analysis:")
            st.write(risk_response["output"])

            st.write("MD&A Analysis:")
            st.write(mdna_response["output"])

            # Function to create bar chart for risk factors
            def create_bar_chart(labels, values, title):
                fig = go.Figure([go.Bar(x=labels, y=values)])
                fig.update_layout(title_text=title, xaxis_title="Risk Factors", yaxis_title="Importance")
                return fig

            # Example visualization for ranked risk factors
            risk_factors = risk_response["output"].split("\n")
            risk_labels = [f"Risk {i+1}" for i in range(len(risk_factors))]
            risk_values = list(range(1, len(risk_factors) + 1))  # Assign a rank value

            fig_risk = create_bar_chart(risk_labels, risk_values, "Ranked Risk Factors")
            st.plotly_chart(fig_risk)

            # Add functionality to visualize Income Statement, Balance Sheet, and Cash Flow Statement similar to previous instructions

        else:
            st.write("No filings found for the given ticker.")
    else:
        st.write("Please enter a ticker symbol.")
