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
import streamlit as st
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
        income_statement_filings = []
        balance_sheet_filings = []
        cash_flow_statement_filings = []

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

                        # Extract Income Statement
                        income_statement_text = extract_section(filepath, "Item 8.", "Item 9.")
                        if income_statement_text:
                            income_statement_filings.append(Document(page_content=income_statement_text, metadata={"source": filepath}))

                        # Extract Balance Sheet
                        balance_sheet_text = extract_section(filepath, "Item 8.", "Item 9.")
                        if balance_sheet_text:
                            balance_sheet_filings.append(Document(page_content=balance_sheet_text, metadata={"source": filepath}))

                        # Extract Cash Flow Statement
                        cash_flow_statement_text = extract_section(filepath, "Item 8.", "Item 9.")
                        if cash_flow_statement_text:
                            cash_flow_statement_filings.append(Document(page_content=cash_flow_statement_text, metadata={"source": filepath}))

        if risk_factor_filings and mdna_filings and income_statement_filings and balance_sheet_filings and cash_flow_statement_filings:
            st.write(f"Found {len(risk_factor_filings)} filings with risk factors.")
            st.write(f"Found {len(mdna_filings)} filings with MD&A sections.")
            st.write(f"Found {len(income_statement_filings)} filings with Income Statements.")
            st.write(f"Found {len(balance_sheet_filings)} filings with Balance Sheets.")
            st.write(f"Found {len(cash_flow_statement_filings)} filings with Cash Flow Statements.")

            # Function to create Sankey diagram
            def create_sankey(labels, sources, targets, values, title):
                fig = go.Figure(data=[go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=labels,
                    ),
                    link=dict(
                        source=sources,
                        target=targets,
                        value=values,
                    ))])
                fig.update_layout(title_text=title, font_size=10)
                return fig

            # Example visualization for Income Statement
            labels = ["Revenue", "Cost of Revenue", "Gross Profit", "Operating Expenses", "Operating Profit", "Net Profit", "Products", "Services"]
            sources = [0, 0, 2, 2, 3, 4, 0, 0]
            targets = [2, 1, 4, 3, 5, 5, 2, 2]
            values = [383.3, 214.1, 169.1, 54.8, 114.3, 97.0, 298.1, 85.2]

            fig = create_sankey(labels, sources, targets, values, "Income Statement FY23")
            st.plotly_chart(fig)

            # Example visualization for Balance Sheet
            labels_bs = ["Current Assets", "Non-Current Assets", "Total Assets", "Current Liabilities", "Non-Current Liabilities", "Total Liabilities", "Equity"]
            sources_bs = [0, 1, 2, 2, 4, 3, 2]
            targets_bs = [2, 2, 3, 4, 5, 6, 6]
            values_bs = [143.6, 209.0, 352.6, 145.3, 145.1, 290.4, 73.8]

            fig_bs = create_sankey(labels_bs, sources_bs, targets_bs, values_bs, "Balance Sheet FY23")
            st.plotly_chart(fig_bs)

            # Example visualization for Cash Flow Statement
            labels_cf = ["Net Income", "Cash from Operations", "Cash from Investing", "Cash from Financing", "Net Cash Flow"]
            sources_cf = [0, 1, 1, 1, 1]
            targets_cf = [1, 2, 3, 4, 4]
            values_cf = [97.0, 110.5, 3.7, -108.5, 5.8]

            fig_cf = create_sankey(labels_cf, sources_cf, targets_cf, values_cf, "Cash Flow Statement FY23")
            st.plotly_chart(fig_cf)

        else:
            st.write("No filings found for the given ticker.")
    else:
        st.write("Please enter a ticker symbol.")
