import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, AgentType, Tool
from pydantic import BaseModel, Field

# Get API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

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

        def get_sec_filings(ticker):
            base_url = "https://www.sec.gov/cgi-bin/browse-edgar"
            params = {
                "action": "getcompany",
                "CIK": ticker,
                "type": "10-K",
                "dateb": "",
                "owner": "exclude",
                "start": "",
                "output": "xml",
                "count": "100"
            }
            response = requests.get(base_url, params=params)
            soup = BeautifulSoup(response.content, "lxml")
            return soup.find_all("filing")

        def download_filing(filing_url):
            response = requests.get(filing_url)
            return response.content.decode("utf-8")

        def extract_risk_factors(filing_text):
            soup = BeautifulSoup(filing_text, "html.parser")
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

        filings_metadata = get_sec_filings(ticker)
        for filing in filings_metadata:
            filing_date = filing.find("datefiled").text
            filing_url = "https://www.sec.gov" + filing.find("filinghref").text.replace("-index.htm", ".txt")

            try:
                filing_text = download_filing(filing_url)
                section_text = extract_risk_factors(filing_text)
                filings.append({"date": filing_date, "text": section_text})
            except Exception as e:
                st.write(f"Error fetching section 1A for date {filing_date}: {e}")

        if filings:
            # Process filings with Langchain
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = [doc["text"] for doc in filings]
            split_texts = text_splitter.split_documents(texts)

            embeddings = OpenAIEmbeddings()
            db = Chroma.from_documents(split_texts, embeddings, persist_directory="path/to/persist")
            db.persist()

            class DocumentInput(BaseModel):
                question: str = Field()

            tools = [
                Tool(
                    args_schema=DocumentInput,
                    name="Document Tool",
                    description="Useful for answering questions about the document",
                    func=RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever()),
                )
            ]

            agent = initialize_agent(agent=AgentType.OPENAI_FUNCTIONS, tools=tools, llm=llm, verbose=True)

            # Define the question
            question = f"Summarize the main risks identified by {ticker} in its 10-K filings. In English."

            # Get answer from the agent
            response = agent({"input": question})
            st.write(response["output"])
        else:
            st.write("No filings found for the given ticker.")
