import streamlit as st
import os
from sec_api import ExtractorApi, XbrlApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import RetrievalQA, Tool
from pydantic import BaseModel, Field

# Get API keys from environment variables
extractor_api_key = os.getenv("EXTRACTOR_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize APIs
extractor_api = ExtractorApi(api_key=extractor_api_key)
xbrl_api = XbrlApi(api_key=extractor_api_key)

# Initialize OpenAI API
os.environ["OPENAI_API_KEY"] = openai_api_key
llm = OpenAI(temperature=0)

# Streamlit app layout
st.title("SEC Filings Analysis with ChatGPT")

ticker = st.text_input("Enter the company ticker:")
if st.button("Analyze"):
    if ticker:
        filings = []

        # Extract filings for the given ticker from 1995 to 2023
        for year in range(1995, 2024):
            try:
                # Get the filing metadata
                htm_url = f"https://www.sec.gov/Archives/edgar/data/{ticker}/{year}.htm"
                xbrl_json = xbrl_api.xbrl_to_json(htm_url=htm_url)

                # Extract the 'Risk Factors' section
                section_text = extractor_api.get_section(htm_url, "1A", "text")
                filings.append({"year": year, "text": section_text})
            except Exception as e:
                st.write(f"Error fetching data for year {year}: {e}")

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
                    func=RetrievalQA.from_chain_type(llm=llm, retriever=db),
                )
            ]

            agent = initialize_agent(agent=AgentType.OPENAI_FUNCTIONS, tools=tools, llm=llm, verbose=True)

            # Define the questions
            questions = [
                f"Summarize {ticker}'s financial performance over the past years, including revenue growth, profitability (net income), and margins. In English",
                f"Identify and analyze {ticker}'s earnings per share (EPS diluted and basic), calculate Return on equity (ROE) and Debt-to-Equity ratio for the past few years. In English",
                f"Add bullet points for main risks identified by {ticker} in its 10-K filing. In English"
            ]

            # Get answers from the agent
            for question in questions:
                response = agent({"input": question})
                st.write(response["output"])
        else:
            st.write("No filings found for the given ticker.")
