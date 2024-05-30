import streamlit as st
import os
from sec_api import ExtractorApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, AgentType, Tool
from pydantic import BaseModel, Field

# Get API keys from environment variables
extractor_api_key = os.getenv("EXTRACTOR_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize APIs
extractor_api = ExtractorApi(api_key=extractor_api_key)

# Initialize OpenAI API
os.environ["OPENAI_API_KEY"] = openai_api_key
llm = OpenAI(temperature=0)

# Streamlit app layout
st.title("SEC Filings Analysis with ChatGPT")

ticker = st.text_input("Enter the company ticker:")
if st.button("Analyze"):
    if ticker:
        filings = []

        # Construct the query to fetch 10-K filings for the given ticker
        query = {
            "query": {
                "query_string": {
                    "query": f"ticker:{ticker} AND filedAt:[1995-01-01 TO 2023-12-31] AND formType:\"10-K\""
                }
            },
            "from": "0",
            "size": "100",
            "sort": [{"filedAt": {"order": "desc"}}]
        }

        try:
            results = extractor_api.get_filings(query)
            for filing in results["filings"]:
                filing_url = filing["linkToFilingDetails"]
                filing_date = filing["filedAt"]

                try:
                    section_text = extractor_api.get_section(filing_url, "1A", "text")
                    filings.append({"date": filing_date, "text": section_text})
                except Exception as e:
                    st.write(f"Error fetching section 1A from {filing_url}: {e}")
        except Exception as e:
            st.write(f"Error fetching filings for ticker {ticker}: {e}")

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
