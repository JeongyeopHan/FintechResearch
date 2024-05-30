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

# Initialize OpenAI API
os.environ["OPENAI_API_KEY"] = openai_api_key
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

# Streamlit app layout
st.title("SEC Filings Analysis with ChatGPT")

ticker = st.text_input("Enter the company ticker:")
if st.button("Analyze"):
    if ticker:
        try:
            download_dir = download_filings(ticker)
            st.write(f"Download directory: {download_dir}")

            risk_db, mdna_db = analyze_filings(download_dir)
            st.write(f"Found filings with risk factors and MD&A sections.")

            class DocumentInput(BaseModel):
                question: str = Field()

            risk_tools = [
                Tool(
                    args_schema=DocumentInput,
                    name="risk_document_tool",
                    description="Useful for answering questions about risk factors in the document",
                    func=RetrievalQA.from_chain_type(llm=llm, retriever=risk_db.as_retriever()),
                )
            ]

            mdna_tools = [
                Tool(
                    args_schema=DocumentInput,
                    name="mdna_document_tool",
                    description="Useful for answering questions about MD&A sections in the document",
                    func=RetrievalQA.from_chain_type(llm=llm, retriever=mdna_db.as_retriever()),
                )
            ]

            risk_agent = initialize_agent(agent=AgentType.OPENAI_FUNCTIONS, tools=risk_tools, llm=llm, verbose=True)
            mdna_agent = initialize_agent(agent=AgentType.OPENAI_FUNCTIONS, tools=mdna_tools, llm=llm, verbose=True)

            risk_question = f"Identify five major risks identified by {ticker} in its 10-K filings. In English."
            mdna_question = "What are the key strategic initiatives outlined by the company for future growth, and how does the company plan to address any identified risks or challenges in the coming fiscal year?"

            risk_response = risk_agent({"input": risk_question})
            mdna_response = mdna_agent({"input": mdna_question})

            st.write("Risk Factors Analysis:")
            st.write(risk_response["output"])

            st.write("MD&A Analysis:")
            st.write(mdna_response["output"])

        except Exception as e:
            st.error(str(e))
    else:
        st.write("Please enter a ticker symbol.")

