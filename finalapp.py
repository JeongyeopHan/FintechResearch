import streamlit as st
import os
from edgar_crawler import EdgarCrawler
from edgar_crawler.extract_items import ItemExtractor
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

        # Configure and initialize EdgarCrawler
        crawler = EdgarCrawler(
            user_agent="Jeong 20150613rke3@gmail.com",
            start_year=1995,
            end_year=2023,
            filing_types=["10-K"],
            cik_tickers=[ticker],
            raw_filings_folder="RAW_FILINGS",
            indices_folder="INDICES",
            skip_present_indices=True,
        )

        # Download filings
        crawler.run()

        # Configure and initialize ItemExtractor
        extractor = ItemExtractor(
            raw_filings_folder="RAW_FILINGS",
            extracted_filings_folder="EXTRACTED_FILINGS",
            items_to_extract=["1A"],
            remove_tables=True,
            skip_extracted_filings=True,
        )

        # Extract items
        extracted_items = extractor.run()

        # Process extracted filings
        for item in extracted_items:
            if "1A" in item:
                filings.append({"text": item["1A"]})

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
