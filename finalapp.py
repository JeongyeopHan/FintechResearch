import streamlit as st
import os
from sec_edgar_downloader import Downloader
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

        # Initialize Downloader
        dl = Downloader("JHON", "jhondoe@gmail.com", ".")

        # Download all 10-K filings for the ticker from 1995 to 2023
        dl.get("10-K", ticker, after="1995-01-01", before="2023-12-31")

        # Directory where filings are downloaded
        download_dir = os.path.join(".", "sec-edgar-filings", ticker, "10-K")

        # Debug: List files in the download directory
        st.write(f"Checking directory: {download_dir}")
        if os.path.exists(download_dir):
            st.write(f"Directory exists: {download_dir}")
            files = os.listdir(download_dir)
            if files:
                st.write(f"Files in directory: {files}")
            else:
                st.write("No files found in the directory.")
        else:
            st.write(f"Directory does not exist: {download_dir}")

        # Function to extract risk factors section
        def extract_risk_factors(filepath):
            with open(filepath, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file, 'html.parser')
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

        # Iterate over downloaded filings and extract "Risk Factors"
        for root, dirs, files in os.walk(download_dir):
            for file in files:
                if file.endswith(".html"):
                    filepath = os.path.join(root, file)
                    section_text = extract_risk_factors(filepath)
                    if section_text:
                        filings.append({"text": section_text})

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
    else:
        st.write("Please enter a ticker symbol.")
