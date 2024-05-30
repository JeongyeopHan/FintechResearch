__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import tempfile
import logging
from bs4 import BeautifulSoup, FeatureNotFound
from sec_edgar_downloader import Downloader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import ChatOpenAI, OpenAI
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, AgentType, Tool
from pydantic import BaseModel, Field
from langchain.schema import Document
from tenacity import retry, wait_exponential, stop_after_attempt
from langchain.prompts import PromptTemplate

# Ensure the correct sqlite library is used
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Constants
SOURCE_DIRECTORY = "./sec-edgar-filings"
PERSIST_DIRECTORY = "./persist"

# Define prompt template
prompt_template = """Your name is Bob and you are a human not an AI assistant. These are risk factors section of SEC 10-K annual filings by companies. Sometimes referred as 10-K SEC filings. If they ask about SEC or 10-K filings it is referred to the risk factor section what we have. Use the following risk factor sections of 10-K filings from various company filings to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}

# Define DocumentInput class
class DocumentInput(BaseModel):
    question: str = Field()

# Function to create database from documents
def create_DB():
    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")
    files = os.listdir(SOURCE_DIRECTORY)
    print(files)
    for file in files:
        file_name = file.split('.')[0]
        print(file_name)
        file_path = os.path.join(SOURCE_DIRECTORY, file)
        print("File path: ", file_path)
        loader = TextLoader(file_path)
        pages = loader.load_and_split()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(pages)
        embeddings = OpenAIEmbeddings()
        db = Chroma.from_documents(docs, embeddings, persist_directory=f"{PERSIST_DIRECTORY}/{file}")
    return ""

# Function to create tools for each retriever
def create_tools():
    files = os.listdir(SOURCE_DIRECTORY)
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    tools = []
    embeddings = OpenAIEmbeddings()
    for file in files:
        file_name = file.split('.')[0]
        db = Chroma(persist_directory=f"{PERSIST_DIRECTORY}/{file}", embedding_function=embeddings)
        retrievers = db.as_retriever()
        tools.append(
            Tool(
                args_schema=DocumentInput,
                name=file_name,
                description=f"useful when you want to answer questions about {file_name}",
                func=RetrievalQA.from_chain_type(llm=llm, retriever=retrievers, chain_type_kwargs=chain_type_kwargs),
            )
        )
    return tools

# Function to load agents
def load_agents(tools):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    langchain.debug = True
    agent = initialize_agent(
        agent=AgentType.OPENAI_FUNCTIONS,
        tools=tools,
        llm=llm,
        verbose=True,
    )
    return agent

# Streamlit app layout
st.title("SEC Filings Analysis with ChatGPT")

ticker = st.text_input("Enter the company ticker:")
if st.button("Analyze"):
    if ticker:
        filings = []

        # Initialize Downloader
        dl = Downloader("JHON", "jhondoe@gmail.com", ".")

        # Download all 10-K filings for the ticker from 2023 onward
        dl.get("10-K", ticker, after="2023-11-01", before="2023-12-31")

        # Directory where filings are downloaded
        download_dir = os.path.join(SOURCE_DIRECTORY, ticker, "10-K")

        # Ensure the download directory exists
        if not os.path.exists(download_dir):
            st.error(f"Download directory {download_dir} does not exist.")
            st.stop()

        st.write(f"Checking directory: {download_dir}")

        # Function to extract risk factors section
        def extract_risk_factors(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    soup = BeautifulSoup(file, 'lxml')
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
            except FeatureNotFound:
                st.error("lxml parser not found. Please ensure it is installed.")
                st.stop()
            except Exception as e:
                st.error(f"Error processing file {filepath}: {e}")
                return ""

        # Iterate over downloaded filings directories and extract "Risk Factors"
        for root, dirs, files in os.walk(download_dir):
            for subdir in dirs:
                subdir_path = os.path.join(root, subdir)
                st.write(f"Checking subdir: {subdir_path}")
                for file in os.listdir(subdir_path):
                    st.write(f"Found file: {file}")
                    if file == "full-submission.txt":
                        filepath = os.path.join(subdir_path, file)
                        st.write(f"Processing file: {filepath}")
                        section_text = extract_risk_factors(filepath)
                        if section_text:
                            filings.append(Document(page_content=section_text, metadata={"source": filepath}))

        if filings:
            st.write(f"Found {len(filings)} filings with risk factors.")
            
            # Save risk factors to text files
            os.makedirs(SOURCE_DIRECTORY, exist_ok=True)
            for i, filing in enumerate(filings):
                with open(os.path.join(SOURCE_DIRECTORY, f"filing_{i}.txt"), "w") as f:
                    f.write(filing.page_content)
            
            # Create DB from the downloaded documents
            create_DB()

            # Create tools for each retriever
            tools = create_tools()

            # Load agent with the created tools
            agent = load_agents(tools)

            # Define the question
            question = f"Summarize the main risks identified by {ticker} in its 10-K filings. In English."

            # Get answer from the agent
            response = agent({"input": question})
            st.write(response["output"])
        else:
            st.write("No filings found for the given ticker.")
    else:
        st.write("Please enter a ticker symbol.")
