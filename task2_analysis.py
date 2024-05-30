import os
from bs4 import BeautifulSoup, FeatureNotFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, AgentType, Tool
from pydantic import BaseModel, Field
from langchain.schema import Document
import tempfile

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
        raise Exception("lxml parser not found. Please ensure it is installed.")
    except Exception as e:
        raise Exception(f"Error processing file {filepath}: {e}")

def analyze_filings(filings):
    # Process filings with Langchain
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_texts = text_splitter.split_documents(filings)

    embeddings = OpenAIEmbeddings()

    # Use a temporary directory for Chroma persistence
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            db = Chroma.from_documents(split_texts, embeddings, persist_directory=temp_dir)
            db.persist()
        except Exception as e:
            raise Exception(f"Error initializing Chroma: {e}")

    class DocumentInput(BaseModel):
        question: str = Field()

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

    tools = [
        Tool(
            args_schema=DocumentInput,
            name="Document Tool",
            description="Useful for answering questions about the document",
            func=RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever()),
        )
    ]

    agent = initialize_agent(agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, tools=tools, llm=llm, verbose=True)

    # Define the question
    question = "Summarize the main risks identified in the 10-K filings. In English."

    # Get answer from the agent
    response = agent({"input": question})
    return response["output"]
