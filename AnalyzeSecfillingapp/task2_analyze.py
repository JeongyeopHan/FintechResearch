# analyze.py

import os
import tempfile
from bs4 import BeautifulSoup, FeatureNotFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, AgentType, Tool
from pydantic import BaseModel, Field
from langchain.schema import Document
import plotly.express as px
import pandas as pd

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
        raise RuntimeError("lxml parser not found. Please ensure it is installed.")
    except Exception as e:
        raise RuntimeError(f"Error processing file {filepath}: {e}")

def get_filings(download_dir):
    risk_factor_filings = []
    financial_statements = []

    for root, dirs, files in os.walk(download_dir):
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            for file in os.listdir(subdir_path):
                if file == "full-submission.txt":
                    filepath = os.path.join(subdir_path, file)

                    risk_factors_text = extract_section(filepath, "Item 1A.", "Item 1B.")
                    if risk_factors_text:
                        risk_factor_filings.append(Document(page_content=risk_factors_text, metadata={"source": filepath}))

                    financial_text = extract_section(filepath, "Item 8.", "Item 9.")
                    if financial_text:
                        financial_statements.append(Document(page_content=financial_text, metadata={"source": filepath}))

    return risk_factor_filings, financial_statements

def analyze_documents(risk_factor_filings, financial_statements):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    risk_split_texts = text_splitter.split_documents(risk_factor_filings)
    financial_split_texts = text_splitter.split_documents(financial_statements)

    embeddings = OpenAIEmbeddings()

    with tempfile.TemporaryDirectory() as temp_dir:
        risk_db = Chroma.from_documents(risk_split_texts, embeddings, persist_directory=temp_dir)
        risk_db.persist()

        financial_db = Chroma.from_documents(financial_split_texts, embeddings, persist_directory=temp_dir)
        financial_db.persist()

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

    financial_tools = [
        Tool(
            args_schema=DocumentInput,
            name="financial_document_tool",
            description="Useful for answering questions about financial statements in the document",
            func=RetrievalQA.from_chain_type(llm=llm, retriever=financial_db.as_retriever()),
        )
    ]

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

    risk_agent = initialize_agent(agent=AgentType.OPENAI_FUNCTIONS, tools=risk_tools, llm=llm, verbose=True)
    financial_agent = initialize_agent(agent=AgentType.OPENAI_FUNCTIONS, tools=financial_tools, llm=llm, verbose=True)

    risk_question = "Identify and rank the five major risks identified in the 10-K filings."
    financial_question = "Summarize the financial statements over the recent years."

    risk_response = risk_agent({"input": risk_question})
    financial_response = financial_agent({"input": financial_question})

    return risk_response["output"], financial_response["output"]

def create_bar_chart(labels, values, title):
    fig = go.Figure([go.Bar(x=labels, y=values)])
    fig.update_layout(title_text=title, xaxis_title="Risk Factors", yaxis_title="Importance")
    return fig

def create_line_chart(df, title):
    fig = px.line(df, x='Year', y=['Revenue', 'Net Income', 'Total Assets'], title=title)
    return fig
