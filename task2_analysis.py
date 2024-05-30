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
        raise Exception("lxml parser not found. Please ensure it is installed.")
    except Exception as e:
        raise Exception(f"Error processing file {filepath}: {e}")

def get_filings(download_dir):
    risk_factor_filings = []
    mdna_filings = []

    for root, dirs, files in os.walk(download_dir):
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            for file in os.listdir(subdir_path):
                if file == "full-submission.txt":
                    filepath = os.path.join(subdir_path, file)
                    risk_factors_text = extract_section(filepath, "Item 1A.", "Item 1B.")
                    if risk_factors_text:
                        risk_factor_filings.append(Document(page_content=risk_factors_text, metadata={"source": filepath}))
                    mdna_text = extract_section(filepath, "Item 7.", "Item 7A.")
                    if mdna_text:
                        mdna_filings.append(Document(page_content=mdna_text, metadata={"source": filepath}))

    return risk_factor_filings, mdna_filings

class DocumentInput(BaseModel):
    question: str = Field()

def analyze_filings(risk_factor_filings, mdna_filings, openai_api_key):
    os.environ["OPENAI_API_KEY"] = openai_api_key
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    risk_split_texts = text_splitter.split_documents(risk_factor_filings)
    mdna_split_texts = text_splitter.split_documents(mdna_filings)

    embeddings = OpenAIEmbeddings()

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            risk_db = Chroma.from_documents(risk_split_texts, embeddings, persist_directory=temp_dir)
            risk_db.persist()
            mdna_db = Chroma.from_documents(mdna_split_texts, embeddings, persist_directory=temp_dir)
            mdna_db.persist()
        except Exception as e:
            raise Exception(f"Error initializing Chroma: {e}")

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

    return risk_agent, mdna_agent
