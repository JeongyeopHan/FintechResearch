import os
from bs4 import BeautifulSoup, FeatureNotFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
import tempfile

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

def analyze_filings(download_dir):
    risk_factor_filings = []
    mdna_filings = []

    for root, dirs, files in os.walk(download_dir):
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            for file in os.listdir(subdir_path):
                if file == "full-submission.txt":
                    filepath = os.path.join(subdir_path, file)

                    # Extract Risk Factors
                    risk_factors_text = extract_section(filepath, "Item 1A.", "Item 1B.")
                    if risk_factors_text:
                        risk_factor_filings.append(Document(page_content=risk_factors_text, metadata={"source": filepath}))

                    # Extract MD&A
                    mdna_text = extract_section(filepath, "Item 7.", "Item 7A.")
                    if mdna_text:
                        mdna_filings.append(Document(page_content=mdna_text, metadata={"source": filepath}))

    if not risk_factor_filings and not mdna_filings:
        raise Exception("No filings found for the given ticker.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    risk_split_texts = text_splitter.split_documents(risk_factor_filings)
    mdna_split_texts = text_splitter.split_documents(mdna_filings)

    embeddings = OpenAIEmbeddings()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            risk_db = Chroma.from_documents(risk_split_texts, embeddings, persist_directory=temp_dir)
            risk_db.persist()
        except Exception as e:
            raise Exception(f"Error initializing Chroma for risk factors: {e}")

        try:
            mdna_db = Chroma.from_documents(mdna_split_texts, embeddings, persist_directory=temp_dir)
            mdna_db.persist()
        except Exception as e:
            raise Exception(f"Error initializing Chroma for MD&A: {e}")

    return risk_db, mdna_db
