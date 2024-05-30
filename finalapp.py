# Install necessary packages if not already installed
# !pip install sec-edgar-downloader sec-api langchain streamlit openai

import os
import json
from sec_edgar_downloader import Downloader
from sec_api import XbrlApi
from langchain.vectorstores import SimpleVectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
import streamlit as st

# Initialize API clients
dl = Downloader()
xbrlApi = XbrlApi(api_key="your_sec_api_key")
openai.api_key = 'your_openai_api_key'

# Function to download 10-K filings
def download_10k_filings(tickers, start_year=1995, end_year=2023):
    for ticker in tickers:
        dl.get("10-K", ticker, after=f"{start_year}-01-01", before=f"{end_year}-12-31")

# Function to extract text from the downloaded filings using sec-api
def extract_text_from_filings(ticker):
    filings_dir = f"./sec-edgar-filings/{ticker}/10-K/"
    texts = []
    for filename in os.listdir(filings_dir):
        with open(os.path.join(filings_dir, filename), 'r', encoding='utf-8') as file:
            document_id = json.load(file)['filingId']
            report = xbrlApi.xbrl_to_dict(document_id)
            texts.append(report['document']['content'])
    return texts

# Define the tickers and download the filings
tickers = ['AAPL', 'GOOGL', 'MSFT']  # Example tickers
download_10k_filings(tickers)

# Extract texts from filings
all_texts = {ticker: extract_text_from_filings(ticker) for ticker in tickers}

# Prepare embeddings and vector store
embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
vector_store = SimpleVectorStore(embedding_function=embeddings.embed_text)

# Load documents into the vector store
for ticker in all_texts:
    documents = [{"text": text, "metadata": {"ticker": ticker}} for text in all_texts[ticker]]
    vector_store.add_documents(documents)

# Implement QA system
def ask_question(question):
    relevant_docs = vector_store.similarity_search(question, k=5)
    llm = OpenAI(openai_api_key=openai.api_key)
    context = " ".join([doc["text"] for doc in relevant_docs])
    response = llm.generate_text(f"Context: {context}\nQuestion: {question}\nAnswer:")
    return response["choices"][0]["text"].strip()

# Streamlit app
st.title("SEC EDGAR Filings Q&A System")

ticker = st.text_input("Enter company ticker:")
question = st.text_input("Ask a question about the filings:")

if st.button("Submit"):
    if ticker and question:
        response = ask_question(question)
        st.write(f"Answer: {response}")
    else:
        st.write("Please enter both a ticker and a question.")
