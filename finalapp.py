__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import tempfile
from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup, FeatureNotFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, AgentType, Tool
from pydantic import BaseModel, Field
from langchain.schema import Document
import matplotlib.pyplot as plt

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
        filings = []

        # Initialize Downloader
        dl = Downloader("Jeong", "20150613rke3@gmail.com", ".")

        # Download all 10-K filings for the ticker from 2023 onward
        dl.get("10-K", ticker, after="2021-12-31", before="2024-01-01")

        # Directory where filings are downloaded
        download_dir = os.path.join(".", "sec-edgar-filings", ticker, "10-K")

        # Ensure the download directory exists
        if not os.path.exists(download_dir):
            st.error(f"Download directory {download_dir} does not exist.")
            st.stop()

        st.write(f"Checking directory: {download_dir}")

        # Function to extract M&A section
        def extract_ma_section(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    soup = BeautifulSoup(file, 'lxml')
                    ma_section = ""
                    ma_found = False
                    for line in soup.get_text().splitlines():
                        if "Mergers and Acquisitions" in line:
                            ma_found = True
                        if ma_found:
                            ma_section += line + "\n"
                            if "Item 2." in line:
                                break
                    return ma_section
            except FeatureNotFound:
                st.error("lxml parser not found. Please ensure it is installed.")
                st.stop()
            except Exception as e:
                st.error(f"Error processing file {filepath}: {e}")
                return ""

        # Iterate over downloaded filings directories and extract M&A section
        for root, dirs, files in os.walk(download_dir):
            for subdir in dirs:
                subdir_path = os.path.join(root, subdir)
                st.write(f"Checking subdir: {subdir_path}")
                for file in os.listdir(subdir_path):
                    st.write(f"Found file: {file}")
                    if file == "full-submission.txt":
                        filepath = os.path.join(subdir_path, file)
                        st.write(f"Processing file: {filepath}")
                        section_text = extract_ma_section(filepath)
                        if section_text:
                            filings.append(Document(page_content=section_text, metadata={"source": filepath}))

        if filings:
            st.write(f"Found {len(filings)} filings with M&A sections.")
            
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
                    st.error(f"Error initializing Chroma: {e}")
                    st.stop()

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

            agent = initialize_agent(agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, tools=tools, llm=llm, verbose=True)

            # Define the sentiment analysis question
            question = f"Analyze the sentiment of the M&A text. Provide the sentiment analysis result for the year 2023, including the overall sentiment score, the number of positive words, and the number of negative words."

            # Get answer from the agent
            response = agent({"input": question})
            st.write(response["output"])

            # Plotting sentiment results
            sentiment_data = response["output"].split("\n")
            positive_words = int([line for line in sentiment_data if "number of positive words" in line][0].split(":")[1].strip())
            negative_words = int([line for line in sentiment_data if "number of negative words" in line][0].split(":")[1].strip())

            labels = ['Positive Words', 'Negative Words']
            values = [positive_words, negative_words]

            fig, ax = plt.subplots()
            ax.bar(labels, values, color=['green', 'red'])
            ax.set_xlabel('Sentiment')
            ax.set_ylabel('Number of Words')
            ax.set_title('Sentiment Analysis of M&A Section for 2023')

            st.pyplot(fig)
        else:
            st.write("No filings found for the given ticker.")
    else:
        st.write("Please enter a ticker symbol.")
