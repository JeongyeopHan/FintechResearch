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
from textblob import TextBlob
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

        # Download all 10-K filings for the ticker from 2018 onward
        dl.get("10-K", ticker, after="2018-11-01", before="2023-12-31")

        # Directory where filings are downloaded
        download_dir = os.path.join(".", "sec-edgar-filings", ticker, "10-K")

        # Ensure the download directory exists
        if not os.path.exists(download_dir):
            st.error(f"Download directory {download_dir} does not exist.")
            st.stop()

        st.write(f"Checking directory: {download_dir}")

        # Function to extract specific sections
        def extract_section(filepath, section_title, next_section_title):
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    soup = BeautifulSoup(file, 'html.parser')
                    text = soup.get_text()
                    start = text.find(section_title)
                    end = text.find(next_section_title, start)
                    if start != -1 and end != -1:
                        return text[start:end].strip()
                    elif start != -1:
                        return text[start:].strip()  # In case the end section is not found
                    return ""
            except FeatureNotFound:
                st.error("lxml parser not found. Please ensure it is installed.")
                st.stop()
            except Exception as e:
                st.error(f"Error processing file {filepath}: {e}")
                return ""

        # Iterate over downloaded filings directories and extract "Risk Factors" and "Management's Discussion and Analysis"
        for root, dirs, files in os.walk(download_dir):
            for subdir in dirs:
                subdir_path = os.path.join(root, subdir)
                st.write(f"Checking subdir: {subdir_path}")
                for file in os.listdir(subdir_path):
                    st.write(f"Found file: {file}")
                    if file == "full-submission.txt":
                        filepath = os.path.join(subdir_path, file)
                        st.write(f"Processing file: {filepath}")
                        risk_factors_text = extract_section(filepath, "Item 1A. Risk Factors", "Item 1B.")
                        mda_text = extract_section(filepath, "Item 7. Management's Discussion and Analysis", "Item 7A.")
                        if risk_factors_text:
                            st.write(f"Extracted Risk Factors from {filepath}")
                            filings.append(Document(page_content=risk_factors_text, metadata={"source": filepath}))
                        else:
                            st.write(f"No Risk Factors found in {filepath}")
                        if mda_text:
                            st.write(f"Extracted M&As from {filepath}")
                            mda_year = filepath.split(os.path.sep)[-3]
                            if mda_year in ["2023", "2022", "2021"]:
                                filings.append(Document(page_content=mda_text, metadata={"source": filepath, "type": "MDA"}))
                        else:
                            st.write(f"No M&As found in {filepath}")

        if filings:
            st.write(f"Found {len(filings)} filings with risk factors or M&As sections.")
            
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

            # Define the questions
            risk_factors_question = f"Summarize the main risks identified by {ticker} in its 10-K filings. In English."
            mda_question = f"Extract the M&As sections for the years 2023, 2022, and 2021 for {ticker}. In English."

            # Get answers from the agent
            risk_factors_response = agent({"input": risk_factors_question})
            mda_response = agent({"input": mda_question})

            # Display the responses
            st.write("### Risk Factors Summary")
            st.write(risk_factors_response["output"])

            st.write("### M&As Sections")
            st.write(mda_response["output"])

            # Analyze the sentiment of M&As sections
            mda_documents = [doc for doc in filings if doc.metadata.get("type") == "MDA"]
            sentiment_data = {"Year": [], "Positive Words": [], "Negative Words": []}

            for doc in mda_documents:
                year = doc.metadata["source"].split(os.path.sep)[-3]
                text = doc.page_content
                blob = TextBlob(text)
                positive_words = sum(1 for word in blob.words if TextBlob(word).sentiment.polarity > 0)
                negative_words = sum(1 for word in blob.words if TextBlob(word).sentiment.polarity < 0)
                sentiment_data["Year"].append(year)
                sentiment_data["Positive Words"].append(positive_words)
                sentiment_data["Negative Words"].append(negative_words)

            # Plot the sentiment analysis
            st.write("### Sentiment Analysis of M&As Sections")
            plt.figure(figsize=(10, 5))
            plt.bar(sentiment_data["Year"], sentiment_data["Positive Words"], color='green', alpha=0.6, label='Positive Words')
            plt.bar(sentiment_data["Year"], sentiment_data["Negative Words"], color='red', alpha=0.6, label='Negative Words')
            plt.xlabel("Year")
            plt.ylabel("Word Count")
            plt.title("Sentiment Analysis of M&As Sections (2021-2023)")
            plt.legend()
            st.pyplot(plt)
        else:
            st.write("No filings found for the given ticker.")
    else:
        st.write("Please enter a ticker symbol.")
