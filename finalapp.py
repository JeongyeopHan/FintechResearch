__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup, FeatureNotFound
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
import matplotlib.pyplot as plt
import pandas as pd

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
st.title("SEC Filings Sentiment Analysis with ChatGPT")

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

        # Function to extract risk factors or M&A section
        def extract_section(filepath, section_title):
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    soup = BeautifulSoup(file, 'lxml')
                    section_text = ""
                    section_found = False
                    for line in soup.get_text().splitlines():
                        if section_title in line:
                            section_found = True
                        if section_found:
                            section_text += line + "\n"
                            if "Item 1B." in line or "Item 7." in line:  # assuming sections are delimited by next major item
                                break
                    return section_text
            except FeatureNotFound:
                st.error("lxml parser not found. Please ensure it is installed.")
                st.stop()
            except Exception as e:
                st.error(f"Error processing file {filepath}: {e}")
                return ""

        # Extract Risk Factors and M&A sections from filings
        sections = {
            "Risk Factors": "Item 1A.",
            "M&A Sentiment": "Mergers and Acquisitions"
        }

        for section_name, section_title in sections.items():
            section_filings = []
            for root, dirs, files in os.walk(download_dir):
                for subdir in dirs:
                    subdir_path = os.path.join(root, subdir)
                    st.write(f"Checking subdir: {subdir_path}")
                    for file in os.listdir(subdir_path):
                        st.write(f"Found file: {file}")
                        if file == "full-submission.txt":
                            filepath = os.path.join(subdir_path, file)
                            st.write(f"Processing file: {filepath}")
                            section_text = extract_section(filepath, section_title)
                            if section_text:
                                section_filings.append(Document(page_content=section_text, metadata={"source": filepath}))
            filings.append((section_name, section_filings))

        if filings:
            sentiment_data = []

            for section_name, docs in filings:
                if docs:
                    st.write(f"Found {len(docs)} filings with {section_name.lower()} section.")

                    for doc in docs:
                        year = doc.metadata['source'].split('/')[-2].split('-')[1]  # Extract year from the directory name
                        sentiment_prompt = f"""
                        You are provided with a section of text from an SEC filing.
                        Analyze the sentiment of the text. Provide the sentiment analysis result for the year {year}, including the overall sentiment score, the number of positive words, and the number of negative words.

                        Text to analyze:
                        {doc.page_content}
                        """
                        response = llm(sentiment_prompt).content
                        st.write(f"Sentiment Analysis for document {doc.metadata['source']} ({section_name}):")
                        st.write(response)
                        
                        # Extract sentiment analysis results from the response
                        # Assuming the response is formatted in a way we can parse it
                        overall_score = ...  # Extract overall sentiment score from the response
                        positive_words = ...  # Extract number of positive words from the response
                        negative_words = ...  # Extract number of negative words from the response

                        sentiment_data.append({
                            'Year': year,
                            'Section': section_name,
                            'Overall_Score': overall_score,
                            'Positive_Words': positive_words,
                            'Negative_Words': negative_words
                        })

            sentiment_df = pd.DataFrame(sentiment_data)

            # Plot sentiment analysis results
            fig, ax = plt.subplots(3, 1, figsize=(10, 12))

            # Plot for all years
            sentiment_df.plot(kind='bar', x='Year', y=['Positive_Words', 'Negative_Words'], ax=ax[0])
            ax[0].set_title('Sentiment Analysis for All Years')
            ax[0].set_ylabel('Word Count')

            # Plot for Risk Factors
            risk_factors_df = sentiment_df[sentiment_df['Section'] == 'Risk Factors']
            risk_factors_df.plot(kind='bar', x='Year', y=['Positive_Words', 'Negative_Words'], ax=ax[1])
            ax[1].set_title('Sentiment Analysis for Risk Factors')
            ax[1].set_ylabel('Word Count')

            # Plot for M&A Sentiment
            ma_sentiment_df = sentiment_df[sentiment_df['Section'] == 'M&A Sentiment']
            ma_sentiment_df.plot(kind='bar', x='Year', y=['Positive_Words', 'Negative_Words'], ax=ax[2])
            ax[2].set_title('Sentiment Analysis for M&A Sentiment')
            ax[2].set_ylabel('Word Count')

            st.pyplot(fig)

        else:
            st.write(f"No filings found with the specified sections for the given ticker.")
    else:
        st.write("Please enter a ticker symbol.")
