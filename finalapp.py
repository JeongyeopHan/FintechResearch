import streamlit as st
import os
import json
import pandas as pd
import re
from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader
import openai

# Function to download 10-K filings
@st.cache_data
def download_10k_filings(ticker, emailaddress, download_path):
    download_path = f"/tmp/{ticker}_10k"  # Temporary path for download
    os.makedirs(download_path, exist_ok=True)
    dl = Downloader(ticker, emailaddress, download_path)
    try:
        print(f"Downloading 10-K for {ticker}")
        dl.get("10-K", ticker, after="1995-12-31", before="2023-01-01")
    except Exception as e:
        print(f"An error occurred for {ticker}: {e}")
    return download_path

# Function to remove HTML tags
def remove_html_tags(text):
    return BeautifulSoup(text, "html.parser").get_text()

# Function to clean text
def clean_text(text):
    text = remove_html_tags(text)
    patterns = [
        r'##TABLE_END', r'##TABLE_START', r'&#160;', r'&#8226;', r'&#8220; &#8221;', 
        r'&#8220;&#8212;', r'&#8220;', r'&#8221;', r'&#8212;', r'&#8217;', r'&#59;', 
        r'&#9679;', r'&#8203;', r'&#8211;', r'&#34;', r'#38;', r'&#8729;', r'\n+',
        r'Item [0-9]\.\s*\w+'  
    ]
    regex_patterns = [re.compile(pattern) for pattern in patterns]
    cleaned_text = text
    for pattern in regex_patterns:
        cleaned_text = pattern.sub('', cleaned_text)
    return cleaned_text

# Function to extract relevant sections
def extract_relevant_sections(text):
    sections = {
        "mdna": ("Management's Discussion and Analysis", "Quantitative and Qualitative Disclosures About Market Risk"),
        "risk_factors": ("Risk Factors", "Unresolved Staff Comments"),
        "financials": ("Selected Financial Data", "Management's Discussion and Analysis")
    }

    extracted_sections = {}
    for key, (start_phrase, end_phrase) in sections.items():
        start_idx = text.find(start_phrase)
        end_idx = text.find(end_phrase, start_idx)
        if start_idx != -1 and end_idx != -1:
            extracted_sections[key] = text[start_idx:endidx]
        else:
            extracted_sections[key] = ""

    return extracted_sections

# Function to process and clean filings
@st.cache_data
def process_and_clean_filings(ticker, download_path):
    filings_dir = os.path.join(download_path, ticker, "10-K")
    extracted_data = []

    for filename in os.listdir(filings_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(filings_dir, filename), 'r') as file:
                text = file.read()
                sections = extract_relevant_sections(text)
                cleaned_sections = {key: clean_text(value) for key, value in sections.items()}
                extracted_data.append({
                    'Year': filename.split('.')[0],
                    'Ticker': ticker,
                    'MD&A': cleaned_sections['mdna'],
                    'Risk Factors': cleaned_sections['risk_factors'],
                    'Financials': cleaned_sections['financials']
                })

    return extracted_data

# Function to analyze text using OpenAI API
@st.cache_data
def analyze_text(question, text, api_key):
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": question},
            {"role": "user", "content": text}
        ]
    )
    return response['choices'][0]['message']['content']

# Function to analyze risk factors
def analyze_risk_factors(data, api_key):
    insights = []
    for entry in data:
        risk_factors_text = entry['Risk Factors']
        year = entry['Year']
        ticker = entry['Ticker']
        analysis = analyze_text('Identify the major risk factors.', risk_factors_text, api_key)
        insights.append({
            'Year': year,
            'Ticker': ticker,
            'Risk Factors Analysis': analysis
        })
    return pd.DataFrame(insights)

# Function to analyze MD&A sentiment
def analyze_mda_sentiment(data, api_key):
    insights = []
    for entry in data:
        mda_text = entry['MD&A']
        year = entry['Year']
        ticker = entry['Ticker']
        analysis = analyze_text('Perform sentiment analysis.', mda_text, api_key)
        insights.append({
            'Year': year,
            'Ticker': ticker,
            'MD&A Sentiment Analysis': analysis
        })
    return pd.DataFrame(insights)

# Function to analyze financial performance
def analyze_financial_performance(data, api_key):
    insights = []
    for entry in data:
        financials_text = entry['Financials']
        year = entry['Year']
        ticker = entry['Ticker']
        analysis = analyze_text('Summarize key financial performance metrics such as revenue, net income, and EPS.', financials_text, api_key)
        insights.append({
            'Year': year,
            'Ticker': ticker,
            'Financial Performance Analysis': analysis
        })
    return pd.DataFrame(insights)

# Streamlit app
st.title('10-K Filings Analysis App')

st.sidebar.header('Settings')
ticker_input = st.sidebar.text_input('Enter Company Ticker (e.g., AAPL, MSFT, TSLA)', 'AAPL')
openai_api_key = st.sidebar.text_input('Enter your OpenAI API Key', '')
email_address = "20150613rke3@gmail.com"

if st.sidebar.button('Download and Analyze'):
    with st.spinner('Downloading 10-K filings...'):
        download_path = download_10k_filings(ticker_input, email_address)
        st.success('Download complete!')
        
    with st.spinner('Processing and cleaning filings...'):
        cleaned_data = process_and_clean_filings(ticker_input, download_path)
        st.success('Processing complete!')
        
    with st.spinner('Analyzing risk factors...'):
        risk_factors_df = analyze_risk_factors(cleaned_data, openai_api_key)
        st.success('Risk factors analysis complete!')

    with st.spinner('Analyzing MD&A sentiment...'):
        mda_sentiment_df = analyze_mda_sentiment(cleaned_data, openai_api_key)
        st.success('MD&A sentiment analysis complete!')

    with st.spinner('Analyzing financial performance...'):
        financial_performance_df = analyze_financial_performance(cleaned_data, openai_api_key)
        st.success('Financial performance analysis complete!')

    # Display results
    st.header('Analysis Results')

    st.subheader('Risk Factors Analysis')
    st.write(risk_factors_df)

    st.subheader('MD&A Sentiment Analysis')
    st.write(mda_sentiment_df)

    st.subheader('Financial Performance Analysis')
    st.write(financial_performance_df)
