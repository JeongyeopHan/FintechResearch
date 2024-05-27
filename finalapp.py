import streamlit as st
import os
import json
import pandas as pd
import re
from bs4 import BeautifulSoup
import openai

# Ensure directories exist
CLEANED_DATA_PATH = "cleaned_data"
ANALYSIS_RESULTS_PATH = "analysis_results"

os.makedirs(CLEANED_DATA_PATH, exist_ok=True)
os.makedirs(ANALYSIS_RESULTS_PATH, exist_ok=True)

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
            extracted_sections[key] = text[start_idx:end_idx]
        else:
            extracted_sections[key] = ""

    return extracted_sections

# Function to process and clean filings
def process_and_clean_filings(uploaded_files):
    extracted_data = []

    for uploaded_file in uploaded_files:
        text = uploaded_file.read().decode("utf-8")
        sections = extract_relevant_sections(text)
        cleaned_sections = {key: clean_text(value) for key, value in sections.items()}
        extracted_data.append({
            'Year': uploaded_file.name.split('.')[0],
            'Ticker': uploaded_file.name.split('_')[0],
            'MD&A': cleaned_sections['mdna'],
            'Risk Factors': cleaned_sections['risk_factors'],
            'Financials': cleaned_sections['financials']
        })
    
    ticker = extracted_data[0]['Ticker']
    cleaned_data_path = os.path.join(CLEANED_DATA_PATH, f"{ticker}_cleaned_filings.json")
    with open(cleaned_data_path, "w") as outfile:
        json.dump(extracted_data, outfile)
    
    return cleaned_data_path

# Function to load cleaned data
def load_cleaned_data(file_path):
    with open(file_path, 'r') as infile:
        return json.load(infile)

# Function to analyze text using OpenAI API
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

# Function to save analysis results
def save_analysis_results(insights_df, ticker, analysis_type):
    analysis_path = os.path.join(ANALYSIS_RESULTS_PATH, f"{ticker}_{analysis_type}_analysis_results.json")
    with open(analysis_path, "w") as outfile:
        json.dump(insights_df.to_dict(orient='records'), outfile)
    return analysis_path

# Streamlit app
st.title('10-K Filings Analysis App')

st.sidebar.header('Settings')
ticker_input = st.sidebar.text_input('Enter Company Ticker (e.g., AAPL, MSFT, TSLA)', 'AAPL')
openai_api_key = st.sidebar.text_input('Enter your OpenAI API Key', '')
email_address = "20150613rke3@gmail.com"

# Allow users to upload files
uploaded_files = st.file_uploader("Upload the downloaded 10-K filings", accept_multiple_files=True, type=["txt"])

if uploaded_files and st.sidebar.button('Analyze Uploaded Filings'):
    with st.spinner('Processing and cleaning filings...'):
        cleaned_data_path = process_and_clean_filings(uploaded_files)
        st.success('Processing complete!')
        
    with st.spinner('Loading cleaned data...'):
        cleaned_data = load_cleaned_data(cleaned_data_path)
        st.success('Data loaded!')

    with st.spinner('Analyzing risk factors...'):
        risk_factors_df = analyze_risk_factors(cleaned_data, openai_api_key)
        risk_factors_path = save_analysis_results(risk_factors_df, ticker_input, "risk_factors")
        st.success('Risk factors analysis complete!')

    with st.spinner('Analyzing MD&A sentiment...'):
        mda_sentiment_df = analyze_mda_sentiment(cleaned_data, openai_api_key)
        mda_sentiment_path = save_analysis_results(mda_sentiment_df, ticker_input, "mda_sentiment")
        st.success('MD&A sentiment analysis complete!')

    with st.spinner('Analyzing financial performance...'):
        financial_performance_df = analyze_financial_performance(cleaned_data, openai_api_key)
        financial_performance_path = save_analysis_results(financial_performance_df, ticker_input, "financial_performance")
        st.success('Financial performance analysis complete!')

    # Display results
    st.header('Analysis Results')

    st.subheader('Risk Factors Analysis')
    st.write(risk_factors_df)

    st.subheader('MD&A Sentiment Analysis')
    st.write(mda_sentiment_df)

    st.subheader('Financial Performance Analysis')
    st.write(financial_performance_df)
