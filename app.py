import streamlit as st
import os
import json
import pandas as pd
import plotly.express as px
from download import download_10k_filings
from dataprocess import process_and_clean_filings
from text_analysis import analyze_risk_factors, analyze_mda_sentiment, analyze_financial_performance, load_cleaned_data, save_analysis_results

# Set paths
DOWNLOAD_PATH = "data/tickers"
CLEANED_DATA_PATH = "data/cleaned_data"
ANALYSIS_RESULTS_PATH = "data/analysis_results"

# Ensure directories exist
os.makedirs(DOWNLOAD_PATH, exist_ok=True)
os.makedirs(CLEANED_DATA_PATH, exist_ok=True)
os.makedirs(ANALYSIS_RESULTS_PATH, exist_ok=True)

# Streamlit app
st.title('10-K Filings Analysis App')

st.sidebar.header('Settings')
ticker_input = st.sidebar.text_input('Enter Company Ticker (e.g., AAPL, MSFT, TSLA)', 'AAPL')

if st.sidebar.button('Download and Analyze'):
    with st.spinner('Downloading 10-K filings...'):
        download_10k_filings(ticker_input, DOWNLOAD_PATH)
        st.success('Download complete!')
        
    with st.spinner('Processing and cleaning filings...'):
        cleaned_data_path = process_and_clean_filings(ticker_input, DOWNLOAD_PATH)
        st.success('Processing complete!')
        
    with st.spinner('Loading cleaned data...'):
        cleaned_data = load_cleaned_data(cleaned_data_path)
        st.success('Data loaded!')

    with st.spinner('Analyzing risk factors...'):
        risk_factors_df = analyze_risk_factors(cleaned_data)
        risk_factors_path = save_analysis_results(risk_factors_df, ticker_input, "risk_factors")
        st.success('Risk factors analysis complete!')

    with st.spinner('Analyzing MD&A sentiment...'):
        mda_sentiment_df = analyze_mda_sentiment(cleaned_data)
        mda_sentiment_path = save_analysis_results(mda_sentiment_df, ticker_input, "mda_sentiment")
        st.success('MD&A sentiment analysis complete!')

    with st.spinner('Analyzing financial performance...'):
        financial_performance_df = analyze_financial_performance(cleaned_data)
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

    # Visualization
    st.subheader('MD&A Sentiment Over the Years')
    fig = px.line(mda_sentiment_df, x='Year', y='MD&A Sentiment Analysis', title=f'{ticker_input} MD&A Sentiment Analysis Over the Years')
    st.plotly_chart(fig)

    st.subheader('Financial Performance Over the Years')
    fig = px.line(financial_performance_df, x='Year', y='Financial Performance Analysis', title=f'{ticker_input} Financial Performance Over the Years')
    st.plotly_chart(fig)
