import os
import json
import pandas as pd
import openai

## Set your OpenAI API key
openai.api_key = 'your_openai_api_key'

def load_cleaned_data(file_path):
    with open(file_path, 'r') as infile:
        return json.load(infile)

def analyze_text(question, text):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": question},
            {"role": "user", "content": text}
        ]
    )
    return response['choices'][0]['message']['content']

def analyze_risk_factors(data):
    insights = []
    for entry in data:
        risk_factors_text = entry['Risk Factors']
        year = entry['Year']
        ticker = entry['Ticker']
        analysis = analyze_text('Identify the major risk factors.', risk_factors_text)
        insights.append({
            'Year': year,
            'Ticker': ticker,
            'Risk Factors Analysis': analysis
        })
    return pd.DataFrame(insights)

def analyze_mda_sentiment(data):
    insights = []
    for entry in data:
        mda_text = entry['MD&A']
        year = entry['Year']
        ticker = entry['Ticker']
        analysis = analyze_text('Perform sentiment analysis.', mda_text)
        insights.append({
            'Year': year,
            'Ticker': ticker,
            'MD&A Sentiment Analysis': analysis
        })
    return pd.DataFrame(insights)

def analyze_financial_performance(data):
    insights = []
    for entry in data:
        financials_text = entry['Financials']
        year = entry['Year']
        ticker = entry['Ticker']
        analysis = analyze_text('Summarize key financial performance metrics such as revenue, net income, and EPS.', financials_text)
        insights.append({
            'Year': year,
            'Ticker': ticker,
            'Financial Performance Analysis': analysis
        })
    return pd.DataFrame(insights)

def save_analysis_results(insights_df, ticker, analysis_type):
    analysis_path = os.path.join("data/analysis_results", f"{ticker}_{analysis_type}_analysis_results.json")
    insights_df.to_json(analysis_path, orient='records')
    return analysis_path