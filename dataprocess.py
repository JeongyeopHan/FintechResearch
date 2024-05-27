import os
import re
from bs4 import BeautifulSoup
import json
import pandas as pd

def remove_html_tags(text):
    return BeautifulSoup(text, "html.parser").get_text()

def clean_text(text):
    text = remove_html_tags(text)    
    
    """ Patterns to be removed are bullet piont, quotation marks, quotation dash, opening and closing double quotiation mark, 
    Em dash, right single quotation mark, semicolon, bullet point, zero-width space, en dash, doulbe quotation mark, 
    ampersand, middle dot, consecutive newline characters, and headings.
    """
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
    
    # Save cleaned data to a JSON file
    cleaned_data_path = os.path.join("data/cleaned_data", f"{ticker}_cleaned_filings.json")
    with open(cleaned_data_path, "w") as outfile:
        json.dump(extracted_data, outfile)
    
    return cleaned_data_path

# Example usage
ticker = "AAPL"
download_path = "data/tickers"
cleaned_data_path = process_and_clean_filings(ticker, download_path)
print(f"Cleaned data saved to: {cleaned_data_path}")
