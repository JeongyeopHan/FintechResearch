# SEC Filings Analysis with Langchain and ChatGPT Deploying on Streamlit

This project is part of a programming task to analyze SEC 10-K filings using an LLM API. The project is divided into three main tasks: downloading the 10-K filings, analyzing the filings, and deploying a simple app to visualize the insights.


## Tech Stack
1. **Python**: The primary programming language used for its versatility and extensive libraries.
2. **Langchain**: Used for natural language processing and to facilitate interactions with the LLM.
3. **Streamlit**: Chosen for its simplicity and efficiency in creating web applications, especially for data science and machine learning projects.


## Insights and Their Importance

### Risk Factors Analysis

#### Insight: Identification of Major Risks
**Explanation**: Understanding the major risks identified by a company in its 10-K filings is crucial for investors, analysts, and stakeholders. These risks can significantly impact the company's performance and future prospects. By analyzing these risk factors, users can make more informed decisions about investing in or engaging with the company.

### MD&A (Management's Discussion and Analysis) Analysis

#### Insight: Key Strategic Initiatives
**Explanation**: The MD&A section outlines the company's strategic initiatives for future growth and how it plans to address potential challenges. This insight is valuable for users because it provides a window into the company's long-term vision and operational strategies. Investors and stakeholders can use this information to assess the company's growth potential and risk management capabilities.


## Rationale for the Choice of Streamlit
Streamlit was chosen for this project due to its user-friendly interface and ability to quickly deploy and share data applications. Its integration with Python makes it an ideal choice for building interactive applications for data analysis and machine learning.


## Additional Notes
OpenAI was used as the LLM API with Chroma and Langchain to improve the quality of LLM-generated responses. The project is divided into three parts (download.py, analyze.py, and app.py) for better modularity and maintainability. Due to some debugging issues, the subapp.py was used for the demonstration, but the modular version should work equivalently once fully debugged.


## Video Demo
[streamlit-subapp-2024-05-31-05-05-14.webm](https://github.com/JeongyeopHan/FintechResearch/assets/133887543/db1908a0-bdce-4c7f-a5a7-01d434ac364e)
