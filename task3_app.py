__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from task1_download import download_filings
from task2_analyze import get_filings, analyze_documents, create_bar_chart, create_line_chart

# Get API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Ensure the API keys are not None
if not openai_api_key:
    st.error("API keys are not set properly. Please check your environment variables.")
    st.stop()

# Initialize OpenAI API
os.environ["OPENAI_API_KEY"] = openai_api_key

# Streamlit app layout
st.title("SEC Filings Analysis with ChatGPT")

ticker = st.text_input("Enter the company ticker:")
if st.button("Analyze"):
    if ticker:
        download_dir = download_filings(ticker)

        if not os.path.exists(download_dir):
            st.error(f"Download directory {download_dir} does not exist.")
            st.stop()

        st.write(f"Checking directory: {download_dir}")

        risk_factor_filings, financial_statements = get_filings(download_dir)

        if risk_factor_filings and financial_statements:
            st.write(f"Found {len(risk_factor_filings)} filings with risk factors.")
            st.write(f"Found {len(financial_statements)} filings with financial statements.")

            risk_response, financial_response = analyze_documents(risk_factor_filings, financial_statements)

            st.write("Risk Factors Analysis:")
            st.write(risk_response)

            st.write("Financial Statements Summary:")
            st.write(financial_response)

            # Example risk factors analysis output (replace with actual response)
            risk_factors_ranked = [
                {"Risk Factor": "Data Security Standards", "Importance": 5},
                {"Risk Factor": "Fluctuating Net Sales", "Importance": 4},
                {"Risk Factor": "Gross Margins Pressure", "Importance": 3},
                {"Risk Factor": "New Business Strategies", "Importance": 2},
                {"Risk Factor": "IT System Failures", "Importance": 1}
            ]

            # Example financial summary output (replace with actual response)
            financial_summary = [
                {"Year": 2020, "Revenue": 274515, "Net Income": 57411, "Total Assets": 323888},
                {"Year": 2021, "Revenue": 365817, "Net Income": 94680, "Total Assets": 351002},
                {"Year": 2022, "Revenue": 394328, "Net Income": 99983, "Total Assets": 351002}
            ]

            # Convert to DataFrame for visualization
            risk_factors_df = pd.DataFrame(risk_factors_ranked)
            financial_summary_df = pd.DataFrame(financial_summary)

            # Create bar charts for risk factors
            fig_risk = create_bar_chart(risk_factors_df['Risk Factor'], risk_factors_df['Importance'], 'Major Risk Factors Ranked by Importance')

            # Create line charts for financial summary
            fig_financial = create_line_chart(financial_summary_df, 'Financial Summary Over Recent Years')

            st.plotly_chart(fig_risk)
            st.plotly_chart(fig_financial)

        else:
            st.write("No filings found for the given ticker.")
    else:
        st.write("Please enter a ticker symbol.")
