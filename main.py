import os
import sys
import traceback
import streamlit as st
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

load_dotenv()

try:
    from Utils.web_scraper import WebContentVectorizer
    from Agents.Analysts import CompetitorInsightWorkflow
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.error("Please ensure the script files are in the same directory.")
    st.stop()

def format_analysis_results(results):
    """Format the analysis results for better display"""
    if isinstance(results, dict):
        if 'raw' in results:
            return results['raw']
        elif 'tasks_output' in results:
            return "\n\n".join(task.raw for task in results['tasks_output'] if hasattr(task, 'raw'))
    return str(results)

def main():
    st.set_page_config(
        page_title="Competitor Insight AI",
        page_icon="🕵️",
        layout="wide"
    )

    st.title("🔍 Competitor Insight Analysis")
    st.markdown("Uncover deep competitive intelligence using AI-powered analysis")

    st.sidebar.header("Analysis Configuration")
    url = st.sidebar.text_input("Enter Competitor Website URL", placeholder="https://example.com")
    
    analysis_focus = st.sidebar.selectbox(
        "Analysis Focus",
        [
            "Overall Strategy",
            "Product Lineup",
            "Market Positioning",
            "Best Sellers Collection",
            "Pricing Strategy"
        ]
    )

    if st.sidebar.button("Run Competitive Analysis"):
        if not url:
            st.error("Please enter a valid URL")
            return

        with st.spinner("Analyzing competitor... This may take a few minutes"):
            try:
                api_key = os.getenv('FIRECRAWL_API_KEY')
                if not api_key:
                    st.error("Firecrawl API key not found. Please set FIRECRAWL_API_KEY in .env file.")
                    return

                vectorizer = WebContentVectorizer(api_key)

                st.write("🌐 Scraping website content...")
                content, metadata = vectorizer.scrape_content(url)

                st.write("🔬 Processing and vectorizing content...")
                chunks = vectorizer.process_text(content)
                vectorizer.store_in_chroma(chunks, metadata)

                st.write("🤖 Preparing AI agents...")
                workflow = CompetitorInsightWorkflow()

                query_map = {
                    "Overall Strategy": "Comprehensive competitor e-commerce strategy",
                    "Product Lineup": "Product range and offerings",
                    "Market Positioning": "Market positioning and target segments",
                    "Best Sellers Collection": "Best sellers collection product details",
                    "Pricing Strategy": "Pricing structure and competitive pricing"
                }

                st.write("📊 Generating competitive insights...")
                analysis_results = workflow.run_analysis(
                    query=query_map[analysis_focus]
                )

                # Format results before displaying
                formatted_results = format_analysis_results(analysis_results)

                st.success("Analysis Complete!")
                st.markdown("## 🔍 Competitive Insights")
                st.markdown(formatted_results)  # Using markdown for better formatting

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error(f"Traceback: {traceback.format_exc()}")

    st.sidebar.markdown("---")
    st.sidebar.info("Powered by AI Competitive Intelligence")

if __name__ == "__main__":
    main()