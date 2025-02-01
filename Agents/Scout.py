import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from crewai import Agent, Task, LLM, Crew, Process
from firecrawl import FirecrawlApp

#Loading api key
load_dotenv()

api_key = os.getenv('FIRECRAWL_API_KEY')


#defining llm
Mistral_llm = LLM(
    model="ollama/mistral",
    base_url="http://localhost:11434"
)

from firecrawl import FirecrawlApp

# Initialize Firecrawl with API key
app = FirecrawlApp(api_key=api_key)  # Replace with your actual API key

# Scrape the Minimalist Best Selling page with markdown format
scrape_result = app.scrape_url("https://beminimalist.co/collections/best-sellers", 
                               params={'formats': ['markdown']})

# Get markdown content
markdown_content = scrape_result.get('markdown', '')

# Save results to a markdown file
with open("minimalist_best_sellers.md", "w", encoding="utf-8") as file:
    file.write(markdown_content)

print("Scraping completed. Data saved in 'minimalist_best_sellers.md'")
