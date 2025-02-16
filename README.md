# AI-Powered Competitive Insight Analysis

## 🚀 Overview
The **AI-Powered Competitive Insight Analysis** system is designed to extract strategic insights from competitor websites using AI-driven web scraping, data processing, and intelligent analysis. It helps businesses analyze **product strategies, pricing models, and market positioning** by leveraging multi-agent AI workflows.

## 🔍 Features
- **Automated Web Scraping** – Extracts structured data from competitor websites using **Firecrawl**.
- **AI-Powered Analysis** – Utilizes **CrewAI** agents to generate deep insights into **market positioning, pricing, and product lineup**.
- **Efficient Data Storage** – Processes and stores vectorized content using **ChromaDB** for quick retrieval.
- **Seamless Query Handling** – Uses **LangChain** and **Gemini API** to generate AI-driven insights.

## 🛠 Tech Stack
- **Firecrawl** – Web scraping automation
- **CrewAI** – AI agent-based workflow management
- **LangChain** – Framework for LLM-powered applications
- **ChromaDB** – Vector database for fast and efficient retrieval
- **Gemini API** – LLM for generating AI insights
- **Streamlit** – Interactive UI for user-friendly analysis
- **Python** – Core language for development

## 🏗 How It Works
1. **Scrape Website Data** – Firecrawl extracts web content in markdown format.
2. **Process and Vectorize Data** – The extracted content is split into chunks and stored in ChromaDB.
3. **AI Agent Analysis** – CrewAI agents analyze data for pricing, product strategy, and market positioning.
4. **Display Insights** – AI-generated insights are presented via a Streamlit dashboard.

## 🚀 Future Enhancements
- **Full Website Analysis** – Expand from single-page scraping to entire website analysis.
- **Retrieval-Augmented Generation (RAG)** – Enable precise extraction of specific insights from stored data.

## 🔧 Installation & Setup
### 1️⃣ Clone the Repository:
```bash
git clone https://github.com/your-repo/AI-Competitive-Insight.git
cd AI-Competitive-Insight
```

### 2️⃣ Install Dependencies:
```bash
pip install -r requirements.txt
```

### 3️⃣ Set Up Environment Variables:
Create a `.env` file in the root directory and add:
```env
FIRECRAWL_API_KEY=your_firecrawl_api_key
GOOGLE_API_KEY=your_google_api_key
```

### 4️⃣ Run the Application:
```bash
streamlit run main.py
```

## 🤝 Contributing
Feel free to fork this repository, submit issues, or contribute enhancements! 🚀

## 📜 License
This project is licensed under the MIT License.

