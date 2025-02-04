import os
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
from datetime import datetime

class WebContentVectorizer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.app = FirecrawlApp(api_key=api_key)
        self.embeddings = OllamaEmbeddings(model="llama3.2")
        
    def scrape_content(self, url):
        """
        Scrape content from URL and return as markdown.
        """
        scrape_result = self.app.scrape_url(
            url,
            params={'formats': ['markdown']}
        )
        
        markdown_content = scrape_result.get('markdown', '')

        metadata = {
            'url': url,
            'scrape_date': datetime.now().isoformat(),
            'source': 'firecrawl'
        }
        
        return markdown_content, metadata

    def process_text(self, text):
        """
        Split text into chunks for vectorization.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        return splitter.split_text(text)

    def store_in_chroma(self, chunks, metadata, collection_name="web_content"):
        """
        Store text chunks in ChromaDB.
        """
        client = chromadb.PersistentClient(path="./chroma_db")
        
        metadatas = [{**metadata, 'chunk_id': i} for i in range(len(chunks))]
        
        vector_store = Chroma.from_texts(
            texts=chunks,
            embedding=self.embeddings,
            persist_directory="./chroma_db",
            collection_name=collection_name,
            metadatas=metadatas
        )
        
        vector_store.persist()
        return vector_store

def main():
    load_dotenv()
    api_key = os.getenv('FIRECRAWL_API_KEY')
    
    if not api_key:
        raise ValueError("FIRECRAWL_API_KEY not found in environment variables")
    
    vectorizer = WebContentVectorizer(api_key)
    url = "https://beminimalist.co/collections/best-sellers"
    
    try:
        print("Scraping content...")
        content, metadata = vectorizer.scrape_content(url)
        
        print("Processing text...")
        chunks = vectorizer.process_text(content)
        
        print("Storing in ChromaDB...")
        vectorizer.store_in_chroma(chunks, metadata)
        print("Content successfully vectorized and stored.")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
