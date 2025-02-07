import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
import litellm

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("GOOGLE_API_KEY")
litellm.api_key = os.getenv("GOOGLE_API_KEY")

# Initialize the model using litellm wrapped in ChatOpenAI
llm = ChatOpenAI(
    model_name="gemini/gemini-pro",  # Note the provider prefix
    temperature=0.5,
    openai_api_key=os.getenv("GOOGLE_API_KEY"),
    max_tokens=1000
)

class CustomOutputParser:
    def parse(self, output):
        if isinstance(output, dict):
            if 'raw' in output:
                return output['raw']
            elif 'tasks_output' in output:
                return "\n\n".join(task.raw for task in output['tasks_output'] if hasattr(task, 'raw'))
        return str(output)

class CompetitorInsightWorkflow:
    def __init__(self):
        """
        Initialize workflow with ChromaDB access
        """
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        
        self.vectorstore = Chroma(
            persist_directory="./chroma_db", 
            embedding_function=self.embeddings,
            collection_name="web_content"
        )
    
    def retrieve_context(self, query, top_k=5):
        try:
            results = self.vectorstore.similarity_search(query, k=top_k)
            return "\n\n".join([
                f"Source: {doc.metadata.get('url', 'Unknown')}\n{doc.page_content}" 
                for doc in results
            ])
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return ""
    
    def create_agents(self):
        market_research_agent = Agent(
            role="Competitor Market Research Specialist",
            goal="Analyze the competitor's market positioning and product strategy",
            backstory="An expert market researcher with deep insights into e-commerce trends and competitive landscapes.",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )
        
        product_analysis_agent = Agent(
            role="Product Strategy Analyst",
            goal="Conduct in-depth analysis of competitor's product offerings and pricing",
            backstory="A meticulous product strategist who can break down product details and identify unique selling propositions.",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )
        
        return market_research_agent, product_analysis_agent
    
    def create_tasks(self, agents, context):
        market_research_agent, product_analysis_agent = agents
        
        market_positioning_task = Task(
            description=f"""
            Analyze the competitor's market positioning based on the following context:
            {context}
            
            Provide a comprehensive report that includes:
            1. Target market segments
            2. Unique value propositions
            3. Pricing strategy
            4. Market differentiation factors
            """,
            agent=market_research_agent,
            expected_output="Detailed market positioning analysis report"
        )
        
        product_analysis_task = Task(
            description=f"""
            Conduct a detailed analysis of the competitor's product offerings using:
            {context}
            
            Create a report that covers:
            1. Product range and categories
            2. Key product features
            3. Pricing structure
            """,
            agent=product_analysis_agent,
            expected_output="Comprehensive product strategy analysis"
        )
        
        return [market_positioning_task, product_analysis_task]
    
    def run_analysis(self, query="Competitor e-commerce strategy"):
        context = self.retrieve_context(query)
        agents = self.create_agents()
        tasks = self.create_tasks(agents, context)
        
        crew = Crew(
            agents=agents,
            tasks=tasks,
            verbose=True,
            output_parser=CustomOutputParser()  # Add custom output parser
        )
        
        result = crew.kickoff()
        return result

def main():
    workflow = CompetitorInsightWorkflow()
    analysis_results = workflow.run_analysis(
        query="Best sellers collection product details"
    )
    print(analysis_results)

if __name__ == "__main__":
    main()