import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from langchain_chroma import Chroma  # Updated import
from langchain_ollama import OllamaEmbeddings


llm = LLM(
    model="ollama/llama3.2",
    base_url="http://localhost:11434"
)

class CompetitorInsightWorkflow:
    def __init__(self):
        """
        Initialize workflow with ChromaDB access
        """
        self.embeddings = OllamaEmbeddings(model="llama3.2")
        
        # Updated Chroma initialization
        self.vectorstore = Chroma(
            persist_directory="./chroma_db", 
            embedding_function=self.embeddings,
            collection_name="web_content"
        )
    
    def retrieve_context(self, query, top_k=5):
        """
        Retrieve context from vectorstore based on query
        """
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
        """
        Create specialized agents for competitor analysis
        """
        # Market Research Agent
        market_research_agent = Agent(
            role="Competitor Market Research Specialist",
            goal="Analyze the competitor's market positioning and product strategy",
            backstory="An expert market researcher with deep insights into e-commerce trends and competitive landscapes.",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )
        
        # Product Analysis Agent
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
        """
        Create tasks for each agent based on retrieved context
        """
        market_research_agent, product_analysis_agent = agents
        
        # Market Positioning Task
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
        
        # Product Analysis Task
        product_analysis_task = Task(
            description=f"""
            Conduct a detailed analysis of the competitor's product offerings using:
            {context}
            
            Create a report that covers:
            1. Product range and categories
            2. Key product features
            3. Pricing structure
            4. Potential gaps in their product lineup
            """,
            agent=product_analysis_agent,
            expected_output="Comprehensive product strategy analysis"
        )
        
        # Trend Prediction Task

        
        return [market_positioning_task, product_analysis_task]
    
    def run_analysis(self, query="Competitor e-commerce strategy"):
        """
        Run the complete competitor analysis workflow
        """
        # Retrieve context
        context = self.retrieve_context(query)
        
        # Create agents
        agents = self.create_agents()
        
        # Create tasks
        tasks = self.create_tasks(agents, context)
        
        # Create and run crew (Fixed verbose parameter)
        crew = Crew(
            agents=agents,
            tasks=tasks,
            verbose=True  # Changed from 2 to True
        )
        
        # Execute the workflow
        result = crew.kickoff()
        
        return result

def main():
    # Initialize and run workflow
    workflow = CompetitorInsightWorkflow()
    
    # You can modify the query to focus on specific aspects
    analysis_results = workflow.run_analysis(
        query="Best sellers collection product details"
    )
    
    # Print or further process results
    print(analysis_results)

if __name__ == "__main__":
    main()