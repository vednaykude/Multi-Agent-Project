#!/usr/bin/env python3
"""
Interactive Multi-Agent Research Assistant
"""

import os
import json
from typing import Dict, List, Any, TypedDict
from datetime import datetime

# imports
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langgraph.graph import StateGraph, END

# State management for the graph
class ResearchState(TypedDict):
    query: str
    retrieved_docs: List[Document]
    summary: str
    fact_check_results: Dict[str, Any]
    final_response: str
    metadata: Dict[str, Any]

class InteractiveResearchAssistant:
    def __init__(self, model_name: str = "llama3"):
        """Initialize the research assistant with Ollama LLaMA 3"""
        print("🤖 Initializing Research Assistant...")
        self.model_name = model_name
        self.llm = Ollama(model=model_name, temperature=0.1) #low temperature for factual consistency
        
        print("📚 Loading embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2" # Efficient embeddings model for text similarity
        )
        
        self.vector_store = None
        self.setup_knowledge_base()
        
        self.workflow = self.create_workflow() # Create the LangGraph workflow, DAG
        print("✅ Research Assistant ready!")
    
    def setup_knowledge_base(self):
        """Setup vector database with comprehensive sample documents"""
        print("🗃️ Setting up knowledge base...")
        
        # Dummy sample documents for knowledge base
        # In a real application, these would be loaded from files or a database
        sample_docs = [
            # Climate Change & Agriculture
            Document(
                page_content="""Climate change significantly impacts global agriculture through 
                rising temperatures, altered precipitation patterns, and increased frequency of 
                extreme weather events. These changes affect crop yields, growing seasons, and 
                agricultural productivity worldwide. Farmers are adapting by changing planting 
                dates, switching to drought-resistant crops, and implementing new irrigation 
                technologies.""",
                metadata={"source": "climate_agriculture_overview", "topic": "climate_agriculture"}
            ),
            Document(
                page_content="""Rising global temperatures affect crop physiology, with heat stress 
                reducing yields for temperature-sensitive crops like wheat and rice. Optimal growing 
                zones are shifting poleward at a rate of approximately 70 kilometers per decade. 
                This forces farmers to adapt their practices or relocate their operations.""",
                metadata={"source": "temperature_impacts", "topic": "climate_agriculture"}
            ),
            Document(
                page_content="""Changes in precipitation patterns create both drought and flooding 
                challenges for farmers. Water scarcity affects irrigation systems, while excessive 
                rainfall can lead to soil erosion and crop damage. Some regions experience more 
                intense droughts, while others face unprecedented flooding.""",
                metadata={"source": "water_impacts", "topic": "climate_agriculture"}
            ),
            
            # Technology & AI
            Document(
                page_content="""Artificial Intelligence is revolutionizing multiple industries 
                through machine learning, natural language processing, and computer vision. 
                AI applications range from autonomous vehicles and medical diagnosis to 
                financial trading and content creation. The technology continues to advance 
                rapidly with new breakthroughs in neural networks and deep learning.""",
                metadata={"source": "ai_overview", "topic": "technology"}
            ),
            Document(
                page_content="""Machine learning algorithms can identify patterns in large 
                datasets that humans might miss. These systems learn from data without being 
                explicitly programmed for specific tasks. Applications include recommendation 
                systems, fraud detection, predictive maintenance, and personalized medicine.""",
                metadata={"source": "machine_learning", "topic": "technology"}
            ),
            
            # Health & Medicine
            Document(
                page_content="""Personalized medicine uses genetic information, lifestyle factors, 
                and environmental data to tailor medical treatments to individual patients. 
                This approach can improve treatment effectiveness, reduce adverse reactions, 
                and optimize drug dosages. Advances in genomics and biomarkers are making 
                personalized medicine more accessible.""",
                metadata={"source": "personalized_medicine", "topic": "health"}
            ),
            Document(
                page_content="""Mental health awareness has increased significantly, leading to 
                better understanding of conditions like depression, anxiety, and PTSD. Treatment 
                approaches include therapy, medication, lifestyle changes, and emerging 
                technologies like virtual reality therapy and digital mental health platforms.""",
                metadata={"source": "mental_health", "topic": "health"}
            ),
            
            # Economics & Business
            Document(
                page_content="""Remote work has transformed the modern workplace, accelerated by 
                the COVID-19 pandemic. Companies have adopted hybrid models, invested in digital 
                collaboration tools, and restructured their operations. This shift has implications 
                for productivity, work-life balance, urban planning, and commercial real estate.""",
                metadata={"source": "remote_work", "topic": "business"}
            ),
            Document(
                page_content="""Sustainable business practices are becoming essential for 
                long-term success. Companies are adopting circular economy principles, 
                reducing carbon footprints, and implementing ESG (Environmental, Social, 
                Governance) frameworks. Consumers increasingly prefer brands that demonstrate 
                environmental and social responsibility.""",
                metadata={"source": "sustainable_business", "topic": "business"}
            ),
            
            # Science & Research
            Document(
                page_content="""Quantum computing represents a paradigm shift in computational 
                power, using quantum mechanical phenomena to process information. Unlike 
                classical bits, quantum bits (qubits) can exist in multiple states simultaneously, 
                potentially solving complex problems exponentially faster than traditional computers.""",
                metadata={"source": "quantum_computing", "topic": "science"}
            )
        ]
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=sample_docs,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        print(f"✅ Knowledge base loaded with {len(sample_docs)} documents")
    
    def retriever_agent(self, state: ResearchState) -> ResearchState:
        """Agent 1: Retrieve relevant documents from knowledge base"""
        print("🔍 Searching knowledge base...")
        
        query = state["query"]
        
        # Search vector database
        relevant_docs = self.vector_store.similarity_search(
            query, k=5  # Retrieve top 5 most relevant documents
        )
        
        state["retrieved_docs"] = relevant_docs
        state["metadata"] = {
            "retrieval_timestamp": datetime.now().isoformat(),
            "docs_found": len(relevant_docs)
        }
        
        print(f"✅ Found {len(relevant_docs)} relevant documents")
        for i, doc in enumerate(relevant_docs[:3]):  # Show first 3
            topic = doc.metadata.get('topic', 'general')
            print(f"   📄 Doc {i+1}: {topic} - {doc.page_content[:100]}...")
        
        return state
    
    def summarizer_agent(self, state: ResearchState) -> ResearchState:
        """Agent 2: Summarize retrieved documents"""
        print("📝 Creating comprehensive summary...")
        
        docs = state["retrieved_docs"]
        query = state["query"]
        
        # Combine document contents
        combined_content = "\n\n".join([doc.page_content for doc in docs])
        
        # Create summarization prompt
        summary_prompt = PromptTemplate(
            input_variables=["query", "content"],
            template="""
            Based on the following documents, create a comprehensive summary that addresses this query: {query}

            Documents:
            {content}

            Please provide a well-structured summary that:
            1. Directly addresses the query
            2. Synthesizes information from multiple sources
            3. Maintains factual accuracy
            4. Is concise but comprehensive
            5. Organizes information logically

            Summary:
            """
        )
        
        # Generate summary
        summary_chain = LLMChain(llm=self.llm, prompt=summary_prompt)
        summary = summary_chain.run(query=query, content=combined_content)
        
        state["summary"] = summary.strip()
        print("✅ Summary created")
        return state
    
    def fact_checker_agent(self, state: ResearchState) -> ResearchState:
        """Agent 3: Fact-check the summary"""
        print("🔍 Verifying information accuracy...")
        
        summary = state["summary"]
        docs = state["retrieved_docs"]
        
        # Create fact-checking prompt
        fact_check_prompt = PromptTemplate(
            input_variables=["summary", "sources"],
            template="""
            Please fact-check the following summary against the provided source documents.
            
            Summary to check:
            {summary}
            
            Source documents:
            {sources}
            
            Provide a fact-check analysis with:
            - accuracy_score: Rate the overall accuracy from 1-10
            - verified_claims: List claims that are well-supported by sources
            - concerns: Any potential issues or missing context
            - confidence: Your confidence level in the summary
            
            Format your response as a clear analysis (not JSON).
            
            Fact-check analysis:
            """
        )
        
        # Prepare source material
        sources_text = "\n\n".join([
            f"Source {i+1}: {doc.page_content}" 
            for i, doc in enumerate(docs)
        ])
        
        # Generate fact-check
        fact_check_chain = LLMChain(llm=self.llm, prompt=fact_check_prompt)
        fact_check_result = fact_check_chain.run(
            summary=summary, 
            sources=sources_text
        )
        
        state["fact_check_results"] = {
            "analysis": fact_check_result,
            "sources_count": len(docs)
        }
        
        print("✅ Fact-checking completed")
        return state
    
    def responder_agent(self, state: ResearchState) -> ResearchState:
        """Agent 4: Generate final response"""
        print("📤 Crafting final response...")
        
        query = state["query"]
        summary = state["summary"]
        fact_check = state["fact_check_results"]
        
        # Create response prompt
        response_prompt = PromptTemplate(
            input_variables=["query", "summary", "fact_check"],
            template="""
            Create a comprehensive, well-structured response to this query: {query}
            
            Based on:
            Summary: {summary}
            
            Fact-check results: {fact_check}
            
            Please provide a response that:
            1. Directly answers the user's question
            2. Is well-structured and easy to read
            3. Uses clear headings and organization
            4. Incorporates verified information
            5. Acknowledges any limitations or areas for further research
            6. Provides actionable insights where appropriate
            
            Response:
            """
        )
        
        # Generate final response
        response_chain = LLMChain(llm=self.llm, prompt=response_prompt)
        final_response = response_chain.run(
            query=query,
            summary=summary,
            fact_check=str(fact_check)
        )
        
        state["final_response"] = final_response.strip()
        print("✅ Final response ready!")
        return state
    
    def create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(ResearchState)
        
        # Add nodes (agents)
        workflow.add_node("retriever", self.retriever_agent)
        workflow.add_node("summarizer", self.summarizer_agent)
        workflow.add_node("fact_checker", self.fact_checker_agent)
        workflow.add_node("responder", self.responder_agent)
        
        # Define the flow
        workflow.set_entry_point("retriever")
        workflow.add_edge("retriever", "summarizer")
        workflow.add_edge("summarizer", "fact_checker")
        workflow.add_edge("fact_checker", "responder")
        workflow.add_edge("responder", END)
        
        return workflow.compile()
    
    def research(self, query: str) -> Dict[str, Any]:
        """Main research method"""
        print(f"\n🚀 Starting research for: '{query}'")
        print("=" * 80)
        
        # Initialize state
        initial_state = ResearchState(
            query=query,
            retrieved_docs=[],
            summary="",
            fact_check_results={},
            final_response="",
            metadata={}
        )
        
        # Run the workflow
        final_state = self.workflow.invoke(initial_state)
        
        print("=" * 80)
        print("🎉 Research completed!")
        
        return {
            "query": query,
            "final_response": final_state["final_response"],
            "summary": final_state["summary"],
            "fact_check_results": final_state["fact_check_results"],
            "metadata": final_state["metadata"],
            "sources_count": len(final_state["retrieved_docs"])
        }
    
    def interactive_mode(self):
        """Interactive mode for asking questions"""
        print("\n" + "="*80)
        print("🎯 INTERACTIVE RESEARCH ASSISTANT")
        print("="*80)
        print("Ask me anything! I'll research it using my multi-agent system.")
        print("Type 'quit', 'exit', or 'bye' to stop.")
        print("Examples:")
        print("  • How does AI impact healthcare?")
        print("  • What are the effects of remote work?")
        print("  • Tell me about quantum computing")
        print("  • How does climate change affect farming?")
        print("-" * 80)
        
        while True:
            try:
                # Get user input
                query = input("\n💭 Your Question: ").strip()
                
                # Check for exit commands
                if query.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\n👋 Thanks for using the Research Assistant! Goodbye!")
                    break
                
                # Skip empty queries
                if not query:
                    print("Please enter a question!")
                    continue
                
                # Process the query
                result = self.research(query)
                
                # Display results
                print(f"\n📋 RESEARCH RESULTS:")
                print(f"🔍 Query: {result['query']}")
                print(f"\n📄 ANSWER:")
                print("-" * 40)
                print(result['final_response'])
                print("-" * 40)
                print(f"📊 Sources used: {result['sources_count']}")
                
                # Ask if user wants to continue
                continue_choice = input("\n❓ Ask another question? (y/n): ").strip().lower()
                if continue_choice in ['n', 'no']:
                    print("\n👋 Thanks for using the Research Assistant! Goodbye!")
                    break
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again with a different question.")

def main():
    """Main function - choose between interactive or example mode"""
    print("🤖 Multi-Agent Research Assistant")
    print("Choose mode:")
    print("1. Interactive mode (ask your own questions)")
    print("2. Run examples")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    # Initialize the research assistant
    assistant = InteractiveResearchAssistant()
    
    if choice == "1":
        # Interactive mode
        assistant.interactive_mode()
    else:
        # Example mode
        queries = [
            "How does artificial intelligence impact healthcare?",
            "What are the benefits and challenges of remote work?",
            "Tell me about quantum computing and its potential applications"
        ]
        
        for query in queries:
            result = assistant.research(query)
            print(f"\n📋 RESEARCH RESULTS:")
            print(f"Query: {result['query']}")
            print(f"\n📄 Answer:")
            print(result['final_response'])
            print(f"\n📊 Sources: {result['sources_count']}")
            print("\n" + "="*80)
            input("Press Enter for next example...")

if __name__ == "__main__":
    main()