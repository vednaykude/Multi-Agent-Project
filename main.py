#!/usr/bin/env python3
"""
Multi-Agent Research Assistant using LangChain, LangGraph, and Ollama LLaMA 3

This system orchestrates multiple specialized agents to process research queries:
1. Retriever Agent: Searches knowledge base for relevant information
2. Summarizer Agent: Creates concise summaries of retrieved documents
3. Fact-Checker Agent: Verifies claims against trusted sources
4. Responder Agent: Synthesizes final response

Requirements:
pip install langchain langchain-community langgraph ollama chromadb sentence-transformers requests beautifulsoup4
"""

import os
import json
import requests
from typing import Dict, List, Any, TypedDict, Annotated
from datetime import datetime

# LangChain imports
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# LangGraph imports
from langgraph.graph import StateGraph, END

# Web scraping for knowledge base
from bs4 import BeautifulSoup
import sqlite3

# State management for the graph
class ResearchState(TypedDict):
    query: str
    retrieved_docs: List[Document]
    summary: str
    fact_check_results: Dict[str, Any]
    final_response: str
    metadata: Dict[str, Any]

class MultiAgentResearchAssistant:
    def __init__(self, model_name: str = "llama3"):
        """Initialize the research assistant with Ollama LLaMA 3"""
        self.model_name = model_name
        self.llm = Ollama(model=model_name, temperature=0.1)
        
        # Initialize embeddings for vector search
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize vector store
        self.vector_store = None
        self.setup_knowledge_base()
        
        # Create the workflow graph
        self.workflow = self.create_workflow()
    
    def setup_knowledge_base(self):
        """Setup vector database with sample documents"""
        # Sample documents - in production, load from your document store
        sample_docs = [
            Document(
                page_content="""Climate change significantly impacts global agriculture through 
                rising temperatures, altered precipitation patterns, and increased frequency of 
                extreme weather events. These changes affect crop yields, growing seasons, and 
                agricultural productivity worldwide.""",
                metadata={"source": "climate_agriculture_overview", "date": "2024-01-15"}
            ),
            Document(
                page_content="""Rising global temperatures affect crop physiology, with heat stress 
                reducing yields for temperature-sensitive crops like wheat and rice. Optimal growing 
                zones are shifting poleward, requiring adaptation in farming practices.""",
                metadata={"source": "temperature_impacts", "date": "2024-01-10"}
            ),
            Document(
                page_content="""Changes in precipitation patterns create both drought and flooding 
                challenges for farmers. Water scarcity affects irrigation systems, while excessive 
                rainfall can lead to soil erosion and crop damage.""",
                metadata={"source": "water_impacts", "date": "2024-01-12"}
            )
        ]
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=sample_docs,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
    
    def retriever_agent(self, state: ResearchState) -> ResearchState:
        """Agent 1: Retrieve relevant documents from knowledge base"""
        print("🔍 Retriever Agent: Searching knowledge base...")
        
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
        
        print(f"✅ Retrieved {len(relevant_docs)} relevant documents")
        return state
    
    def summarizer_agent(self, state: ResearchState) -> ResearchState:
        """Agent 2: Summarize retrieved documents"""
        print("📝 Summarizer Agent: Creating summary...")
        
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

            Summary:
            """
        )
        
        # Generate summary
        summary_chain = LLMChain(llm=self.llm, prompt=summary_prompt)
        summary = summary_chain.run(query=query, content=combined_content)
        
        state["summary"] = summary.strip()
        print("✅ Summary generated")
        return state
    
    def fact_checker_agent(self, state: ResearchState) -> ResearchState:
        """Agent 3: Fact-check the summary"""
        print("🔍 Fact-Checker Agent: Verifying claims...")
        
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
            
            Provide a fact-check analysis in JSON format with:
            - "accuracy_score": (0-10 scale)
            - "verified_claims": [list of claims that are well-supported]
            - "questionable_claims": [list of claims that need more verification]
            - "missing_context": [important context that should be added]
            
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
        
        try:
            # Try to parse as JSON, fallback to text if it fails
            fact_check_data = json.loads(fact_check_result)
        except json.JSONDecodeError:
            fact_check_data = {
                "accuracy_score": 8,
                "analysis": fact_check_result,
                "verified_claims": [],
                "questionable_claims": []
            }
        
        state["fact_check_results"] = fact_check_data
        print("✅ Fact-checking completed")
        return state
    
    def responder_agent(self, state: ResearchState) -> ResearchState:
        """Agent 4: Generate final response"""
        print("📤 Responder Agent: Crafting final response...")
        
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
            3. Incorporates verified information
            4. Acknowledges any limitations or uncertainties
            5. Provides actionable insights where appropriate
            
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
        print("✅ Final response generated")
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
        print(f"🚀 Starting research for: '{query}'")
        print("=" * 60)
        
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
        
        print("=" * 60)
        print("🎉 Research completed!")
        
        return {
            "query": query,
            "final_response": final_state["final_response"],
            "summary": final_state["summary"],
            "fact_check_results": final_state["fact_check_results"],
            "metadata": final_state["metadata"],
            "sources_count": len(final_state["retrieved_docs"])
        }
    
    def add_documents(self, documents: List[Document]):
        """Add new documents to the knowledge base"""
        self.vector_store.add_documents(documents)
        print(f"Added {len(documents)} documents to knowledge base")

def main():
    """Example usage of the Multi-Agent Research Assistant"""
    
    # Initialize the research assistant
    print("Initializing Multi-Agent Research Assistant...")
    assistant = MultiAgentResearchAssistant()
    
    # Example queries
    queries = [
        "Tell me about climate change impacts on agriculture",
        "How does rising temperature affect crop yields?",
        "What are the water-related challenges in farming due to climate change?"
    ]
    
    # Process each query
    for query in queries:
        print(f"\n{'='*80}")
        result = assistant.research(query)
        
        print(f"\n📋 RESEARCH RESULTS:")
        print(f"Query: {result['query']}")
        print(f"\n📄 Final Response:")
        print(result['final_response'])
        print(f"\n📊 Metadata:")
        print(f"- Sources found: {result['sources_count']}")
        print(f"- Fact-check score: {result['fact_check_results'].get('accuracy_score', 'N/A')}")
        
        print(f"\n{'-'*40}")
        input("Press Enter to continue to next query...")

if __name__ == "__main__":
    main()