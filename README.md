# 🤖 Interactive Multi-Agent Research Assistant

A local, privacy-preserving AI research assistant powered by **LangChain**, **LangGraph**, **Ollama**, and **Chroma**. This assistant processes natural language queries through a multi-agent architecture to retrieve, summarize, fact-check, and respond using curated knowledge.

---

## 🧠 Features

- 🔍 **Retriever Agent** — Finds the most relevant documents from a vector-based knowledge base.
- 📝 **Summarizer Agent** — Synthesizes retrieved content into a concise summary.
- ✅ **Fact Checker Agent** — Verifies the accuracy of the summary.
- 📤 **Responder Agent** — Generates a clear and final answer to the original query.
- 💬 **Interactive CLI** — Ask your own questions and get complete, structured answers.

---

## 🛠️ Tech Stack

| Component         | Technology                                    |
|------------------|-----------------------------------------------|
| Language Model    | [Ollama](https://ollama.com/) (`llama3`)      |
| Vector Store      | [ChromaDB](https://www.trychroma.com/)        |
| Embeddings        | HuggingFace: `all-MiniLM-L6-v2`               |
| Agent Framework   | [LangChain](https://www.langchain.com/)       |
| Workflow Engine   | [LangGraph](https://github.com/langchain-ai/langgraph) |
| Prompt Templates  | LangChain `PromptTemplate`, `LLMChain`        |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/multi-agent-research-assistant.git
cd multi-agent-research-assistant
