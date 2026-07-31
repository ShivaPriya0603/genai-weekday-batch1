
**Week 4 — Capstone Task: RAG-as-a-Tool Chat Assistant

Task Code: M1Task1

Module: 1

Maps to: 4.2 Advanced Retrieval (Hybrid Search + Metadata Filtering), 3.1 OpenAI API Deep Dive (Function Calling)

Format: Individual hands-on task, take-home

Submission: Completed .ipynb notebook

Stack: LangChain + Ollama (FAISS, BM25Retriever, EnsembleRetriever) for retrieval, OpenAI API (gpt-4o-mini) for the tool-calling assistant

Objective

Combine Week 3 (function calling) and Week 4 (hybrid RAG) into one assistant: your hybrid RAG pipeline becomes a single callable tool, and a tool-calling chat loop invokes it whenever the user asks something about the documents — answering directly, without the tool, when they don't.

Part 1 — Hybrid RAG, wrapped as a tool

* Load any two PDFs, chunk them, and build a FAISS retriever and a BM25Retriever, combined via EnsembleRetriever
* Wrap retrieval + generation into one Python function, e.g. search_documents(query: str) -> str, that returns an answer with page-level source citations
* Test this function on its own with one query and show the output, before it's wired into the assistant

Part 2 — Tool-calling assistant

* Define search_documents as a tool with a JSON schema
* Build a chat loop (simulate with a list of at least 5 inputs, ending in "quit") that keeps real conversation history
* Include a mix of inputs: some that should trigger the tool (ask about the documents), and some that shouldn't (general questions, small talk) — the assistant should call the tool only when the question is actually about the documents
* For each turn, show: user input → whether the tool was called → tool output (if called) → final response

Notebook structure requirements

* Two sections: "1. RAG Tool" and "2. Tool-Calling Assistant", each with a short markdown cell explaining what it does
* Tool schema, retrievers, and prompts as named variables — not inline
* Conversation history must persist across turns in Part 2
* Fully executed notebook, outputs visible

Constraints

* Part 1: LangChain + Ollama only
* Part 2: raw OpenAI SDK, no agent framework
* Minimum of 6 chat conversations each 2 on llm memory, pdf1 and pdf2 from vector database.

**
