# Mini RAG System – LangChain

A simple Retrieval-Augmented Generation (RAG) system built with LangChain, Ollama, and ChromaDB.

## Features

- PDF document loading and text splitting.
- Embedding generation using `nomic-embed-text`.
- Vector storage and similarity search with ChromaDB.
- Question answering using `llama3.2:3b`.
- Basic security guardrails against jailbreak attempts and off-topic queries.
- Source document retrieval.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/)
- Required models:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

## Installation

```bash
pip install -U langchain langchain-community langchain-chroma langchain-ollama langchain-text-splitters pypdf
```

## Project Structure

```text
mini-rag/
├── documents/       # PDF documents
├── chroma_db/       # Persisted vector database
├── config.py        # Document loading and indexing
└── main.py          # RAG pipeline and guardrails
```

## Usage

**1. Add your PDF documents** to the `documents/` folder.

**2. Build the vector database:**

```bash
python config.py
```

**3. Run the RAG assistant:**

```bash
python main.py
```

**4. Ask questions** about your documents or type `exit` to quit.

## Security

The system implements basic guardrails to:
- Detect predefined jailbreak expressions.
- Reject queries with insufficient document relevance.

*Note: These guardrails provide basic protection and are not a comprehensive security solution.*

## Technologies

Python · LangChain · Ollama · ChromaDB · RAG
