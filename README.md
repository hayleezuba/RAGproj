# Career Advisor RAG System

A Retrieval-Augmented Generation (RAG) system that uses O*NET occupational data to provide grounded career advice and information. The current iteration only contains information from structured O*Net csvs about Robotics Engineers. Future iterations will include unstructured documents covering more careers.

## Features

- Document ingestion from structured career datasets
- Text chunking with overlap
- Semantic embeddings using HuggingFace
- Vector search with ChromaDB
- Context-aware responses using local LLMs through Ollama

## Tech Stack

- Python
- ChromaDB
- HuggingFace Embeddings
- Ollama
- LangChain Text Splitters

## Example Query

"What skills do Robotics Engineers need?"

## Future Improvements

- FastAPI backend
- Docker containerization
- PDF and Web site ingestion
- Source citations
- Web interface
