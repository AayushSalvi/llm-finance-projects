"""
config.py - All settings for the RAG pipeline.

Define:
1. EMBEDDING MODEL - which model turns text into vectors
   (e.g., "sentence-transformers/all-MiniLM-L6-v2")

2. RERANKER MODEL - cross-encoder for re-ranking
   (e.g., "cross-encoder/ms-marco-MiniLM-L-6-v2")

3. LLM - which model generates the final answer
   (e.g., "mistralai/Mistral-7B-Instruct-v0.3")

4. CHUNKING - chunk_size (512), chunk_overlap (64)

5. RETRIEVAL - top_k for initial retrieval (20), top_n after rerank (3-5)

6. PATHS - where filings and vector DB are stored
"""

# TODO: Define CONFIG dict with all settings
