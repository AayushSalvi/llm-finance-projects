"""
retrieval/search.py - Hybrid retrieval (dense + sparse).

The heart of RAG: given a question, find the most relevant chunks.

YOUR JOB - implement two retrieval methods and combine them:

1. DENSE (semantic):
   - Embed the query with the same model used for chunks
   - Query ChromaDB for the nearest chunks by cosine similarity
   - Good at: understanding meaning, paraphrases, synonyms

2. SPARSE (keyword):
   - Use BM25 to find chunks with matching keywords
   - Good at: exact terms, company names, specific numbers

3. HYBRID (combine both with Reciprocal Rank Fusion):
   - Each method ranks chunks; RRF merges the rankings
   - score = sum over methods of 1/(k + rank)
   - Best of both worlds

Library: sentence-transformers, chromadb, rank-bm25

Why hybrid? Dense retrieval might miss an exact ticker symbol;
sparse retrieval might miss a paraphrase. Together they're robust.
"""

# TODO: Implement dense_search, sparse_search, hybrid_search
