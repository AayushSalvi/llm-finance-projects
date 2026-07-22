"""
ingest/embed.py - Embed chunks and store in vector database.

YOUR JOB:
1. Load the sentence-transformer embedding model
2. For each chunk, compute its embedding (a vector, e.g., 384-dim)
3. Store chunks + embeddings + metadata in ChromaDB
4. ChromaDB handles the similarity search index automatically

Library: sentence-transformers, chromadb

Concept: An embedding is a vector that captures the MEANING of text.
Similar meanings = nearby vectors. This is what enables semantic search:
"What are the risks?" can match a chunk about "risk factors" even
without exact keyword overlap.

Batch the embedding (encode 100 chunks at a time) for efficiency.
"""

# TODO: Embed chunks, store in ChromaDB
