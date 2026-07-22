# SEC-RAG: Filing Analyzer

Ask questions about SEC 10-K filings in plain English, get cited answers.

## The RAG Pipeline
1. `config.py` - settings: models, chunk sizes, retrieval params
2. `ingest/download.py` - download 10-K filings from SEC EDGAR
3. `ingest/chunk.py` - split filings into overlapping chunks
4. `ingest/embed.py` - embed chunks, store in ChromaDB vector database
5. `retrieval/search.py` - hybrid retrieval (dense embeddings + BM25 keyword)
6. `retrieval/rerank.py` - re-rank retrieved chunks with a cross-encoder
7. `generate/answer.py` - feed question + chunks to LLM, generate cited answer
8. `app.py` - tie it all together, interactive Q&A

## Key Concepts
- Chunking strategies (fixed-size, recursive, overlap)
- Dense retrieval (semantic embeddings + cosine similarity)
- Sparse retrieval (BM25 keyword matching)
- Hybrid retrieval (Reciprocal Rank Fusion)
- Cross-encoder re-ranking
- Grounded generation with citations
- "Lost in the middle" problem

## No Training Required
Unlike Projects 1-2, RAG uses models as-is. The intelligence comes from
retrieving the RIGHT context, not from updating model weights.
