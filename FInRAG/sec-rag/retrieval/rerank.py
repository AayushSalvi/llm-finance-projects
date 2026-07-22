"""
retrieval/rerank.py - Re-rank retrieved chunks with a cross-encoder.

Initial retrieval returns ~20 candidates fast but imprecisely.
Re-ranking scores them more carefully, keeps the best 3-5.

YOUR JOB:
1. Load a cross-encoder model (e.g., ms-marco-MiniLM)
2. For each (query, chunk) pair, compute a relevance score
3. Sort by score, return top_n chunks

Library: sentence-transformers (CrossEncoder)

Bi-encoder vs Cross-encoder:
- Bi-encoder (retrieval): encodes query and chunk SEPARATELY, fast,
  used for the initial search over thousands of chunks
- Cross-encoder (rerank): encodes query and chunk TOGETHER, slow but
  accurate, used to re-rank the ~20 candidates from retrieval

You use both: bi-encoder to narrow 10,000 -> 20, cross-encoder to
narrow 20 -> 3.
"""

# TODO: Implement rerank() using a cross-encoder
