"""
app.py - Tie the full RAG pipeline together.

The complete flow:
  question -> hybrid_search -> rerank -> generate_answer -> cited response

YOUR JOB:
1. Load all components (embedder, vector DB, reranker, LLM)
2. Accept a question (argparse or interactive loop)
3. Run the pipeline:
   a. hybrid_search(question) -> ~20 candidate chunks
   b. rerank(question, candidates) -> top 3-5 chunks
   c. generate_answer(question, top_chunks) -> cited answer
4. Display the answer and its sources

Usage:
  python app.py --question "What are Apple's main risk factors?"
  python app.py --interactive
"""

# TODO: Wire the full pipeline together
