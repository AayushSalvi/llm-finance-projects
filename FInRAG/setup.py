#!/usr/bin/env python3
"""
SEC-RAG Project Scaffold
=========================
Build a Retrieval-Augmented Generation system over SEC 10-K filings.
Ask questions in plain English, get cited answers from real filings.

Creates the directory structure and starter files.
YOU write the code (with guidance).

Usage:
    python setup_sec_rag.py
    cd sec-rag
    pip install -r requirements.txt
"""

import os

PROJECT = "sec-rag"

FILES = {

    "requirements.txt": """torch>=2.0
transformers>=4.40
sentence-transformers
chromadb
rank-bm25
sec-edgar-downloader
beautifulsoup4
langchain
langchain-text-splitters
accelerate
bitsandbytes
""",

    ".gitignore": """sec_filings/
sec_vectordb/
*.bin
__pycache__/
*.pyc
.venv/
logs/
""",

    "README.md": """# SEC-RAG: Filing Analyzer

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
""",

    "config.py": '''"""
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
''',

    "ingest/__init__.py": "",

    "ingest/download.py": '''"""
ingest/download.py - Download SEC 10-K filings from EDGAR.

SEC EDGAR provides free access to all public company filings.

YOUR JOB:
1. Use sec-edgar-downloader to fetch 10-K filings for a list of companies
   (e.g., AAPL, MSFT, GOOGL, JPM)
2. The downloader saves raw HTML/text files
3. Parse the HTML with BeautifulSoup to extract clean text
4. Save each filing as clean text for the next step

Library: sec-edgar-downloader, beautifulsoup4

Note: SEC requires a user-agent (your name + email) for API access.
"""

# TODO: Download filings, parse HTML, save clean text
''',

    "ingest/chunk.py": '''"""
ingest/chunk.py - Split filings into chunks.

Why chunk? A 10-K filing is 100+ pages. You can't feed the whole thing
to an LLM. And you want to retrieve only the RELEVANT sections.

YOUR JOB:
1. Load the clean filing text
2. Split into chunks using RecursiveCharacterTextSplitter
   - chunk_size: 512 characters (~128 tokens)
   - chunk_overlap: 64 characters (preserves context across boundaries)
3. Attach metadata to each chunk (which company, which filing)
4. Return list of chunks ready for embedding

Library: langchain-text-splitters (RecursiveCharacterTextSplitter)

Chunking strategy matters:
- Too small: loses context, retrieval returns fragments
- Too large: retrieval is imprecise, wastes LLM context
- Overlap: prevents splitting a sentence/idea across two chunks
"""

# TODO: Load filings, split into chunks with metadata
''',

    "ingest/embed.py": '''"""
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
''',

    "retrieval/__init__.py": "",

    "retrieval/search.py": '''"""
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
''',

    "retrieval/rerank.py": '''"""
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
''',

    "generate/__init__.py": "",

    "generate/answer.py": '''"""
generate/answer.py - Generate a cited answer from retrieved chunks.

The final step: feed the question + best chunks to an LLM.

YOUR JOB:
1. Load an LLM (Mistral 7B, 4-bit quantized like Project 2)
2. Build a prompt that includes:
   - The retrieved chunks as context (with source labels)
   - The user's question
   - Instructions to answer ONLY from the context and cite sources
3. Generate the answer
4. Return answer + which sources were used

Library: transformers, bitsandbytes

Prompt design matters:
- Tell the model to say "not found in context" if the answer isn't there
  (reduces hallucination)
- Put the most relevant chunks first and last (Lost in the Middle problem)
- Ask for citations so the answer is verifiable
"""

# TODO: Build prompt, generate grounded answer with citations
''',

    "app.py": '''"""
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
''',
}

DIRS = ["ingest", "retrieval", "generate", "sec_filings", "sec_vectordb", "logs"]


def main():
    print(f"Creating project: {PROJECT}/\n")
    os.makedirs(PROJECT, exist_ok=True)
    for d in DIRS:
        os.makedirs(os.path.join(PROJECT, d), exist_ok=True)

    for filepath, content in FILES.items():
        full_path = os.path.join(PROJECT, filepath)
        os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content.lstrip("\n"))
        print(f"  {filepath}")

    print(f"\nDone! {len(FILES)} files created.\n")
    print(f"  cd {PROJECT}")
    print(f"  pip install -r requirements.txt")
    print(f"  # Start with config.py, then follow the pipeline order in README.md")


if __name__ == "__main__":
    main()