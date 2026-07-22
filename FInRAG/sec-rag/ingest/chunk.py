"""
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
