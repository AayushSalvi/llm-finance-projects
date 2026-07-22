"""
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
