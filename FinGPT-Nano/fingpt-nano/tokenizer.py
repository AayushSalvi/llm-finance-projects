"""
tokenizer.py - Tokenizer wrapper around tiktoken (GPT-2 BPE).

Implement: encode(text) -> list[int], decode(ids) -> str
Test: tokenize some financial sentences to see how BPE splits words.

Library: tiktoken
"""

"""Build Tokenizer class with encode/decode methods"""
import tiktoken

class Tokenizer:
    """Wrapper around GPT-2's BPE tokenizer"""

    def __init__(self):
        self.encoder = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.encoder.n_vocab #50257

    def encode(self,text:str) -> list[int]:
        #convert text to token ids
        return self.encoder.encode(text)
    
    def decode(self, token_ids:list[int]) ->str:
        #convert token IDs back to text
        return self.encoder.decode(token_ids)
    
    def tokenize(self, text:str)-> list[str]:
        ids = self.encode(text)

        return [self.encoder.decode([id]) for id in ids]
    

if __name__ == "__main__":
    tok = Tokenizer()
    print("=" * 60)
    print("TOKENIZATION DEMO")
    print(f"Vocabulary size: {tok.vocab_size:,} tokens")
    print("=" * 60)

    examples = [
        "The company reported quarterly revenue of $5.2 billion.",
        "EBITDA margins expanded by 200 basis points year-over-year.",
        "The Federal Reserve raised interest rates by 25bps.",
        "AAPL stock surged 8% after strong iPhone sales data.",
        "Net income attributable to common shareholders was $3.4B.",
    ]

    for text in examples:
        ids = tok.encode(text)
        tokens = tok.tokenize(text)

        print(f"\nText:    \"{text}\"")
        print(f"Tokens:  {tokens}")
        print(f"IDs:     {ids}")
        print(f"Count:   {len(ids)} tokens")

    # Demonstrate encode → decode roundtrip
    print("\n" + "=" * 60)
    print("ROUNDTRIP TEST")
    print("=" * 60)
    test = "Revenue grew 15% to $42.3 billion in Q4 2024."
    ids = tok.encode(test)
    recovered = tok.decode(ids)
    print(f"Original:  \"{test}\"")
    print(f"Recovered: \"{recovered}\"")
    print(f"Match: {test == recovered}")
