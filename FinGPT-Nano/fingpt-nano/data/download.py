"""
data/download.py - Download financial text data.

Fetch financial news from HuggingFace datasets.
Save as a single text file: data/corpus.txt

Library: datasets (HuggingFace)
Dataset suggestion: "ashraq/financial-news-articles"
"""

# TODO: Download dataset, extract text, save to corpus.txt
"""
data/download.py - Download financial text data.

Downloads financial news articles from HuggingFace and saves
as a single text file for tokenization.

Run: python data/download.py
"""

import os
from datasets import load_dataset


def download_financial_data(output_path="data/corpus.txt"):
    """Download financial news and save as raw text."""

    print("Downloading financial news from HuggingFace...")
    print("(First run downloads ~50MB, then it's cached)\n")

    # Load dataset — this downloads and caches automatically
    # "ashraq/financial-news-articles" has ~16k financial news articles
    try:
        ds = load_dataset("ashraq/financial-news-articles", split="train")
        text_column = "content"  # This dataset uses "content" for article text
        print(f"Loaded {len(ds):,} articles")
        print(f"Columns: {ds.column_names}")
        print(f"First entry: {ds[0]}")
    except Exception as e:
        # Backup dataset if the primary one is unavailable
        print(f"Primary dataset failed: {e}")
        print("Trying backup dataset...")
        ds = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
        text_column = "text"
        print(f"Loaded {len(ds):,} entries from backup")
        print(f"Columns: {ds.column_names}")
        print(f"First entry: {ds[0]}")

    # Extract and filter text
    # Skip entries that are empty, None, or too short to be useful
    texts = []
    skipped = 0
    for item in ds:
        text = item.get("textlets ", "")
        if text and len(text.strip()) > 100:  # At least ~20 words
            texts.append(text.strip())
        else:
            skipped += 1

    print(f"Kept: {len(texts):,} articles")
    print(f"Skipped: {skipped:,} (too short or empty)")

    # Save to a single text file
    # Double newline between articles so they don't bleed together
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n\n")

    # Print stats
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    total_chars = sum(len(t) for t in texts)

    print(f"\nSaved to: {output_path}")
    print(f"File size: {file_size:.1f} MB")
    print(f"Total characters: {total_chars:,}")



if __name__ == "__main__":
    download_financial_data()