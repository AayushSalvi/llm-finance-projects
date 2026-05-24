"""
data/prepare.py - Tokenize corpus into binary training data.

Read data/corpus.txt, tokenize with your tokenizer,
split 90/10, save as data/train.bin and data/val.bin (uint16 numpy arrays).
"""

# TODO: Load corpus, tokenize, split, save as .bin files
import os 
import sys 
import numpy as np 

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer import Tokenizer

def prepare_data(
    input_path="data/corpus.txt",
    train_path="data/train.bin",
    val_path="data/val.bin",
    val_fraction=0.1):

    if not os.path.exists(input_path):
        print(f"Error input path not found")
        print("Run 'python data/download.py' first.")
        return
    
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    print("Tokenizing")
    tok = Tokenizer()
    tokens = tok.encode(text)

    data = np.array(tokens,dtype=np.uint16)

    n_val = int(len(data)*val_fraction)
    n_train = len(data) - n_val

    train_data = data[:n_train]
    val_data = data[n_train:]

    train_data.tofile(train_path)
    val_data.tofile(val_path)

    print(f"\n{'=' * 50}")
    print(f"Train: {n_train:>12,} tokens  ({100 * (1 - val_fraction):.0f}%)")
    print(f"Val:   {n_val:>12,} tokens  ({100 * val_fraction:.0f}%)")
    print(f"Total: {len(data):>12,} tokens")
    print(f"{'=' * 50}")
    print(f"Saved: {train_path} ({os.path.getsize(train_path) / 1e6:.1f} MB)")
    print(f"Saved: {val_path} ({os.path.getsize(val_path) / 1e6:.1f} MB)")

    sample_ids = tokens[:15]
    for i, tid in enumerate(sample_ids):
        token_str = tok.decode([tid])
        print(f"  [{i:2d}] ID={tid:>6d} → \"{token_str}\"")

    reloaded = np.fromfile(train_path, dtype=np.uint16)
    assert len(reloaded) == n_train, "Reload size mismatch!"
    assert np.array_equal(reloaded[:100], train_data[:100]), "Data mismatch!"
    print("\nVerification passed — binary files are correct.")


if __name__ == "__main__":
    prepare_data()
