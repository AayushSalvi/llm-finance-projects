#!/usr/bin/env python3
"""
FinGPT-Nano Project Scaffold
=============================
Creates the directory structure and empty starter files.
YOU write all the code.

Usage:
    python setup_fingpt_nano.py
    cd fingpt-nano
    pip install -r requirements.txt
"""

import os

PROJECT = "fingpt-nano"

FILES = {

    "requirements.txt": """torch>=2.0
numpy
tiktoken
datasets
wandb
matplotlib
""",

    ".gitignore": """data/*.bin
data/corpus.txt
checkpoints/
logs/
__pycache__/
*.pyc
.venv/
""",

    "README.md": """# FinGPT-Nano
A ~20M parameter GPT trained from scratch on financial text.

## Build Order
1. `config.py` - define all hyperparameters
2. `tokenizer.py` - wrap tiktoken (GPT-2 BPE)
3. `data/download.py` - fetch financial news
4. `data/prepare.py` - tokenize corpus into train.bin + val.bin
5. `model.py` - build transformer from scratch (THE core file)
6. `train.py` - training loop
7. `generate.py` - text generation with sampling
8. `evaluate.py` - perplexity measurement
""",

    "config.py": '''"""
config.py - All hyperparameters in one place.

Define: vocab_size, block_size (context window), n_layer, n_head,
n_embd, dropout, learning_rate, batch_size, max_steps, device, paths.

Every other file imports from here.
"""

# TODO: Create a CONFIG dict with all hyperparameters
''',

    "tokenizer.py": '''"""
tokenizer.py - Tokenizer wrapper around tiktoken (GPT-2 BPE).

Implement: encode(text) -> list[int], decode(ids) -> str
Test: tokenize some financial sentences to see how BPE splits words.

Library: tiktoken
"""

# TODO: Build Tokenizer class with encode/decode methods
''',

    "data/__init__.py": "",

    "data/download.py": '''"""
data/download.py - Download financial text data.

Fetch financial news from HuggingFace datasets.
Save as a single text file: data/corpus.txt

Library: datasets (HuggingFace)
Dataset suggestion: "ashraq/financial-news-articles"
"""

# TODO: Download dataset, extract text, save to corpus.txt
''',

    "data/prepare.py": '''"""
data/prepare.py - Tokenize corpus into binary training data.

Read data/corpus.txt, tokenize with your tokenizer,
split 90/10, save as data/train.bin and data/val.bin (uint16 numpy arrays).
"""

# TODO: Load corpus, tokenize, split, save as .bin files
''',

    "model.py": '''"""
model.py - The Transformer Architecture (build from scratch).

Build bottom-up, test each piece before moving on:

  1. CausalSelfAttention
     - Q, K, V projections
     - Scaled dot-product: softmax(QK^T / sqrt(d_k)) * V
     - Causal mask (lower triangular)
     - Multi-head: split into h heads, concat, project

  2. FeedForward
     - Linear(d -> 4d) -> GELU -> Linear(4d -> d)

  3. TransformerBlock
     - Pre-Norm: LayerNorm BEFORE attention and FFN
     - Residual connections around both

  4. FinGPTNano (full model)
     - Token embedding + position embedding
     - N transformer blocks
     - Final LayerNorm -> Linear head -> logits
     - Weight tying: embedding weights = output head weights

Test each class: pass random tensor, verify output shape matches input.
"""

# TODO: Build all 4 classes
''',

    "train.py": '''"""
train.py - Training loop.

Implement:
  - get_batch(): load random sequences from .bin files
  - get_lr(step): linear warmup + cosine decay schedule
  - estimate_loss(): average loss over multiple val batches
  - train(): the main loop
    forward pass -> cross-entropy loss -> backward -> clip grads -> AdamW step

Key details:
  - AdamW (not Adam) with weight_decay=0.1, betas=(0.9, 0.95)
  - Gradient clipping at 1.0
  - Save checkpoints periodically
  - Log loss every N steps
"""

# TODO: Build the training loop
''',

    "generate.py": '''"""
generate.py - Text generation with sampling strategies.

Implement autoregressive generation:
  1. Encode prompt -> token IDs
  2. Forward pass -> get logits for last position
  3. Apply sampling strategy -> pick next token
  4. Append token, repeat

Sampling strategies to implement:
  - Temperature scaling (low=focused, high=creative)
  - Top-k filtering (only sample from k most likely)
  - Top-p / nucleus (sample from tokens covering p% probability)

Add argparse for: --prompt, --temperature, --top_k, --top_p, --max_tokens
"""

# TODO: Build generate() function with all sampling strategies
''',

    "evaluate.py": '''"""
evaluate.py - Evaluate model quality.

Implement:
  - Perplexity = exp(average cross-entropy loss)
    PPL=1 is perfect, PPL=50257 is random
  - Plot training loss curve from saved logs
  - Compare train vs val loss (check for overfitting)

Library: matplotlib for plotting
"""

# TODO: Build evaluation and plotting
''',
}

DIRS = ["data", "checkpoints", "logs"]


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
    print(f"  # Start with config.py, then follow the build order in README.md")


if __name__ == "__main__":
    main()