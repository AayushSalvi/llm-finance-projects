# FinGPT-Nano
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
