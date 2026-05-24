#!/usr/bin/env python3
"""
FinSentiment-LLM Project Scaffold
===================================
Fine-tune Mistral 7B on financial sentiment data using QLoRA.

Creates the directory structure and starter files.
YOU write all the code (with guidance).

Usage:
    python setup_finsentiment.py
    cd finsentiment-llm
    pip install -r requirements.txt
"""

import os

PROJECT = "finsentiment-llm"

FILES = {

    "requirements.txt": """torch>=2.0
transformers>=4.40
peft>=0.10
trl>=0.8
datasets
bitsandbytes
accelerate
scikit-learn
wandb
""",

    ".gitignore": """checkpoints/
logs/
__pycache__/
*.pyc
.venv/
wandb/
""",

    "README.md": """# FinSentiment-LLM
Fine-tune Mistral 7B on financial sentiment using QLoRA.

## Build Order
1. `config.py` - model name, LoRA hyperparameters, training args
2. `data/prepare.py` - download Financial PhraseBank, format for SFT
3. `data/explore.py` - inspect dataset, check class balance, view examples
4. `model.py` - load base model in 4-bit, attach LoRA adapters
5. `train.py` - SFT training loop with TRL's SFTTrainer
6. `evaluate.py` - compare base model vs fine-tuned on test set
7. `inference.py` - interactive sentiment analysis on custom text

## Concepts Covered
- 4-bit quantization (NormalFloat4, double quantization)
- LoRA: rank, alpha, target modules, dropout
- QLoRA: quantized base + higher-precision adapters
- SFT data formatting for instruction tuning
- Gradient checkpointing, packing, cosine LR schedule
- F1, precision, recall evaluation
- Adapter-only saving (27 MB vs 14 GB full model)

## Hardware Requirements
- GPU with 16GB+ VRAM (e.g., A10G, T4, RTX 3090)
- ~15-20 AWS credits on g5.xlarge spot instance
""",

    "config.py": '''"""
config.py - All hyperparameters for fine-tuning.

Three sections to define:

1. MODEL CONFIG
   - Which base model to use (Mistral 7B Instruct)
   - Quantization settings (4-bit NormalFloat, double quantization)

2. LoRA CONFIG
   - r (rank): controls adapter capacity. 8-64, typically 16
   - lora_alpha: scaling factor, typically 2x rank
   - target_modules: which layers to adapt (Q, K, V, O, gate, up, down)
   - dropout: regularization on LoRA weights

3. TRAINING CONFIG
   - learning_rate: 2e-4 is standard for LoRA
   - batch_size: limited by GPU memory
   - epochs: 3 is typical for SFT
   - max_seq_length: 512 tokens is enough for sentiment

Key insight: LoRA only updates ~0.06% of parameters.
The rest stay frozen in 4-bit precision.
"""

# TODO: Define model_config, lora_config, training_config
''',

    "data/__init__.py": "",

    "data/prepare.py": '''"""
data/prepare.py - Download and format financial sentiment data.

Dataset: Financial PhraseBank (takala/financial_phrasebank)
- 4,840 financial sentences annotated by experts
- Labels: positive (0), neutral (1), negative (2)
- Subset "sentences_allagree" = only sentences where ALL annotators agreed

YOUR JOB:
1. Load the dataset from HuggingFace
2. Format each example as an instruction-response pair:

   <s>[INST] Analyze the sentiment of this financial text.
   Classify as: positive, negative, or neutral.

   Text: "The company reported strong Q4 earnings"
   [/INST]
   Sentiment: positive</s>

3. Split into train/test (90/10)
4. Save as HuggingFace Dataset object

WHY THIS FORMAT?
Mistral uses [INST]...[/INST] tags for instruction tuning.
The model learns: see [INST] → follow the instruction → generate response.
"""

# TODO: Load dataset, format for SFT, split, save
''',

    "data/explore.py": '''"""
data/explore.py - Explore the dataset before training.

Good practice: ALWAYS look at your data before training.

Implement:
1. Print class distribution (how many positive/neutral/negative?)
2. Print 5 example sentences from each class
3. Print sentence length statistics (min, max, mean, median)
4. Print token count statistics (how many tokens per example?)
5. Check for duplicates or data quality issues

This helps you:
- Spot class imbalance (might need weighted loss or oversampling)
- Understand what "positive" vs "neutral" looks like in financial text
- Set max_seq_length appropriately in config
"""

# TODO: Load prepared dataset, print statistics and examples
''',

    "model.py": '''"""
model.py - Load base model with QLoRA.

Two main things happen here:

1. QUANTIZATION (BitsAndBytesConfig):
   Load Mistral 7B in 4-bit precision using NormalFloat4 format.
   14 GB model → ~3.5 GB in memory.

   Key settings:
   - load_in_4bit=True
   - bnb_4bit_quant_type="nf4" (NormalFloat4, from QLoRA paper)
   - bnb_4bit_compute_dtype=torch.bfloat16 (compute in higher precision)
   - bnb_4bit_use_double_quant=True (quantize the quantization constants too)

2. LoRA ADAPTERS (PeftConfig):
   Attach small trainable matrices to specific layers.
   Only these adapters get updated during training.

   target_modules for Mistral:
   - Attention: q_proj, k_proj, v_proj, o_proj
   - FFN (SwiGLU): gate_proj, up_proj, down_proj

   After LoRA: ~6.8M trainable out of ~7.2B total (0.09%)

Implement:
- load_model(): returns quantized model with LoRA adapters + tokenizer
- Print trainable vs total parameters to verify
"""

# TODO: Build load_model() function
''',

    "train.py": '''"""
train.py - Fine-tune with SFTTrainer.

TRL's SFTTrainer wraps HuggingFace Trainer with SFT-specific features:
- Packing: concatenates short examples to fill max_seq_length (efficient)
- Dataset formatting: handles the text field automatically
- LoRA-aware: works seamlessly with PEFT models

Training args that matter:
- per_device_train_batch_size: 4 (limited by GPU memory)
- gradient_accumulation_steps: 4 (effective batch = 16)
- learning_rate: 2e-4 (standard for LoRA, higher than full fine-tuning)
- lr_scheduler_type: "cosine" (same idea as Project 1)
- bf16: True (compute in bfloat16 for speed)
- gradient_checkpointing: True (saves VRAM, recomputes activations)

After training:
- Save ONLY the LoRA adapter weights (~27 MB)
- NOT the full model (14 GB) — the base model is unchanged

Run: python train.py
"""

# TODO: Build training pipeline using SFTTrainer
''',

    "evaluate.py": '''"""
evaluate.py - Compare base model vs fine-tuned model.

This is the most important file for proving your model works.

Implement:
1. Load the fine-tuned model (base + LoRA adapter)
2. Load the base model WITHOUT the adapter
3. Run both on the same test set
4. For each test example:
   - Feed the prompt to the model
   - Extract the predicted sentiment from the generated text
   - Compare with the true label
5. Print classification_report (precision, recall, F1 per class)
6. Print confusion matrix
7. Show side-by-side examples where they differ

Expected results:
- Base model: ~55-65% accuracy (it can do sentiment but isn't specialized)
- Fine-tuned: ~85-92% accuracy (trained specifically on financial sentiment)

That gap is what LoRA fine-tuning gives you.
"""

# TODO: Build evaluation pipeline
''',

    "inference.py": '''"""
inference.py - Interactive financial sentiment analysis.

Load your fine-tuned model and analyze any financial text.

Implement:
1. Load model (base + LoRA adapter)
2. Accept text input from user (argparse or interactive loop)
3. Format as instruction prompt
4. Generate response
5. Parse and display sentiment + reasoning

Usage:
  python inference.py --text "Apple reported record quarterly revenue of $94.8B"
  python inference.py --interactive

This is the "demo" file — shows your model actually works on new text.
"""

# TODO: Build inference function with argparse
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
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content.lstrip("\n"))
        print(f"  {filepath}")

    print(f"\nDone! {len(FILES)} files created.\n")
    print(f"  cd {PROJECT}")
    print(f"  pip install -r requirements.txt")
    print(f"  # Start with config.py, then follow the build order in README.md")


if __name__ == "__main__":
    main()