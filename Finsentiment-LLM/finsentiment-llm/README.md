# FinSentiment-LLM
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
