"""
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
