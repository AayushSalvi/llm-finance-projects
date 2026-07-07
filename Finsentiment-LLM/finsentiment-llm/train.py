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

from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig

from config import TRAINING_CONFIG, MODEL_CONFIG, LORA_CONFIG
from model import load_model
from log_experiment import log_experiment
import time


def train():
    """Fine Tunning the model on financial sentiment data"""
    print("loading model...")

    model, tokenizer = load_model()

    print("Loading Dataset")
    dataset = load_from_disk("data/financial_sentiment")
    print(f"Train: {len(dataset['train']):,} examples")
    print(f"Test:  {len(dataset['test']):,} examples")

    training_args = SFTConfig(
        output_dir=TRAINING_CONFIG["output_dir"],
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        lr_scheduler_type=TRAINING_CONFIG["lr_scheduler_type"],
        bf16=TRAINING_CONFIG["bf16"],
        gradient_checkpointing=TRAINING_CONFIG["gradient_checkpointing"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        eval_strategy=TRAINING_CONFIG["eval_strategy"],
        save_strategy=TRAINING_CONFIG["save_strategy"],
        load_best_model_at_end=TRAINING_CONFIG["load_best_model_at_end"],
        dataset_text_field="text",
        packing=TRAINING_CONFIG["packing"],
        max_length=MODEL_CONFIG["max_seq_length"],
        loss_type="nll",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        args=training_args,
    )

    # ── Train! ──
    # Under the hood, this does exactly what Project 1 loop did:
    #   for each batch:
    #     logits, loss = model(input_ids, labels)
    #     loss.backward()
    #     clip_grad_norm_()
    #     optimizer.step()
    #     scheduler.step()
    #
    # But it also handles:
    #   - Gradient accumulation (effective batch = 4 × 4 = 16)
    #   - Mixed precision (BF16 forward, FP32 gradients)
    #   - Gradient checkpointing (recompute activations to save memory)
    #   - Evaluation at end of each epoch
    #   - Saving best model based on eval loss

    print("\n" + "=" * 60)
    print("Starting training...")
    print(f"  Epochs: {TRAINING_CONFIG['num_train_epochs']}")
    print(f"  Batch size: {TRAINING_CONFIG['per_device_train_batch_size']} × {TRAINING_CONFIG['gradient_accumulation_steps']} = {TRAINING_CONFIG['per_device_train_batch_size'] * TRAINING_CONFIG['gradient_accumulation_steps']} effective")
    print(f"  Learning rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"  Max seq length: {MODEL_CONFIG['max_seq_length']}")
    print("=" * 60 + "\n")

    start_time = time.time()
    trainer.train()
    training_time = (time.time() - start_time) / 60

    adapter_path = f"{TRAINING_CONFIG['output_dir']}/final_adapter"
    trainer.save_model(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    print(f"\n{'=' * 60}")
    print("Training complete!")
    print(f"  Adapter saved to: {adapter_path}")
    print(f"  Adapter size: ~27 MB (vs 14 GB full model)")
    print(f"  To load later:")
    print(f"    model = AutoModelForCausalLM.from_pretrained('{MODEL_CONFIG['model_name']}')")
    print(f"    model = PeftModel.from_pretrained(model, '{adapter_path}')")
    print("=" * 60)

    log_experiment(
        config={**MODEL_CONFIG, **LORA_CONFIG, **TRAINING_CONFIG},
        results={
            "training_time_minutes": round(training_time, 1),
        },
        notes=""
    )

    


if __name__ == "__main__":    
    train() 
    
