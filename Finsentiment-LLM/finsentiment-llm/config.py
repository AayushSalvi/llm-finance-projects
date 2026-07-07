"""
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

import torch 

MODEL_CONFIG = {
   "model_name" : "mistralai/Mistral-7B-Instruct-v0.3",
   "max_seq_length" : 512
}

QUANT_CONFIG ={
   "load_in_4bit" : True,
   "bnb_4bit_quant_type" : "nf4",
   "bnb_4bit_compute_dtype" : torch.bfloat16,
   "bnb_4bit_use_double_quant" : True
}

LORA_CONFIG = {
   "r" : 16,
   "lora_alpha" : 32,
   "lora_dropout" : 0.05,
   "bias" : "none",
   "task_type" : "CAUSAL_LM",
   "target_modules" : [
      "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
      "gate_proj", "up_proj", "down_proj",       # FFN
   ]
}


TRAINING_CONFIG = {
   "output_dir" : "./checkpoints",
   "num_train_epochs":3,
   "per_device_train_batch_size" : 4,
   "gradient_accumulation_steps": 4,
   "learning_rate" : 2e-4,
   "weight_decay" : 0.01,
   "learning_ratio" : 0.03,
   "lr_scheduler_type" : "cosine",
   "warmup_ratio": 0.03,
   "bf16" : True,
   "gradient_checkpointing": True,
   "logging_steps" : 10,
   "eval_strategy" : "epoch",
   "save_strategy" : "epoch",
   "load_best_model_at_end" : True,
   "packing" : False
}


if __name__ == "__main__":
   print("="*50)
   print("FinSentiment-LLM Config")

   print("\nModel")
   for k,v in MODEL_CONFIG.items():
      print(f"  {k:25s}: {v}")

   print("\nQuantization:")
   for k, v in QUANT_CONFIG.items():
      print(f"  {k:25s}: {v}")

   print("\nLoRA:")
   for k, v in LORA_CONFIG.items():
      print(f"  {k:25s}: {v}")

   print("\nTraining:")
   for k, v in TRAINING_CONFIG.items():
      print(f"  {k:25s}: {v}")

   d = 4096 
   r = LORA_CONFIG["r"]
   n_layers = 32 # Mistral 7B has 32 layers 
   n_modules = len(LORA_CONFIG["target_modules"])
   params_per_module = d * r + r * d 
   total_lora_params = n_layers * n_modules * params_per_module

   print(f"\n  Estimated LoRA params:   {total_lora_params:,} ({total_lora_params/1e6:.1f}M)")
   print(f"  Base model params:       ~7,200,000,000 (7.2B)")
   print(f"  Trainable fraction:      {total_lora_params/7.2e9*100:.2f}%")
   print(f"  Adapter save size:       ~{total_lora_params * 2 / 1e6:.0f} MB (BF16)")
