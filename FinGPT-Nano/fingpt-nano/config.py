"""
config.py - All hyperparameters in one place.

Define: vocab_size, block_size (context window), n_layer, n_head,
n_embd, dropout, learning_rate, batch_size, max_steps, device, paths.

Every other file imports from here.
"""

# TODO: Create a CONFIG dict with all hyperparameters
import torch 

CONFIG = {
    "vocab_size" : 50257, #GPT-2 tokenizer vocab
    "block_size" : 256, #context window
    "n_layer" : 6, # number of transformer blocks
    "n_head" : 6, # attention heads
    "n_embd" : 384, #n_embd must be divisible by n_head, because each head gets n_embd // n_head dimensions
    "bias" : False,
    "dropout": 0.1,
    "learning_rate": 3e-4,
    "min_lr": 3e-5,
    "batch_size": 32,
    "max_steps": 5000,
    "warmup_steps" : 200, #linearly increases LR from 0 to peak
    "beta1": 0.9,
    "beta2": 0.95,
    "eval_interval": 500,
    "eval_steps" :20, #number of batches to average for evaluation (reduces noise)
    "log_interval" : 100, # print training loss
    "grad_clip" : 1.0, # Gradient clipping - cap the gradient magnitude
    "save_interval": 1000,
    "device" : "cuda" if torch.cuda.is_available() else "cpu",

    # Paths 
    "train_data": "data/train.bin",
    "val_data": "data/val.bin",
    "checkpoint_dir": "checkpoints",
    "log_dir": "logs",
}
# This gives ~20M parameters — trainable on a laptop CPU in a few hours


if __name__ == "__main__":
    print("FinGPT-Nano Config")
    print("=" * 40)
    for key, value in CONFIG.items():
        print(f"  {key:20s}: {value}")
