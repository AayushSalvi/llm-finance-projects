"""
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

import os 
import time
import math 
import numpy as np 
import torch 

from config import CONFIG
from model import FinGPTNano

def get_batch(split):
  data_path = CONFIG["train_data"] if split == "train" else CONFIG["val_data"]
  data = np.memmap(data_path, dtype=np.uint16, mode= "r")

  ix = torch.randint(len(data) - CONFIG["block_size"] - 1, (CONFIG["batch_size"],))

  x = torch.stack(
    [torch.from_numpy(data[i:i + CONFIG["block_size"]].astype(np.int64)) for i in ix]
  )

  y = torch.stack(
    [torch.from_numpy(data[i+1:i+1+CONFIG["block_size"]].astype(np.int64)) for i in ix]
  )

  return x.to(CONFIG["device"]), y.to(CONFIG["device"])

def get_lr(step):
  if step < CONFIG["warmup_steps"]:
    return CONFIG["learning_rate"] * (step + 1) / CONFIG["warmup_steps"]
  
  progress = (step - CONFIG["warmp_steps"]) / (CONFIG["max_steps"] - CONFIG["warmup_steps"])
  progress = min(progress,1.0) 
  coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
  return CONFIG["min_lr"] + coeff * (CONFIG["learning_rate"] - CONFIG["min_lr"])


@torch.no_grad()
def estimate_loss(model):
  model.eval()
  results = {}

  for split in ["train","val"]:
    losses = []
    for _ in range(CONFIG["eval_steps"]):
      x,y = get_batch(split)
      _,loss= model(x,y)
      losses.append(loss.item())

    results[split] = np.mean(losses)

  model.train()
  return results



def train():
  """Main training function."""
  

  # Print config
  print("=" * 60)
  print("FinGPT-Nano Training")
  print("=" * 60)
  for key, value in CONFIG.items():
      print(f"  {key:20s}: {value}")

  # Create output directories
  os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
  os.makedirs(CONFIG["log_dir"], exist_ok=True)

  # Check data exists
  if not os.path.exists(CONFIG["train_data"]):
      print("\nERROR: Training data not found!")
      print("Run these first:")
      print("  python data/download.py")
      print("  python data/prepare.py")
      return

  # Create model
  print("\nCreating model...")
  model = FinGPTNano()
  model = model.to(CONFIG["device"])

  # Create optimizer
  # AdamW (not Adam!) — decouples weight decay from gradient updates
  # This is important: standard Adam applies weight decay to the gradient,
  # which interacts poorly with adaptive learning rates. AdamW applies
  # weight decay directly to the weights themselves.
  optimizer = torch.optim.AdamW(
      model.parameters(),
      lr=CONFIG["learning_rate"],
      betas=(CONFIG["beta1"], CONFIG["beta2"]),
      weight_decay=CONFIG["weight_decay"],
  )

  # ── Training Loop ──
  print(f"\nTraining on {CONFIG['device']} for {CONFIG['max_steps']:,} steps...")
  print(f"Each step processes {CONFIG['batch_size'] * CONFIG['block_size']:,} tokens")
  print("-" * 60)

  loss_history = []
  best_val_loss = float("inf")
  start_time = time.time()

  for step in range(CONFIG["max_steps"]):

      # ── Update learning rate ──
      lr = get_lr(step)
      for param_group in optimizer.param_groups:
          param_group["lr"] = lr

      # ── Get a batch ──
      x, y = get_batch("train")

      # ── Forward pass ──
      logits, loss = model(x, y)

      # ── Backward pass ──
      optimizer.zero_grad(set_to_none=True)  # Clear old gradients
      loss.backward()                         # Compute new gradients

      # ── Gradient clipping ──
      # If gradients are too large, rescale them so their total norm = grad_clip
      # Without this, a single bad batch can produce huge gradients
      # that blow up all the weights (training loss → infinity)
      torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])

      # ── Update weights ──
      optimizer.step()

      # ── Logging ──
      loss_val = loss.item()
      loss_history.append(loss_val)

      if step % CONFIG["log_interval"] == 0:
          elapsed = time.time() - start_time
          tokens_seen = (step + 1) * CONFIG["batch_size"] * CONFIG["block_size"]
          tokens_per_sec = tokens_seen / elapsed if elapsed > 0 else 0
          print(
              f"Step {step:>5d}/{CONFIG['max_steps']} │ "
              f"Loss: {loss_val:.4f} │ "
              f"LR: {lr:.2e} │ "
              f"Tok/s: {tokens_per_sec:,.0f} │ "
              f"Time: {elapsed:.0f}s"
          )

      # ── Evaluation ──
      if step > 0 and step % CONFIG["eval_interval"] == 0:
          losses = estimate_loss(model)
          print(
              f"\n  ► Eval @ step {step}: "
              f"train={losses['train']:.4f}  "
              f"val={losses['val']:.4f}"
          )

          # Save best model (lowest validation loss)
          if losses["val"] < best_val_loss:
              best_val_loss = losses["val"]
              path = os.path.join(CONFIG["checkpoint_dir"], "best_model.pt")
              torch.save({
                  "model_state_dict": model.state_dict(),
                  "optimizer_state_dict": optimizer.state_dict(),
                  "step": step,
                  "val_loss": best_val_loss,
                  "config": CONFIG,
              }, path)
              print(f"  ► New best model saved (val_loss={best_val_loss:.4f})")
          print()

      # ── Periodic checkpoint ──
      if step > 0 and step % CONFIG["save_interval"] == 0:
          path = os.path.join(CONFIG["checkpoint_dir"], f"step_{step}.pt")
          torch.save({
              "model_state_dict": model.state_dict(),
              "step": step,
              "config": CONFIG,
          }, path)

  # ── Save loss history ──
  np.save(os.path.join(CONFIG["log_dir"], "loss_history.npy"), np.array(loss_history))

  # ── Final summary ──
  total_time = time.time() - start_time
  print("=" * 60)
  print("Training complete!")
  print(f"  Total time:     {total_time / 60:.1f} minutes")
  print(f"  Final loss:     {loss_history[-1]:.4f}")
  print(f"  Best val loss:  {best_val_loss:.4f}")
  print(f"  Checkpoints:    {CONFIG['checkpoint_dir']}/")
  print(f"  Loss history:   {CONFIG['log_dir']}/loss_history.npy")
  print("=" * 60)


if __name__ == "__main__":
  train()