"""
evaluate.py - Evaluate model quality.

Implement:
  - Perplexity = exp(average cross-entropy loss)
    PPL=1 is perfect, PPL=50257 is random
  - Plot training loss curve from saved logs
  - Compare train vs val loss (check for overfitting)

Library: matplotlib for plotting
"""

# TODO: Build evaluation and plotting
import os
import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (works on servers without displays)
import matplotlib.pyplot as plt

from config import CONFIG
from model import FinGPTNano


def evaluate_perplexity(model, data_path, num_batches=50):
  """
  Compute average loss and perplexity on a dataset.

  Perplexity = exp(loss)
  It represents the "effective vocabulary size" the model is
  choosing between at each position. Lower is better.
  """
  data = np.memmap(data_path, dtype=np.uint16, mode="r")
  model.eval()

  losses = []
  with torch.no_grad():
      for _ in range(num_batches):
          ix = torch.randint(
              len(data) - CONFIG["block_size"] - 1,
              (CONFIG["batch_size"],),
          )
          x = torch.stack(
              [torch.from_numpy(data[i : i + CONFIG["block_size"]].astype(np.int64)) for i in ix]
          ).to(CONFIG["device"])
          y = torch.stack(
              [torch.from_numpy(data[i + 1 : i + 1 + CONFIG["block_size"]].astype(np.int64)) for i in ix]
          ).to(CONFIG["device"])

          _, loss = model(x, y)
          losses.append(loss.item())

  avg_loss = np.mean(losses)
  perplexity = math.exp(avg_loss)
  return avg_loss, perplexity


def plot_loss_curve(log_dir="logs", save_path=None):
  """
  Plot and save the training loss curve.

  Two views:
    Left:  Raw loss (with smoothed overlay)
    Right: Log-scale loss (shows early rapid improvement)
  """
  loss_path = os.path.join(log_dir, "loss_history.npy")
  if not os.path.exists(loss_path):
      print(f"No loss history found at {loss_path}")
      print("Train the model first: python train.py")
      return

  losses = np.load(loss_path)
  print(f"Loaded {len(losses):,} loss values")

  # Smooth with moving average for clearer trend
  window = min(100, len(losses) // 10)
  if window > 1:
      smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
  else:
      smoothed = losses

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

  # Left plot: linear scale
  ax1.plot(losses, alpha=0.2, color="steelblue", linewidth=0.5, label="Raw")
  ax1.plot(
      range(window - 1, window - 1 + len(smoothed)),
      smoothed, color="steelblue", linewidth=2, label="Smoothed"
  )
  ax1.set_xlabel("Step")
  ax1.set_ylabel("Cross-Entropy Loss")
  ax1.set_title("Training Loss")
  ax1.legend()
  ax1.grid(True, alpha=0.3)

  # Right plot: log scale (shows early improvement better)
  ax2.plot(losses, alpha=0.2, color="coral", linewidth=0.5, label="Raw")
  ax2.plot(
      range(window - 1, window - 1 + len(smoothed)),
      smoothed, color="coral", linewidth=2, label="Smoothed"
  )
  ax2.set_xlabel("Step")
  ax2.set_ylabel("Loss (log scale)")
  ax2.set_title("Training Loss (Log Scale)")
  ax2.set_yscale("log")
  ax2.legend()
  ax2.grid(True, alpha=0.3)

  plt.tight_layout()

  if save_path is None:
      save_path = os.path.join(log_dir, "loss_curve.png")
  plt.savefig(save_path, dpi=150, bbox_inches="tight")
  print(f"Loss curve saved to {save_path}")
  plt.close()


def main():
  # Load best model
  checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], "best_model.pt")
  if not os.path.exists(checkpoint_path):
      print("No trained model found. Run 'python train.py' first.")
      return

  checkpoint = torch.load(
      checkpoint_path,
      map_location=CONFIG["device"],
      weights_only=False,
  )
  model = FinGPTNano()
  model.load_state_dict(checkpoint["model_state_dict"])
  model = model.to(CONFIG["device"])

  step = checkpoint.get("step", "?")
  print("=" * 50)
  print(f"FinGPT-Nano Evaluation (checkpoint: step {step})")
  print("=" * 50)

  # Evaluate on both sets
  print("\nValidation set:")
  val_loss, val_ppl = evaluate_perplexity(model, CONFIG["val_data"])
  print(f"  Loss:       {val_loss:.4f}")
  print(f"  Perplexity: {val_ppl:.2f}")

  print("\nTraining set:")
  train_loss, train_ppl = evaluate_perplexity(model, CONFIG["train_data"])
  print(f"  Loss:       {train_loss:.4f}")
  print(f"  Perplexity: {train_ppl:.2f}")

  # Interpretation
  print(f"\n{'─' * 50}")
  print("Interpretation:")

  gap = val_loss - train_loss
  if gap > 0.5:
      print(f"  Train-val gap: {gap:.2f} — overfitting likely")
  elif gap > 0.2:
      print(f"  Train-val gap: {gap:.2f} — slight overfitting, acceptable")
  else:
      print(f"  Train-val gap: {gap:.2f} — healthy")

  random_ppl = CONFIG["vocab_size"]
  print(f"\n  Random baseline: {random_ppl:,} PPL")
  print(f"  Your model:      {val_ppl:.0f} PPL")
  print(f"  Improvement:     {random_ppl / val_ppl:.0f}x better than random")

  if val_ppl > 500:
      print("\n  → Model is still mostly random. Needs more training.")
  elif val_ppl > 100:
      print("\n  → Model is learning patterns but still quite uncertain.")
  elif val_ppl > 50:
      print("\n  → Model has learned basic language structure. Good!")
  else:
      print("\n  → Strong language modeling for this model size!")

  # Plot loss curve
  print()
  plot_loss_curve()


if __name__ == "__main__":
  main()
