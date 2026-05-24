"""
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
import argparse
import os
import torch
import torch.nn.functional as F

from config import CONFIG
from model import FinGPTNano
from tokenizer import Tokenizer


def load_model(checkpoint_path=None):
  """Load a trained model from checkpoint."""

  if checkpoint_path is None:
      checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], "best_model.pt")

  if not os.path.exists(checkpoint_path):
      print(f"No checkpoint found at {checkpoint_path}")
      print("Train the model first: python train.py")
      return None

  print(f"Loading model from {checkpoint_path}...")
  checkpoint = torch.load(
      checkpoint_path,
      map_location=CONFIG["device"],
      weights_only=False,
  )

  model = FinGPTNano()
  model.load_state_dict(checkpoint["model_state_dict"])
  model = model.to(CONFIG["device"])
  model.eval()  # Disable dropout for generation

  step = checkpoint.get("step", "?")
  val_loss = checkpoint.get("val_loss", "?")
  print(f"Loaded checkpoint from step {step} (val_loss={val_loss})")

  return model


@torch.no_grad()  # No gradients needed for generation — saves memory
def generate(model, prompt, max_tokens=200, temperature=0.8, top_k=None, top_p=None):
  """
  Generate text token by token.

  The loop:
    1. Encode prompt into token IDs
    2. Feed to model → get probability distribution over next token
    3. Apply sampling strategy to pick one token
    4. Append that token to the sequence
    5. Repeat from step 2

  The model always sees at most block_size tokens (its context window).
  If the sequence grows longer, we crop from the left — the model
  "forgets" the earliest tokens, like a sliding window.
  """
  tok = Tokenizer()
  tokens = tok.encode(prompt)
  x = torch.tensor([tokens], dtype=torch.long, device=CONFIG["device"])

  for _ in range(max_tokens):
      # Crop to context window if sequence exceeds block_size
      x_cond = x[:, -CONFIG["block_size"]:]

      # Forward pass — get logits for the LAST token position only
      logits, _ = model(x_cond)
      logits = logits[:, -1, :]  # [1, vocab_size]

      # ── Temperature ──
      # Dividing logits by temperature before softmax:
      #   T < 1.0 → sharpens distribution (more confident, less random)
      #   T = 1.0 → no change
      #   T > 1.0 → flattens distribution (less confident, more random)
      if temperature != 1.0:
          logits = logits / temperature

      # ── Top-k filtering ──
      # Only keep the k most probable tokens, set everything else to -inf
      # This prevents the model from picking very unlikely tokens
      if top_k is not None:
          top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
          # Everything below the k-th value gets masked out
          logits[logits < top_values[:, [-1]]] = float("-inf")

      # ── Top-p (nucleus) filtering ──
      # Keep the smallest set of tokens whose cumulative probability >= p
      # More adaptive than top-k: if the model is confident, fewer tokens
      # are kept; if uncertain, more tokens are kept
      if top_p is not None:
          sorted_logits, sorted_indices = torch.sort(logits, descending=True)
          cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

          # Find where cumulative probability exceeds top_p
          sorted_mask = cumulative_probs > top_p
          # Shift right so the first token above threshold is kept
          sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
          sorted_mask[:, 0] = False

          # Map back to original token positions and zero them out
          mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
          logits[mask] = float("-inf")

      # ── Sample ──
      probs = F.softmax(logits, dim=-1)
      next_token = torch.multinomial(probs, num_samples=1)

      # Append to sequence
      x = torch.cat([x, next_token], dim=1)

  # Decode everything back to text
  return tok.decode(x[0].tolist())


def main():
  parser = argparse.ArgumentParser(description="Generate text with FinGPT-Nano")
  parser.add_argument("--prompt", type=str, default="The company reported quarterly revenue of")
  parser.add_argument("--max_tokens", type=int, default=200)
  parser.add_argument("--temperature", type=float, default=0.8)
  parser.add_argument("--top_k", type=int, default=None)
  parser.add_argument("--top_p", type=float, default=None)
  parser.add_argument("--checkpoint", type=str, default=None)
  parser.add_argument("--num_samples", type=int, default=1)
  args = parser.parse_args()

  model = load_model(args.checkpoint)
  if model is None:
      return

  print(f"\nPrompt: \"{args.prompt}\"")
  print(f"Settings: temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")

  for i in range(args.num_samples):
      if args.num_samples > 1:
          print(f"\n{'─' * 50}")
          print(f"Sample {i + 1}/{args.num_samples}")

      output = generate(
          model,
          prompt=args.prompt,
          max_tokens=args.max_tokens,
          temperature=args.temperature,
          top_k=args.top_k,
          top_p=args.top_p,
      )

      print(f"{'─' * 50}")
      print(output)


if __name__ == "__main__":
  main()