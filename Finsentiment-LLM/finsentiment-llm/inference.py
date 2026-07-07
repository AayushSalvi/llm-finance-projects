"""
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
"""
inference.py - Interactive financial sentiment analysis.

Load your fine-tuned model and analyze any financial text.
This is your demo file — proves the model works on real input.

Run:
  CUDA_VISIBLE_DEVICES=4 python inference.py --text "Apple reported record revenue"
  CUDA_VISIBLE_DEVICES=4 python inference.py --interactive
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from config import MODEL_CONFIG, QUANT_CONFIG, TRAINING_CONFIG


def load_model():
  """Load base model + LoRA adapter."""

  adapter_path = f"{TRAINING_CONFIG['output_dir']}/final_adapter"

  print(f"Loading base model: {MODEL_CONFIG['model_name']}")
  print(f"Loading adapter: {adapter_path}")

  bnb_config = BitsAndBytesConfig(
      load_in_4bit=QUANT_CONFIG["load_in_4bit"],
      bnb_4bit_quant_type=QUANT_CONFIG["bnb_4bit_quant_type"],
      bnb_4bit_compute_dtype=QUANT_CONFIG["bnb_4bit_compute_dtype"],
      bnb_4bit_use_double_quant=QUANT_CONFIG["bnb_4bit_use_double_quant"],
  )

  base_model = AutoModelForCausalLM.from_pretrained(
      MODEL_CONFIG["model_name"],
      quantization_config=bnb_config,
      device_map="auto",
  )

  model = PeftModel.from_pretrained(base_model, adapter_path)
  tokenizer = AutoTokenizer.from_pretrained(adapter_path)
  tokenizer.pad_token = tokenizer.eos_token

  model.eval()
  print("Model loaded!\n")
  return model, tokenizer


def analyze(model, tokenizer, text):
  """Analyze sentiment of a single financial text."""

  prompt = (
      f"<s>[INST] Analyze the sentiment of the following financial text. "
      f"Classify as: positive, negative, or neutral.\n\n"
      f"Text: \"{text}\"\n"
      f"[/INST]\n"
      f"Sentiment:"
  )

  inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

  with torch.no_grad():
      output = model.generate(
          **inputs,
          max_new_tokens=10,
          temperature=0.1,
          do_sample=True,
          pad_token_id=tokenizer.eos_token_id,
      )

  generated_ids = output[0][inputs["input_ids"].shape[1]:]
  response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

  # Extract label
  label = "unclear"
  for sentiment in ["positive", "negative", "neutral"]:
      if sentiment in response.lower():
          label = sentiment
          break

  return label, response


def interactive_mode(model, tokenizer):
  """Interactive loop — type financial text, get sentiment."""

  print("=" * 50)
  print("Financial Sentiment Analyzer")
  print("Type financial text and get sentiment analysis.")
  print("Type 'quit' to exit.")
  print("=" * 50)

  while True:
      text = input("\nEnter text: ").strip()

      if text.lower() in ["quit", "exit", "q"]:
          print("Goodbye!")
          break

      if not text:
          continue

      label, raw = analyze(model, tokenizer, text)
      print(f"Sentiment: {label}")


def main():
  parser = argparse.ArgumentParser(description="Financial Sentiment Analysis")
  parser.add_argument("--text", type=str, default=None, help="Text to analyze")
  parser.add_argument("--interactive", action="store_true", help="Interactive mode")
  args = parser.parse_args()

  model, tokenizer = load_model()

  if args.interactive:
      interactive_mode(model, tokenizer)
  elif args.text:
      label, raw = analyze(model, tokenizer, args.text)
      print(f"Text:      \"{args.text}\"")
      print(f"Sentiment: {label}")
      print(f"Raw output: {raw}")
  else:
      # Default: run on sample financial texts
      samples = [
          "Revenue increased 23% year-over-year to $4.8 billion.",
          "The company announced 2,000 layoffs amid declining sales.",
          "The board appointed a new CEO effective January 1.",
          "Net losses widened to $120 million in the third quarter.",
          "Operating margins expanded by 300 basis points.",
      ]

      print("=" * 60)
      print("SAMPLE PREDICTIONS")
      print("=" * 60)

      for text in samples:
          label, _ = analyze(model, tokenizer, text)
          print(f"\n  [{label:>8}]  \"{text}\"")


if __name__ == "__main__":
  main()