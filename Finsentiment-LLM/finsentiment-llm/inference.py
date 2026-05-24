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
