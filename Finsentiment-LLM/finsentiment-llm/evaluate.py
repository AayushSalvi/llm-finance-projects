"""
evaluate.py - Compare base model vs fine-tuned model.

This is the most important file for proving your model works.

Implement:
1. Load the fine-tuned model (base + LoRA adapter)
2. Load the base model WITHOUT the adapter
3. Run both on the same test set
4. For each test example:
   - Feed the prompt to the model
   - Extract the predicted sentiment from the generated text
   - Compare with the true label
5. Print classification_report (precision, recall, F1 per class)
6. Print confusion matrix
7. Show side-by-side examples where they differ

Expected results:
- Base model: ~55-65% accuracy (it can do sentiment but isn't specialized)
- Fine-tuned: ~85-92% accuracy (trained specifically on financial sentiment)

That gap is what LoRA fine-tuning gives you.
"""

# TODO: Build evaluation pipeline
