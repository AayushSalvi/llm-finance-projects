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

import torch 
from datasets import load_from_disk
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel 
import json
import os

from config import MODEL_CONFIG, QUANT_CONFIG, TRAINING_CONFIG

def load_base_model():
   print("LOADING BASE MODEL (NO ADAPTER)")
   bnb_config = BitsAndBytesConfig(
      load_in_4bit=QUANT_CONFIG["load_in_4bit"],
      bnb_4bit_quant_type=QUANT_CONFIG["bnb_4bit_quant_type"],
      bnb_4bit_compute_dtype=QUANT_CONFIG["bnb_4bit_compute_dtype"],
      bnb_4bit_use_double_quant=QUANT_CONFIG["bnb_4bit_use_double_quant"],
   )

   model = AutoModelForCausalLM.from_pretrained(
      MODEL_CONFIG["model_name"],
      quantization_config = bnb_config,
      device_map = "auto",
   )

   tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["model_name"])
   tokenizer.pad_token = tokenizer.eos_token 

   model.eval()
   return model, tokenizer


def load_finetuned_model():
   print("Loading fine-tuned model (base + adapter)...")
   bnb_config = BitsAndBytesConfig(
      load_in_4bit=QUANT_CONFIG["load_in_4bit"],
      bnb_4bit_quant_type=QUANT_CONFIG["bnb_4bit_quant_type"],
      bnb_4bit_compute_dtype=QUANT_CONFIG["bnb_4bit_compute_dtype"],
      bnb_4bit_use_double_quant=QUANT_CONFIG["bnb_4bit_use_double_quant"],
   )

   base_model = AutoModelForCausalLM.from_pretrained(
      MODEL_CONFIG["model_name"],
      quantization_config = bnb_config,
      device_map = "auto",
   )

   # Apply LoRa adapter on top 
   # This merges the 27MB adapter with the 3.5GB base model 
   adapter_path = f"{TRAINING_CONFIG['output_dir']}/final_adapter"
   model = PeftModel.from_pretrained(base_model,adapter_path)

   tokenizer = AutoTokenizer.from_pretrained(adapter_path)
   tokenizer.pad_token = tokenizer.eos_token

   model.eval()
   return model, tokenizer


def predict_sentiment(model, tokenizer, sentence):
   """
   Run a single prediction
   
   Feed the instruction prompt to the model, generate a short response, 
   and extract the sentiment label from the generated text.
   """

   prompt = {
      f"<s>[INST] Analyze the sentiment of the following financial text. "
        f"Classify as: positive, negative, or neutral.\n\n"
        f"Text: \"{sentence}\"\n"
        f"[/INST]\n"
        f"Sentiment:"
   }

   inputs = tokenizer(prompt, return_tensors = "pt").to(model.device)

   with torch.no_grad():
      output = model.generate(
         **inputs,
         max_new_tokens = 10,
         temperature = 0.1,
         do_sample = True,
         pad_token_id = tokenizer.eos_token_id,
      )

   generated_ids = output[0][inputs["input_ids"].shape[1]:]
   response = tokenizer.decode(generated_ids, skip_special_tokens= True).strip().lower()

   # Extract sentiment
   for label in ["positive","negative","neutral"]:
      if label in response:
         return label
      
   return f"unclear: {response}"

def evaluate_model(model, tokenizer,test_data,model_name="Model"):
   """Run Evaluation on full test test"""
   label_map = {0:"negative",1:"neutral",2:"positive"}

   y_true = []
   y_pred = [] 
   unclear = 0 

   print(f"\n Evaluating {model_name} on {len(test_data)} examples...")

   for i, example in enumerate(test_data):
      true_label = label_map[example["label"]]
      pred_label = predict_sentiment(model,tokenizer,example["sentence"])

      y_true.append(true_label)

      if pred_label.startswith("unclear"):
         unclear +=1
         y_pred.append("neutral")

      else:
         y_pred.append(pred_label)

      if (i + 1) % 50 == 0:
         correct_so_far = sum(1 for t, p in zip(y_true, y_pred) if t == p)
         print(f"  [{i+1}/{len(test_data)}] Accuracy so far: {correct_so_far/(i+1)*100:.1f}%")

   
   print(f"\n{'=' * 60}")
   print(f"RESULTS: {model_name}")
   print("=" * 60)

   if unclear > 0:
      print(f"Unclear predictions: {unclear}/{len(test_data)} ({unclear/len(test_data)*100:.1f}%)")

   print(f"\n{classification_report(y_true, y_pred, digits=3)}")

   # Confusion matrix
   labels = ["negative", "neutral", "positive"]
   cm = confusion_matrix(y_true, y_pred, labels=labels)
   print("Confusion Matrix:")
   print(f"{'':>12} {'neg':>8} {'neu':>8} {'pos':>8}  ← predicted")
   for i, row_label in enumerate(labels):
      print(f"  {row_label:>10} {cm[i][0]:>8} {cm[i][1]:>8} {cm[i][2]:>8}")
   print(f"  ↑ actual")

   accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
   return accuracy

def show_side_by_side(base_model, base_tok, ft_model, ft_tok, test_data, n=10):
   """Show examples where base and fine-tuned models differ."""

   print(f"\n{'=' * 60}")
   print("SIDE-BY-SIDE COMPARISON (where models disagree)")
   print("=" * 60)

   label_map = {0: "negative", 1: "neutral", 2: "positive"}
   shown = 0

   for example in test_data:
      if shown >= n:
         break

      true_label = label_map[example["label"]]
      base_pred = predict_sentiment(base_model, base_tok, example["sentence"])
      ft_pred = predict_sentiment(ft_model, ft_tok, example["sentence"])

      # Only show examples where they disagree
      if base_pred != ft_pred:
         shown += 1
         base_correct = "✓" if base_pred == true_label else "✗"
         ft_correct = "✓" if ft_pred == true_label else "✗"

         print(f"\n--- Example {shown} ---")
         print(f"Text:       \"{example['sentence'][:100]}...\"")
         print(f"True label: {true_label}")
         print(f"Base model: {base_pred} {base_correct}")
         print(f"Fine-tuned: {ft_pred} {ft_correct}")


def main():
   # Load test data
   dataset = load_from_disk("data/financial_sentiment")
   test_data = dataset["test"]

   print("=" * 60)
   print("FinSentiment-LLM Evaluation")
   print(f"Test set: {len(test_data)} examples")
   print("=" * 60)

   # Evaluate base model
   base_model, base_tok = load_base_model()
   base_accuracy = evaluate_model(base_model, base_tok, test_data, "Base Mistral 7B")

   # Free GPU memory before loading fine-tuned model
   del base_model
   torch.cuda.empty_cache()

   # Evaluate fine-tuned model
   ft_model, ft_tok = load_finetuned_model()
   ft_accuracy = evaluate_model(ft_model, ft_tok, test_data, "Fine-tuned (LoRA)")

   # Summary
   print(f"\n{'=' * 60}")
   print("SUMMARY")
   print("=" * 60)
   print(f"  Base model accuracy:      {base_accuracy*100:.1f}%")
   print(f"  Fine-tuned accuracy:      {ft_accuracy*100:.1f}%")
   print(f"  Improvement:              +{(ft_accuracy - base_accuracy)*100:.1f} percentage points")
   print(f"  Relative improvement:     {(ft_accuracy/base_accuracy - 1)*100:.1f}%")

   # Side-by-side comparison (reload base for comparison)
   # Uncomment if you have enough GPU memory for both models:
   # base_model2, base_tok2 = load_base_model()
   # show_side_by_side(base_model2, base_tok2, ft_model, ft_tok, test_data)

   import json
   log_path = "logs/experiments.jsonl"
   if os.path.exists(log_path):
      with open(log_path, "r") as f:
         lines = f.readlines()
      if lines:
         last = json.loads(lines[-1])
         last["results"]["base_accuracy"] = round(base_accuracy, 4)
         last["results"]["ft_accuracy"] = round(ft_accuracy, 4)
         last["results"]["improvement"] = round((ft_accuracy - base_accuracy) * 100, 1)
         lines[-1] = json.dumps(last) + "\n"
         with open(log_path, "w") as f:
               f.writelines(lines)
         print("Updated experiment log with eval results.")
   


if __name__ == "__main__":
   main()