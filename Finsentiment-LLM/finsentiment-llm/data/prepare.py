"""
data/prepare.py - Download and format financial sentiment data.

Dataset: Financial PhraseBank (takala/financial_phrasebank)
- 4,840 financial sentences annotated by experts
- Labels: positive (0), neutral (1), negative (2)
- Subset "sentences_allagree" = only sentences where ALL annotators agreed

YOUR JOB:
1. Load the dataset from HuggingFace
2. Format each example as an instruction-response pair:

   <s>[INST] Analyze the sentiment of this financial text.
   Classify as: positive, negative, or neutral.

   Text: "The company reported strong Q4 earnings"
   [/INST]
   Sentiment: positive</s>

3. Split into train/test (90/10)
4. Save as HuggingFace Dataset object

WHY THIS FORMAT?
Mistral uses [INST]...[/INST] tags for instruction tuning.
The model learns: see [INST] → follow the instruction → generate response.
"""

# TODO: Load dataset, format for SFT, split, save

from datasets import load_dataset,Dataset, DatasetDict

def prepare_data(output_dir="data"):
   """Download Financial PhraseBank and format for instruction tuning."""

   print("Loading Financial PhraseBank from HuggingFace...")
   ds_train = load_dataset("FinanceMTEB/financial_phrasebank", split="train")
   ds_test = load_dataset("FinanceMTEB/financial_phrasebank", split="test")
   print(f"Train: {len(ds_train):,} examples, Test: {len(ds_test):,} examples")

   # Check columns and distribution
   print(f"Columns: {ds_train.column_names}")
   counts = {}
   for item in ds_train:
      counts[item["label_text"]] = counts.get(item["label_text"], 0) + 1
   print(f"Distribution: {counts}")

   # Format both splits
   def format_split(ds):
      formatted = []
      for item in ds:
         sentence = item["text"]
         label = item["label_text"]

         text = (
         f"[INST] Analyze the sentiment of the following financial text. "
         f"Classify as: positive, negative, or neutral.\n\n"
         f"Text: \"{sentence}\"\n"
         f"[/INST]\n"
         f"Sentiment: {label}"
         )

         formatted.append({
               "text": text,
               "label": item["label"],
               "sentence": sentence,
         })
      return Dataset.from_list(formatted)

   train_dataset = format_split(ds_train)
   test_dataset = format_split(ds_test)

   # Save as HuggingFace DatasetDict
   from datasets import DatasetDict
   dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
   dataset.save_to_disk(f"{output_dir}/financial_sentiment")

   print(f"\nTrain: {len(train_dataset):,} examples")
   print(f"Test:  {len(test_dataset):,} examples")
   print(f"Saved to {output_dir}/financial_sentiment/")

   # Preview
   print(f"\n{'=' * 60}")
   print("PREVIEW (3 examples):")
   print("=" * 60)
   for i in range(3):
      print(f"\n--- Example {i+1} ---")
      print(train_dataset[i]["text"])

   return dataset


if __name__ == "__main__":
   prepare_data()