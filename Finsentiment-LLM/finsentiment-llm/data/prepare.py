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
