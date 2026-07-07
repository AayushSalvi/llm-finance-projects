"""
model.py - Load base model with QLoRA.

Two main things happen here:

1. QUANTIZATION (BitsAndBytesConfig):
   Load Mistral 7B in 4-bit precision using NormalFloat4 format.
   14 GB model → ~3.5 GB in memory.

   Key settings:
   - load_in_4bit=True
   - bnb_4bit_quant_type="nf4" (NormalFloat4, from QLoRA paper)
   - bnb_4bit_compute_dtype=torch.bfloat16 (compute in higher precision)
   - bnb_4bit_use_double_quant=True (quantize the quantization constants too)

2. LoRA ADAPTERS (PeftConfig):
   Attach small trainable matrices to specific layers.
   Only these adapters get updated during training.

   target_modules for Mistral:
   - Attention: q_proj, k_proj, v_proj, o_proj
   - FFN (SwiGLU): gate_proj, up_proj, down_proj

   After LoRA: ~6.8M trainable out of ~7.2B total (0.09%)

Implement:
- load_model(): returns quantized model with LoRA adapters + tokenizer
- Print trainable vs total parameters to verify
"""

# TODO: Build load_model() function


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from config import MODEL_CONFIG, QUANT_CONFIG, LORA_CONFIG

def load_model():
   model_name = MODEL_CONFIG["model_name"]
   print(f"loading {model_name}")

   bnb_config = BitsAndBytesConfig(
      load_in_4bit=QUANT_CONFIG["load_in_4bit"],
      bnb_4bit_quant_type=QUANT_CONFIG["bnb_4bit_quant_type"],
      bnb_4bit_compute_dtype=QUANT_CONFIG["bnb_4bit_compute_dtype"],
      bnb_4bit_use_double_quant=QUANT_CONFIG["bnb_4bit_use_double_quant"],
   )

   model = AutoModelForCausalLM.from_pretrained(
      model_name,
      quantization_config = bnb_config,
      device_map="auto",
   )

   tokenizer = AutoTokenizer.from_pretrained(model_name)

   tokenizer.pad_token = tokenizer.eos_token
   tokenizer.padding_side = "right"

   model = prepare_model_for_kbit_training(model)

   lora_config = LoraConfig(
      r=LORA_CONFIG["r"],
      lora_alpha=LORA_CONFIG["lora_alpha"],
      target_modules=LORA_CONFIG["target_modules"],
      lora_dropout=LORA_CONFIG["lora_dropout"],
      bias=LORA_CONFIG["bias"],
      task_type=LORA_CONFIG["task_type"],
   )

   model = get_peft_model(model,lora_config)
   model.print_trainable_parameters()

   print(f"Model loaded on: {model.device}")
   print(f"Tokenizer vocab size: {tokenizer.vocab_size:,}")

   return model, tokenizer

if __name__ == "__main__":
   model, tokenizer = load_model()

   test_text = "The company reported strong quarterly earnings."
   inputs = tokenizer(test_text, return_tensors="pt").to(model.device)

   print(f"\nTest input: \"{test_text}\"")
   print(f"Token IDs: {inputs['input_ids'][0].tolist()}")
   print(f"Token count: {len(inputs['input_ids'][0])}")

   with torch.no_grad():
      outputs = model(**inputs)
      print(f"Output logits shape: {outputs.logits.shape}")
      print(f"Loss: N/A (no labels provided)")

   print("\n Model loaded and forward pass successful!")
