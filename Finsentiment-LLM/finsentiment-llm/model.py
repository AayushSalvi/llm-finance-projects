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
