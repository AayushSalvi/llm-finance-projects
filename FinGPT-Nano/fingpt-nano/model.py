"""
model.py - The Transformer Architecture (build from scratch).

Build bottom-up, test each piece before moving on:

  1. CausalSelfAttention
     - Q, K, V projections
     - Scaled dot-product: softmax(QK^T / sqrt(d_k)) * V
     - Causal mask (lower triangular)
     - Multi-head: split into h heads, concat, project

  2. FeedForward
     - Linear(d -> 4d) -> GELU -> Linear(4d -> d)

  3. TransformerBlock
     - Pre-Norm: LayerNorm BEFORE attention and FFN
     - Residual connections around both

  4. FinGPTNano (full model)
     - Token embedding + position embedding
     - N transformer blocks
     - Final LayerNorm -> Linear head -> logits
     - Weight tying: embedding weights = output head weights

Test each class: pass random tensor, verify output shape matches input.
"""

# TODO: Build all 4 classes

import math 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from config import CONFIG

class CausalSelfAttention(nn.Module):
   """
   Multi-head self-attention with causal masking.

   Each token produces a Query ("what am I looking for?"),
   Key ("what do I contain?"), and Value ("what info do I carry?").

   Attention score between token i and j = dot product of Q_i and K_j.
   High score = token i should pay attention to token j.

   Causal mask: token i can only attend to tokens 0, 1, ..., i.
   It cannot see future tokens — this makes generation possible.
   """
   def __init__(self):
      super().__init__()
      n_embd = CONFIG["n_embd"]
      n_head = CONFIG["n_head"]
      bias = CONFIG["bias"]
      dropout = CONFIG["dropout"]
      block_size = CONFIG["block_size"]

      assert n_embd % n_head == 0, "n_embd must be divisible by n_head"

      self.n_head = n_head
      self.d_head = n_embd // n_head  # Dimensions per head
      
      self.c_attn = nn.Linear(n_embd, 3* n_embd, bias=bias)

      self.c_proj = nn.Linear(n_embd,n_embd,bias=bias)

      self.attn_dropout = nn.Dropout(dropout)
      self.resid_dropout = nn.Dropout(dropout)

      self.register_buffer(
         "mask",
         torch.tril(torch.ones(block_size,block_size)).view(1,1,block_size,block_size)
      )

   def forward(self,x):
      B,T,C = x.size()

      qkv = self.c_attn(x)
      q,k,v = qkv.split(C,dim=2)

      q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
      k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
      v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

      att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
      att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

      att = F.softmax(att, dim=-1)
      att = self.attn_dropout(att)

      y = att @ v

      y = y.transpose(1, 2).contiguous().view(B, T, C)

      y = self.resid_dropout(self.c_proj(y))

      return y 

# if __name__ == "__main__":
#     B, T, C = 2, CONFIG["block_size"], CONFIG["n_embd"]

#     print("Testing CausalSelfAttention...")
#     attn = CausalSelfAttention()
#     x = torch.randn(B, T, C)
#     out = attn(x)
#     assert out.shape == (B, T, C), f"Shape mismatch: {out.shape}"
#     print(f"  Input:  {x.shape} -> Output: {out.shape} ")

class FeedForward(nn.Module):
   """
   Position-wise feed-forward network.

   Two linear layers with GELU activation:
   Linear(n_embd → 4*n_embd)  — expand
   GELU activation             — non-linearity
   Linear(4*n_embd → n_embd)  — compress back

   The 4x expansion is standard across nearly all transformers.
   
   While attention mixes information BETWEEN tokens,
   the FFN processes each token position INDEPENDENTLY.
   Research suggests factual knowledge is stored in FFN weights.
   """
   def __init__(self):
      super().__init__()
      n_embd = CONFIG["n_embd"]
      bias = CONFIG["bias"]
      dropout = CONFIG["dropout"]

      self.c_fc   = nn.Linear(n_embd, 4 * n_embd, bias=bias)  # Expand
      self.gelu   = nn.GELU()
      self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)  # Compress
      self.dropout = nn.Dropout(dropout)

   def forward(self,x):
      x = self.c_fc(x)
      x = self.gelu(x)
      x = self.c_proj(x)
      x = self.dropout(x)

      return x 


class TransformerBlock(nn.Module):
   """
   One transformer block: Attention + FFN with Pre-Norm and residuals.

   The data flow:
   x  ──────────────────────────── + ──── (residual skip)
   │                               ↑
   └→ LayerNorm → Attention ────────┘

   x  ──────────────────────────── + ──── (residual skip)
   │                               ↑
   └→ LayerNorm → FFN ─────────────┘

   Pre-Norm = normalize BEFORE the sub-layer (modern approach).
   Residuals = add the input back to the output (skip connection).
   This lets gradients flow directly through the skip path,
   making deep networks trainable.
   """

   def __init__(self):
      super().__init__()

      n_embd = CONFIG["n_embd"]

      self.ln_1 = nn.LayerNorm(n_embd)     # Norm before attention
      self.attn = CausalSelfAttention()
      self.ln_2 = nn.LayerNorm(n_embd)     # Norm before FFN
      self.ffn  = FeedForward()

   def forward(self, x):
      # Attention with residual
      x = x + self.attn(self.ln_1(x))

      # FFN with residual
      x = x + self.ffn(self.ln_2(x))

      return x      

class FinGPTNano(nn.Module):
   """
   The complete GPT model.

   Flow:
   Token IDs [B, T]
      → Token Embedding + Position Embedding  [B, T, n_embd]
      → N * TransformerBlock                  [B, T, n_embd]
      → Final LayerNorm                       [B, T, n_embd]
      → Linear output head                    [B, T, vocab_size]

   Weight tying: the token embedding matrix and the output head
   share the same weights. The embedding maps token→vector,
   the output head maps vector→token. Using the same matrix for
   both saves parameters and improves performance.
   """
   
   def __init__(self):
      super().__init__()
      V = CONFIG["vocab_size"]
      D = CONFIG["n_embd"]
      T = CONFIG["block_size"]
      N = CONFIG["n_layer"]

      self.tok_emb = nn.Embedding(V,D)

      self.pos_emb = nn.Embedding(T,D)

      self.drop = nn.Dropout(CONFIG["dropout"])

      self.blocks = nn.ModuleList([TransformerBlock() for _ in range(N)])

      self.ln_f = nn.LayerNorm(D)

      self.lm_head = nn.Linear(D,V,bias=False)

      self.tok_emb.weight = self.lm_head.weight

      self.apply(self._init_weights)

      n_params = sum(p.numel() for p in self.parameters())
      print(f"FinGPT-Nano: {n_params:,} parameters ({n_params / 1e6:.1f}M)")

   def _init_weights(self,module):
      """
        Initialize weights following GPT-2 conventions.
        Linear layers: normal distribution with std=0.02
        Embeddings: normal distribution with std=0.02
        Biases: zeros
        
        Bad initialization → gradients explode or vanish at step 1.
      """
      if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
               torch.nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

   
   def forward(self, idx, targets=None):
      """
        Args:
            idx:     [B, T] token IDs
            targets: [B, T] target token IDs (optional, for computing loss)

        Returns:
            logits: [B, T, vocab_size]
            loss:   scalar (only if targets provided)
        """
      B, T = idx.size()
      device = idx.device

      #Embeddings
      tok_emb = self.tok_emb(idx)                                    # [B, T, D]
      pos_emb = self.pos_emb(torch.arange(T, device=device))        # [T, D]
      x = self.drop(tok_emb + pos_emb)

      #pass through all transformer blocks
      for block in self.blocks:
         x = block(x)

      #Final norm + project to vocabulary
      x = self.ln_f(x)
      logits = self.lm_head(x)

      #Compute loss if targets provided
      loss = None
      if targets is not None:
         loss = F.cross_entropy(
            logits.view(-1,logits.size(-1)),
            targets.view(-1),
            )
         
      return logits,loss
         
      
if __name__ == "__main__":
   B = 2
   T = CONFIG["block_size"]
   C = CONFIG["n_embd"]
   V = CONFIG["vocab_size"]

   x = torch.randn(B, T, C)

   # Test 1
   print("1. CausalSelfAttention...")
   attn = CausalSelfAttention()
   assert attn(x).shape == (B, T, C)
   print(f"   {x.shape} → {attn(x).shape}")

   # Test 2
   print("2. FeedForward...")
   ffn = FeedForward()
   assert ffn(x).shape == (B, T, C)
   print(f"   {x.shape} → {ffn(x).shape}")

   # Test 3
   print("3. TransformerBlock...")
   block = TransformerBlock()
   assert block(x).shape == (B, T, C)
   print(f"   {x.shape} → {block(x).shape}")

   # Test 4
   print("4. FinGPTNano (full model)...")
   model = FinGPTNano()
   idx = torch.randint(0, V, (B, T))
   targets = torch.randint(0, V, (B, T))
   logits, loss = model(idx, targets)
   print(f"   Input:  {idx.shape}")
   print(f"   Logits: {logits.shape}")
   print(f"   Loss:   {loss.item():.4f} (random ≈ {math.log(V):.2f})")

   print("\nAll tests passed!")
