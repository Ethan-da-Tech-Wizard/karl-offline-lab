# ==============================================================================
# Program: mini_transformer.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Minimal standalone PyTorch transformer architecture implementation.
#          Provides an isolated, lightweight neural network for algorithmic
#          verification and unit testing.
# ==============================================================================

"""
Educational Decoder-Only Transformer (Mini-GPT) in PyTorch
==========================================================
This module implements a standard GPT-style character-level language model.
It is designed for educational purposes: clean, heavily annotated,
and self-contained.

Architecture:
- CharTokenizer: Character-level BPE-alternative.
- Head: Single-head self-attention.
- MultiHeadAttention: Multi-head self-attention.
- FeedForward: Multi-layer perceptron (MLP).
- Block: Multi-head self-attention + MLP with pre-layer normalization and residual connections.
- MiniGPT: Main model with token/position embeddings, Blocks, and LM linear head.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

class CharTokenizer:
    """A simple character-level tokenizer that maps characters to integers and back."""
    def __init__(self):
        self.chars = []
        self.vocab_size = 0
        self.stoi = {}
        self.itos = {}

    def fit(self, text: str):
        """Build the vocabulary from ``text`` (sorted unique characters; a single space if empty)."""
        self.chars = sorted(list(set(text)))
        # Ensure at least some characters are present
        if not self.chars:
            self.chars = [" "]
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s: str) -> list[int]:
        """Convert a string to ids; unknown characters map to id 0."""
        return [self.stoi.get(c, 0) for c in s]  # Fallback to index 0 for unknown characters

    def decode(self, tokens: list[int]) -> str:
        """Convert ids back to text; unknown ids are dropped."""
        return "".join([self.itos.get(i, "") for i in tokens])


class Head(nn.Module):
    """One head of self-attention."""
    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Lower-triangular mask stored as a buffer: moves with the model but is not trained.
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return attention output of shape (B, T, head_size) for input x of shape (B, T, n_embd)."""
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        
        # Compute attention scores ("affinities")
        # Scaled dot-product scores. NOTE: the scale uses C (n_embd), where the original paper uses head_size.
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)  # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        # Apply causal masking (prevent looking into the future)
        # Causal mask: future positions get -inf so softmax gives them zero weight.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)
        
        # Perform weighted aggregation of values
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v      # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""
    def __init__(self, n_heads: int, n_embd: int, head_size: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate head outputs, then project back to n_embd with dropout."""
        # Concatenate outputs from all attention heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Apply projection layer and dropout
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """A simple linear layer followed by non-linearity and projection."""
    def __init__(self, n_embd: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP to every position independently."""
        return self.net(x)


class Block(nn.Module):
    """Transformer block: self-attention followed by feed-forward, with residual links."""
    def __init__(self, n_embd: int, n_heads: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        # Heads split the embedding evenly (n_embd must be divisible by n_heads to keep the width).
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(n_heads, n_embd, head_size, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pre-norm attention and feed-forward with residual (skip) connections."""
        # Pre-LayerNorm formulation (standard in modern architectures)
        # Adding the input back ("residual") keeps gradients flowing through deep stacks.
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """The main Decoder-Only GPT-style Language Model."""
    def __init__(self, vocab_size: int, n_embd: int = 128, n_heads: int = 4, n_layers: int = 4, block_size: int = 128, dropout: float = 0.1):
        """Build the embedding tables, transformer blocks, final LayerNorm and output head."""
        super().__init__()
        self.block_size = block_size
        
        # Token and position embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_heads, block_size, dropout) for _ in range(n_layers)
        ])
        
        # Final layer normalization
        self.ln_f = nn.LayerNorm(n_embd)
        # Output language modeling head
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute logits (B, T, vocab) and, if ``targets`` is given, the cross-entropy loss."""
        B, T = idx.shape

        # idx and targets are both (B, T) tensors of integers
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        # Learned position embedding; adding it gives the model word-order information.
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, C)
        
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)     # (B, T, C)
        x = self.ln_f(x)       # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            # cross_entropy expects (N, classes) and (N,), so flatten the batch and time axes.
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None
    ) -> torch.Tensor:
        """
        Generate new tokens given a prompt context tensor idx of shape (B, T).
        Uses temperature and top-k filtering.
        """
        # Autoregressive loop: sample one token, append, repeat.
        for _ in range(max_new_tokens):
            # Crop context idx to block_size if it exceeds limits
            # The position table only has block_size rows, so feed at most the last block_size tokens.
            idx_cond = idx[:, -self.block_size:]
            
            # Forward pass to get logits for the last token
            logits, _ = self(idx_cond)
            # Focus only on the last time step
            # Only the final position predicts the next token; dividing by temperature sharpens (<1) or flattens (>1) the distribution.
            logits = logits[:, -1, :] / max(temperature, 1e-5)  # (B, vocab_size)
            
            # Optional top-k filtering
            # Top-k: keep only the k best candidates by setting the rest to -inf before softmax.
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
            
        return idx
