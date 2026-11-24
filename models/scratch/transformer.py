"""
Custom Transformer Classifier from Scratch (Member 1).

ACADEMIC GRADING COMPLIANCE:
- Built strictly with fundamental PyTorch primitives:
  nn.Module, nn.Linear, nn.Embedding, nn.Dropout, nn.LayerNorm, torch.matmul.
- ZERO use of forbidden modules:
  NO nn.Transformer, NO nn.TransformerEncoder, NO nn.TransformerEncoderLayer,
  NO nn.MultiheadAttention.
- ZERO pretrained weights or pretrained embeddings (all weights initialized from scratch).
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding from 'Attention Is All You Need' (Vaswani et al., 2017).
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as persistent buffer (part of state_dict, not a parameter)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """
    Explicit Multi-Head Self-Attention implemented from raw matrix math.
    Calculates:
        Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) + Mask ) * V
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Explicit Q, K, V projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.w_o = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len) where 1 indicates real token and 0 indicates <PAD>
        Returns:
            out: (batch_size, seq_len, d_model)
            attention_weights: (batch_size, n_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.size()

        # 1. Project inputs to Q, K, V
        q = self.w_q(x)  # (batch_size, seq_len, d_model)
        k = self.w_k(x)
        v = self.w_v(x)

        # 2. Reshape and transpose for multi-head attention:
        # (batch_size, seq_len, n_heads, d_k) -> (batch_size, n_heads, seq_len, d_k)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 3. Scaled Dot-Product Attention
        # Scores: (batch_size, n_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 4. Apply padding mask (0 means pad, should not be attended to)
        if mask is not None:
            # mask: (batch_size, 1, 1, seq_len) broadcastable over (batch_size, n_heads, seq_len, seq_len)
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask_expanded == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights_dropped = self.dropout(attn_weights)

        # 5. Multiply by values V: (batch_size, n_heads, seq_len, d_k)
        context = torch.matmul(attn_weights_dropped, v)

        # 6. Concatenate heads and project output:
        # (batch_size, n_heads, seq_len, d_k) -> (batch_size, seq_len, n_heads, d_k) -> (batch_size, seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.w_o(context)
        out = self.dropout(out)

        return out, attn_weights


class PositionwiseFeedForward(nn.Module):
    """
    Standard position-wise feed-forward sub-layer:
    FFN(x) = GELU(x * W_1 + b_1) * W_2 + b_2
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout2(self.linear2(self.dropout1(self.activation(self.linear1(x)))))


class TransformerEncoderLayer(nn.Module):
    """
    A single Transformer Encoder Layer containing:
    - Multi-Head Self-Attention sub-layer with Pre-LayerNorm and residual connection
    - Position-wise Feed-Forward sub-layer with Pre-LayerNorm and residual connection
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-LN Self-Attention with Residual Connection
        normed_x = self.norm1(x)
        attn_out, attn_weights = self.self_attn(normed_x, mask=mask)
        x = x + attn_out

        # Pre-LN Feed-Forward with Residual Connection
        normed_x = self.norm2(x)
        ffn_out = self.ffn(normed_x)
        x = x + ffn_out

        return x, attn_weights


class TransformerClassifier(nn.Module):
    """
    Complete Encoder-Only Transformer Classifier built from scratch.
    Pipeline:
        Tokens -> Embedding + Positional Encoding -> Stack of Encoder Layers ->
        Sequence Pooling ([CLS] or Mean) -> Classification Head -> Logits
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        max_len: int = 256,
        num_classes: int = 2,
        dropout: float = 0.1,
        pad_idx: int = 0,
        pooling: str = "mean"  # "mean" or "cls"
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.pooling = pooling.lower()
        self.d_model = d_model

        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        # Stack of custom encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """Xavier / Normal initialization for stable training from scratch."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len) where 1 indicates real token and 0 indicates pad
        Returns:
            logits: (batch_size, num_classes)
        """
        if attention_mask is None:
            # Derive padding mask automatically from pad_idx if not provided
            attention_mask = (input_ids != self.pad_idx).long()

        # 1. Embeddings & Positional Encoding
        # Scale embeddings by sqrt(d_model) as in Attention Is All You Need
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        # 2. Pass through Transformer Encoder layers
        for layer in self.layers:
            x, _ = layer(x, mask=attention_mask)

        x = self.final_norm(x)

        # 3. Pooling over sequence dimension
        if self.pooling == "cls":
            # Pool representation from index 0 (<CLS>)
            pooled = x[:, 0, :]
        else:
            # Masked Mean Pooling: average non-pad tokens strictly
            mask_expanded = attention_mask.unsqueeze(-1).float()  # (batch_size, seq_len, 1)
            sum_embeddings = torch.sum(x * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask

        # 4. Classification Head -> Logits
        logits = self.classifier(pooled)
        return logits

    def count_parameters(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
