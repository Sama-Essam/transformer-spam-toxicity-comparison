"""
Custom PyTorch Transformer from Scratch (Member 1).
STRICT RULE: Only use basic PyTorch primitives (nn.Module, nn.Linear, nn.Embedding, nn.Dropout, nn.LayerNorm).
DO NOT use torch.nn.Transformer or pretrained embeddings.
"""

import torch
import torch.nn as nn
import math


class TransformerClassifier(nn.Module):
    """
    Placeholder stub for Custom Transformer Classifier (Member 1).
    Full implementation to be completed in Phase 1.
    """
    def __init__(self, vocab_size: int = 10000, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, d_ff: int = 256, max_len: int = 256,
                 num_classes: int = 2, dropout: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.pad_idx = pad_idx
        # Full architecture components will be instantiated here in Phase 1

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError("Model A architecture will be implemented in Phase 1.")
