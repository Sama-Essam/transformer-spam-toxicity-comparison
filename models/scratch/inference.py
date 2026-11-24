"""
Standardized Inference Interface for Model A (Member 1 -> Member 3 Handoff).

Exposes ScratchModelPredictor:
- Loads artifacts/model_a_scratch.pt and artifacts/scratch_tokenizer.json
- Runs deterministic inference with latency profiling
- Returns standardized dict schema for Member 3's Streamlit dashboard
"""

import time
import os
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F

from .transformer import TransformerClassifier
from .tokenizer import SimpleTokenizer


class ScratchModelPredictor:
    """
    Standardized inference wrapper for Model A (Custom Transformer from Scratch).
    Provides a clean, uniform interface for Member 3's Streamlit application.
    """
    LABEL_MAP = {0: "Ham", 1: "Spam"}

    def __init__(
        self,
        model_path: str = "artifacts/model_a_scratch.pt",
        tokenizer_path: str = "artifacts/scratch_tokenizer.json",
        device: Optional[str] = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 1. Load Tokenizer
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(
                f"Tokenizer artifact not found at '{tokenizer_path}'. "
                "Ensure Phase 1 training or data preparation has been executed."
            )
        self.tokenizer = SimpleTokenizer.load(tokenizer_path)

        # 2. Load Model Checkpoint
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model checkpoint not found at '{model_path}'. "
                "Ensure Phase 1 training has been completed."
            )
        checkpoint = torch.load(model_path, map_location=self.device)

        model_kwargs = checkpoint.get("model_kwargs", {
            "vocab_size": self.tokenizer.vocab_size,
            "d_model": 128,
            "n_heads": 4,
            "n_layers": 2,
            "d_ff": 256,
            "max_len": self.tokenizer.max_length,
            "num_classes": 2,
            "dropout": 0.1,
            "pad_idx": self.tokenizer.pad_id,
            "pooling": "mean"
        })

        self.model = TransformerClassifier(**model_kwargs)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.model_name = "Model A (Custom Transformer - From Scratch)"
        self.trainable_params = self.model.count_parameters()

    @torch.no_grad()
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Runs single-sample inference on a raw text string with latency benchmarking.

        Args:
            text: Raw input string (e.g. SMS message or comment).

        Returns:
            Dictionary matching the standardized handoff schema:
            {
                "model_name": str,
                "label": str,          # "Ham" or "Spam"
                "confidence": float,   # Probability of winning class [0.0 - 1.0]
                "probabilities": dict, # {"Ham": float, "Spam": float}
                "latency_ms": float    # Single-sample inference latency in milliseconds
            }
        """
        # Profile start
        start_time = time.perf_counter()

        # Tokenize and tensorize
        input_ids, attention_mask = self.tokenizer.encode(text)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        mask_tensor = torch.tensor([attention_mask], dtype=torch.long, device=self.device)

        # Forward pass
        logits = self.model(input_tensor, attention_mask=mask_tensor)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().tolist()

        # Latency measurement
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Class probabilities
        prob_dict = {
            self.LABEL_MAP[0]: round(probs[0], 4),
            self.LABEL_MAP[1]: round(probs[1], 4)
        }

        predicted_id = 1 if probs[1] >= probs[0] else 0
        predicted_label = self.LABEL_MAP[predicted_id]
        confidence = round(probs[predicted_id], 4)

        return {
            "model_name": self.model_name,
            "label": predicted_label,
            "confidence": confidence,
            "probabilities": prob_dict,
            "latency_ms": round(latency_ms, 2)
        }
