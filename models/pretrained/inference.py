"""
Standardized Inference Interface for Model B (Member 2 -> Member 3 Handoff).

Exposes PretrainedModelPredictor:
- Loads fine-tuned checkpoint and tokenizer from artifacts/model_b_pretrained/
- Runs deterministic inference with latency profiling
- Returns standardized dict schema for Member 3's Streamlit dashboard
- Provides standalone predict_pretrained(text) function
"""

import os
import sys
import time
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Ensure models can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from models.pretrained.model import PretrainedTransformerClassifier


class PretrainedModelPredictor:
    """
    Standardized inference wrapper for Model B (Fine-tuned Pretrained Transformer).
    Conforms strictly to the uniform prediction schema expected by Member 3's Streamlit app.
    """
    LABEL_MAP = {0: "Ham", 1: "Spam"}

    def __init__(
        self,
        model_path_or_name: str = "artifacts/model_b_pretrained",
        tokenizer_path_or_name: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 128
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.max_length = max_length
        self.model_path_or_name = model_path_or_name

        # If fine-tuned artifact directory doesn't exist, fallback to base checkpoint
        if not os.path.exists(model_path_or_name):
            fallback_base = "models/pretrained/base_distilbert"
            if os.path.exists(fallback_base):
                model_path_or_name = fallback_base
            else:
                model_path_or_name = "distilbert-base-uncased"
            self.model_path_or_name = model_path_or_name

        tok_path = tokenizer_path_or_name or model_path_or_name
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path)

        # Load Pretrained Classifier
        self.model_wrapper = PretrainedTransformerClassifier(
            model_name_or_path=model_path_or_name,
            num_labels=2,
            id2label=self.LABEL_MAP,
            label2id={"Ham": 0, "Spam": 1}
        )
        self.model = self.model_wrapper.model
        self.model.to(self.device)
        self.model.eval()

        self.model_name = "Model B (Pretrained Transformer - DistilBERT)"
        self.trainable_params = self.model_wrapper.count_parameters()

    @torch.no_grad()
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Runs single-sample inference on a raw text string with precise latency benchmarking.

        Args:
            text: Raw input string (e.g. SMS message or toxic text).

        Returns:
            Standardized dictionary:
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
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # Forward pass
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
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


# Global cached predictor for standalone fast reuse
_cached_pretrained_predictor: Optional[PretrainedModelPredictor] = None


def predict_pretrained(
    text: str,
    model_path: str = "artifacts/model_b_pretrained",
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Clean, reusable standalone inference function for Model B.
    Returns predicted label, class probabilities, confidence, and latency (ms).
    """
    global _cached_pretrained_predictor
    if _cached_pretrained_predictor is None or _cached_pretrained_predictor.model_path_or_name != model_path:
        _cached_pretrained_predictor = PretrainedModelPredictor(
            model_path_or_name=model_path,
            device=device
        )
    return _cached_pretrained_predictor.predict(text)
