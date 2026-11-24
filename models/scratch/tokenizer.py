"""
Custom Tokenizer from Scratch (Member 1).
Trained strictly on train.csv to prevent data leakage.
"""

from typing import List, Dict, Optional
import json


class SimpleTokenizer:
    """
    Placeholder stub for Custom Tokenizer (Member 1).
    Full vocabulary construction and serialization to be completed in Phase 1.
    """
    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        self.vocab = vocab or {}
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.cls_token = "<CLS>"

    def fit(self, texts: List[str], max_vocab_size: int = 10000):
        raise NotImplementedError("Tokenizer fitting will be implemented in Phase 1.")

    def encode(self, text: str, max_length: int = 256) -> List[int]:
        raise NotImplementedError("Tokenizer encoding will be implemented in Phase 1.")

    def save(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab}, f, indent=2)

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(vocab=data.get("vocab", {}))
