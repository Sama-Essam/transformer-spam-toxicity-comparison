"""
Custom Tokenizer from Scratch (Member 1).
Word-level tokenizer with regex tokenization, frequency thresholding,
special tokens (<PAD>, <UNK>, <CLS>, <SEP>), and zero data leakage
(strictly fitted on train.csv).
"""

import os
import re
import json
import csv
from typing import List, Dict, Optional, Tuple, Union


class SimpleTokenizer:
    """
    A standalone word-level tokenizer built from scratch.
    - Tokenizes text using regex word & punctuation splitting.
    - Builds a fixed-size vocabulary from training text only.
    - Encodes text into padded/truncated token IDs and attention masks.
    - Saves and loads tokenizer state to/from JSON.
    """
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    CLS_TOKEN = "<CLS>"
    SEP_TOKEN = "<SEP>"

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        max_length: int = 128,
        lower: bool = True
    ):
        self.lower = lower
        self.max_length = max_length
        self.pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)

        if vocab:
            self.word2idx = dict(vocab)
            self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        else:
            self.word2idx = {
                self.PAD_TOKEN: 0,
                self.UNK_TOKEN: 1,
                self.CLS_TOKEN: 2,
                self.SEP_TOKEN: 3,
            }
            self.idx2word = {v: k for k, v in self.word2idx.items()}

    @property
    def pad_id(self) -> int:
        return self.word2idx[self.PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.word2idx[self.UNK_TOKEN]

    @property
    def cls_id(self) -> int:
        return self.word2idx[self.CLS_TOKEN]

    @property
    def sep_id(self) -> int:
        return self.word2idx[self.SEP_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

    def tokenize(self, text: str) -> List[str]:
        """Splits raw text into tokens."""
        if not isinstance(text, str):
            text = str(text)
        if self.lower:
            text = text.lower()
        return self.pattern.findall(text)

    def fit(self, texts: List[str], max_vocab_size: int = 8000, min_freq: int = 2):
        """
        Fits vocabulary strictly on the provided list of texts (train.csv).
        Special tokens occupy indices 0, 1, 2, 3.
        """
        freqs: Dict[str, int] = {}
        for text in texts:
            for token in self.tokenize(text):
                freqs[token] = freqs.get(token, 0) + 1

        # Sort by frequency descending, then alphabetically for deterministic order
        sorted_tokens = sorted(freqs.items(), key=lambda x: (-x[1], x[0]))

        # Filter by min_freq and limit by max_vocab_size
        filtered_tokens = [tok for tok, count in sorted_tokens if count >= min_freq]
        allowed_vocab_words = filtered_tokens[: max(0, max_vocab_size - 4)]

        # Reset vocabulary preserving special tokens
        self.word2idx = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.CLS_TOKEN: 2,
            self.SEP_TOKEN: 3,
        }
        for idx, token in enumerate(allowed_vocab_words, start=4):
            self.word2idx[token] = idx

        self.idx2word = {v: k for k, v in self.word2idx.items()}
        print(f"Tokenizer fitted: vocab_size={len(self.word2idx)} (filtered from {len(freqs)} unique tokens)")

    def fit_from_csv(self, csv_path: str, text_col: str = "text", max_vocab_size: int = 8000, min_freq: int = 2):
        """Reads train.csv and fits vocabulary strictly on training samples."""
        texts = []
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if text_col in row and row[text_col]:
                    texts.append(row[text_col])
        self.fit(texts, max_vocab_size=max_vocab_size, min_freq=min_freq)

    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        add_special_tokens: bool = True,
        truncation: bool = True,
        padding: bool = True
    ) -> Tuple[List[int], List[int]]:
        """
        Encodes a string into token IDs and an attention mask.
        Returns:
            input_ids: List[int]
            attention_mask: List[int] (1 for real tokens, 0 for padding)
        """
        max_len = max_length if max_length is not None else self.max_length
        tokens = self.tokenize(text)

        if add_special_tokens:
            # Leave space for <CLS> and <SEP>
            effective_len = max_len - 2 if truncation else len(tokens)
            tokens = tokens[:effective_len]
            token_ids = [self.cls_id] + [self.word2idx.get(t, self.unk_id) for t in tokens] + [self.sep_id]
        else:
            if truncation:
                tokens = tokens[:max_len]
            token_ids = [self.word2idx.get(t, self.unk_id) for t in tokens]

        # Truncate if still over
        if truncation and len(token_ids) > max_len:
            token_ids = token_ids[:max_len]

        seq_len = len(token_ids)
        attention_mask = [1] * seq_len

        if padding and seq_len < max_len:
            pad_amount = max_len - seq_len
            token_ids.extend([self.pad_id] * pad_amount)
            attention_mask.extend([0] * pad_amount)

        return token_ids, attention_mask

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Converts token IDs back to a readable string."""
        tokens = []
        specials = {self.pad_id, self.unk_id, self.cls_id, self.sep_id}
        for tid in token_ids:
            if skip_special_tokens and tid in specials:
                continue
            tokens.append(self.idx2word.get(tid, self.UNK_TOKEN))
        return " ".join(tokens)

    def save(self, filepath: str):
        """Serializes tokenizer configuration and vocabulary to JSON."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "vocab": self.word2idx,
            "max_length": self.max_length,
            "lower": self.lower,
            "special_tokens": {
                "pad_token": self.PAD_TOKEN,
                "unk_token": self.UNK_TOKEN,
                "cls_token": self.CLS_TOKEN,
                "sep_token": self.SEP_TOKEN,
                "pad_id": self.pad_id,
                "unk_id": self.unk_id,
                "cls_id": self.cls_id,
                "sep_id": self.sep_id
            }
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Tokenizer saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "SimpleTokenizer":
        """Loads a saved tokenizer from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(
            vocab=data.get("vocab", {}),
            max_length=data.get("max_length", 128),
            lower=data.get("lower", True)
        )
        return tok
