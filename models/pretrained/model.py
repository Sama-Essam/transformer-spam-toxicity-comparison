"""
Pretrained Hugging Face Transformer Classifier (Model B).
Implemented by Member 2 using AutoModelForSequenceClassification and AutoTokenizer.
"""

import os
from typing import Optional, Dict, Any, Union
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedTokenizer,
    PreTrainedModel
)


class PretrainedTransformerClassifier(nn.Module):
    """
    Modular wrapper around Hugging Face Pretrained Transformer for sequence classification.
    Defaults to 'distilbert-base-uncased' for an optimal balance of parameter efficiency,
    fast fine-tuning, and robust contextual representations.
    """
    def __init__(
        self,
        model_name_or_path: str = "distilbert-base-uncased",
        num_labels: int = 2,
        id2label: Optional[Dict[int, str]] = None,
        label2id: Optional[Dict[str, int]] = None,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.num_labels = num_labels

        if id2label is None:
            id2label = {0: "Ham", 1: "Spam"}
        if label2id is None:
            label2id = {"Ham": 0, "Spam": 1}

        self.id2label = id2label
        self.label2id = label2id

        # Load or initialize config
        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
            classifier_dropout=dropout_rate,
            finetuning_task="text-classification"
        )

        # Load backbone + sequence classification head
        self.model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            config=self.config
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Any:
        """
        Forward pass delegating to Hugging Face model.
        Returns SequenceClassifierOutput containing loss (if labels provided) and logits.
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

    def count_parameters(self, trainable_only: bool = True) -> int:
        """
        Returns total number of (trainable) parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def save_pretrained(self, save_directory: str):
        """
        Saves model weights, config, and state dict to the given directory.
        """
        os.makedirs(save_directory, exist_ok=True)
        self.model.save_pretrained(save_directory)
        self.config.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls,
        save_directory: str,
        num_labels: int = 2,
        **kwargs
    ) -> "PretrainedTransformerClassifier":
        """
        Factory method to reload a saved fine-tuned checkpoint.
        """
        return cls(model_name_or_path=save_directory, num_labels=num_labels, **kwargs)
