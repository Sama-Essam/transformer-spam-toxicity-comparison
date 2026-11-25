"""
Pretrained Transformer Module (Model B) - Member 2 Deliverable.
Exposes Model B architecture, checkpoint loading, and inference pipeline.
"""

import os
import sys

# Ensure root is in path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.pretrained.model import PretrainedTransformerClassifier
from models.pretrained.inference import PretrainedModelPredictor, predict_pretrained

__all__ = [
    "PretrainedTransformerClassifier",
    "PretrainedModelPredictor",
    "predict_pretrained"
]


if __name__ == "__main__":
    sample_text = "Congratulations! You won a $1,000 Walmart Gift Card. Click here to claim your prize."
    print("Testing Model B Standalone Predictor:")
    print(f"Sample Input: '{sample_text}'")
    try:
        result = predict_pretrained(sample_text)
        print(f"Prediction Result:\n{result}")
    except Exception as e:
        print(f"Prediction test note: {e}")
