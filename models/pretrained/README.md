# Model B: Pretrained Transformer Classifier (Member 2 Deliverable)

This directory contains the implementation, architecture, and inference interface for **Model B (Fine-Tuned Pretrained Transformer)**.

## Selected Checkpoint: `distilbert-base-uncased`
- **Parameters:** 66,955,010 (~67M parameters, 6 layers, 12 attention heads, 768 hidden dimensions)
- **Pretrained Corpus:** English Wikipedia (2,500M words) + Toronto BookCorpus (800M words)
- **Tokenizer:** WordPiece subword tokenizer (30,522 vocab items)

## Directory Structure
- `model.py`: `PretrainedTransformerClassifier` wrapping Hugging Face `AutoModelForSequenceClassification` and `AutoConfig`.
- `inference.py`: Standardized inference class `PretrainedModelPredictor` and standalone function `predict_pretrained(text)`.
- `base_distilbert/`: Local base weights and tokenizer configuration for offline reproducibility.

## How to Run Inference (Member 3 Integration)
```python
from models.pretrained.inference import PretrainedModelPredictor, predict_pretrained

# Option 1: Standalone function
result = predict_pretrained("URGENT! You have won a £1,000 cash prize. Call 09061701461 to claim.")
print(result)

# Option 2: Object-oriented predictor
predictor = PretrainedModelPredictor(model_path_or_name="artifacts/model_b_pretrained")
print(predictor.predict("Hey, are we still meeting today at 3pm?"))
```

### Standardized Output Schema
```json
{
  "model_name": "Model B (Pretrained Transformer - DistilBERT)",
  "label": "Spam",
  "confidence": 0.9984,
  "probabilities": {
    "Ham": 0.0016,
    "Spam": 0.9984
  },
  "latency_ms": 67.59
}
```

## How to Fine-Tune
```bash
python scripts/train_pretrained.py --epochs 2 --batch_size 32 --lr 3e-5 --seed 42
```
Outputs are saved directly to `artifacts/model_b_pretrained/`.
