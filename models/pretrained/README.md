# Member 2 Integration Guide: Pretrained Transformer (Model B)

Hello Member 2! This directory is reserved for your implementation of **Model B (Pretrained Transformer Classifier)**.

## Guidelines & Alignment
1. **Data Splits:**
   - Consume the exact splits in `data/train.csv`, `data/val.csv`, and `data/test.csv` prepared by Member 1.
   - Do **NOT** modify or resplit `data/test.csv`; both models must be benchmarked on the exact same test instances.
2. **Model Choice:**
   - Recommended lightweight models: `distilbert-base-uncased`, `bert-base-uncased`, or `roberta-base`.
3. **Inference Contract:**
   - Adhere strictly to the interface defined in `models/pretrained/inference.py`. Member 3 relies on `predict(text: str) -> dict` returning matching keys (`model_name`, `label`, `confidence`, `probabilities`, `latency_ms`).
4. **Evaluation Metrics:**
   - Save your final test metrics to `artifacts/model_b_metrics.json` following the schema in `artifacts/README.md`.
