# Data Directory

This directory stores raw and processed data splits:
- `raw/`: Raw downloaded dataset files (e.g. UCI SMS Spam or Jigsaw toxicity).
- `processed/`: Intermediate cleaned data.
- `train.csv`: Stratified training split (fit tokenizer and train models strictly on this).
- `val.csv`: Stratified validation split (for hyperparameter tuning and model checkpointing).
- `test.csv`: Held-out test split (strictly for final evaluation; identical across Member 1 and Member 2).
- `dataset_metadata.json`: Full dataset citation, origin, license, and class distribution metadata.
