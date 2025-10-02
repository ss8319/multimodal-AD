# Model Training Runs

This directory stores timestamped training runs when using the `--save-models` flag.

## Directory Structure

Each run creates a timestamped directory with the following structure:

```
run_YYYYMMDD_HHMMSS/
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── svm_rbf.pkl
│   ├── gradient_boosting.pkl
│   ├── neural_network.pkl
│   └── protein_transformer.pth
├── metadata.json
└── results_summary.csv
```

## Files

- **`models/`**: Trained model files
  - `.pkl` files: sklearn models (pickle format)
  - `.pth` files: PyTorch models (state dict format)
- **`metadata.json`**: Run metadata (timestamp, model names, best model info)
- **`results_summary.csv`**: CV and test set results for all models

## Usage

### Saving Models

```bash
# Train and save all models
uv run python src/protein/main.py --save-models

# With custom data
uv run python src/protein/main.py \
    --train-data path/to/train.csv \
    --test-data path/to/test.csv \
    --save-models
```

### Loading Models

#### Sklearn Models
```python
import pickle

# Load model
with open('src/protein/runs/run_20240115_143022/models/logistic_regression.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
predictions = model.predict(X_new)
```

#### PyTorch Models
```python
import torch
from src.protein.model import ProteinTransformer

# Load checkpoint
checkpoint = torch.load('src/protein/runs/run_20240115_143022/models/protein_transformer.pth')

# Recreate model with saved config
model = ProteinTransformer(**checkpoint['model_config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions
with torch.no_grad():
    X_tensor = torch.FloatTensor(X_new)
    logits = model(X_tensor)
    predictions = torch.argmax(logits, dim=1).numpy()
```

## Notes

- All models are trained on the **full training set** (not individual CV folds)
- Models are saved **after** cross-validation completes
- The scaler is **not** saved separately - you must preprocess new data the same way as training data
- Run directories are gitignored by default

