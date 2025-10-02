# Protein Classification Module

Clean, modular implementation for AD/CN classification from proteomics data.

## Structure

```
src/protein/
├── __init__.py          # Package exports
├── dataset.py           # Data loading and preprocessing
├── model.py             # Model definitions (sklearn + PyTorch)
├── utils.py             # Evaluation and CV utilities
├── main.py              # Main experiment script
└── README.md            # This file
```

## Quick Start

### Basic Usage

```bash
# Run with defaults
uv run python src/protein/main.py

# Custom data paths
uv run python src/protein/main.py \
  --train-data 'src\data\protein\proteomic_encoder_train.csv' \
  --test-data 'src\data\protein\proteomic_encoder_test.csv'

# Change CV folds
uv run python src/protein/main.py --n-folds 10
```

### Command Line Arguments

- `--train-data`: Path to training CSV (default: `src/data/protein/proteomic_train_set.csv`)
- `--test-data`: Path to test CSV (default: `src/data/protein/proteomic_mri_with_labels.csv`)
- `--label-col`: Label column name (default: `research_group`)
- `--id-col`: Subject ID column (default: `RID`)
- `--n-folds`: Number of CV folds (default: 5)
- `--random-state`: Random seed (default: 42)

## Module Components

### `dataset.py` - Data Management

**Classes:**
- `ProteinDataset`: PyTorch Dataset wrapper
- `ProteinDataLoader`: Handles loading, preprocessing, and splitting

**Key Features:**
- Automatic feature extraction (excludes metadata columns)
- Zero-variance feature removal
- Label encoding (AD=1, CN=0)
- StandardScaler normalization
- Train/test split handling

**Usage:**
```python
from dataset import ProteinDataLoader

loader = ProteinDataLoader(
    data_path="path/to/data.csv",
    label_col="research_group",
    id_col="RID"
)

X_train, y_train, X_test, y_test, train_df = loader.get_train_test_split(
    train_path="train.csv",
    test_path="test.csv"
)
```

### `model.py` - Model Definitions

**Models:**
- Logistic Regression
- Random Forest
- SVM (RBF kernel)
- Gradient Boosting
- Neural Network (MLP)
- Protein Transformer (custom PyTorch)

**Functions:**
- `get_classifiers(random_state=42)`: Returns dict of all models

**Usage:**
```python
from model import get_classifiers

classifiers = get_classifiers(random_state=42)
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
```
### `main.py` - Experiment Orchestration

Clean workflow:
1. Load and preprocess data
2. Setup stratified CV
3. Evaluate all classifiers
4. Print summary
5. Save results

## Output Files

The experiment generates:

- `cv_fold_indices.csv`: CV fold assignments (RID, fold, split_type)
- `cv_fold_detailed_results.pkl`: Per-fold metrics for each classifier
- `classifier_results_summary.csv`: Summary table with mean±std for all models

### Reproducibility

All randomness is controlled by `--random-state`:
- CV fold splitting
- Model initialization
- Data shuffling

Same random state → identical results every run.

## Testing

### Run Tests

```bash
# Install dev dependencies (includes pytest)
uv sync --extra dev

# Run all tests
uv run pytest src/protein/test_data_pipeline.py -v

# Run specific test class
uv run pytest src/protein/test_data_pipeline.py::TestDataStructure -v

# Run with coverage
uv run pytest src/protein/test_data_pipeline.py --cov=src/protein --cov-report=html

# Run tests that don't require real data
uv run pytest src/protein/test_data_pipeline.py -k "not real" -v
```

### Test Coverage

The test suite (`test_data_pipeline.py`) validates:

1. **Data Structure** - Required columns, labels, feature counts
2. **Data Loader** - Feature extraction, label encoding, scaling, missing values
3. **Train/Test Split** - No leakage, balance preservation, age distribution
4. **Cross-Validation** - Fold integrity, stratification, no overlap
5. **Model I/O** - Input shapes, output validity, probability ranges
6. **Integration** - Full pipeline from loading to evaluation

### Key Tests

- `test_feature_extraction_excludes_metadata`: Ensures RID, VISCODE, research_group, subject_age are excluded
- `test_label_encoding`: Validates AD=1, CN=0 encoding
- `test_no_data_leakage`: Confirms no RID overlap between train/test
- `test_diagnosis_balance_preserved`: Checks AD/CN ratio consistency
- `test_cv_stratification`: Validates class balance in CV folds
