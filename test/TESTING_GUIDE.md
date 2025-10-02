# Testing Guide for Protein Classification Module

## Overview

Comprehensive pytest suite for validating data pipelines, ensuring data integrity, and preventing common ML errors.

## Installation

```bash
# Install testing dependencies
uv sync --extra dev
```

## Running Tests

### Quick Commands

```bash
# Run all tests (verbose)
uv run pytest src/protein/test_data_pipeline.py -v

# Run with output capture disabled (see print statements)
uv run pytest src/protein/test_data_pipeline.py -v -s

# Run only tests that use mock data (fast)
uv run pytest src/protein/test_data_pipeline.py -k "not real" -v

# Run specific test class
uv run pytest src/protein/test_data_pipeline.py::TestDataLoader -v

# Run single test
uv run pytest src/protein/test_data_pipeline.py::TestDataLoader::test_label_encoding -v

# Generate coverage report
uv run pytest src/protein/test_data_pipeline.py --cov=src/protein --cov-report=term-missing
uv run pytest src/protein/test_data_pipeline.py --cov=src/protein --cov-report=html
```

## Test Structure

### 1. Data Structure Tests (`TestDataStructure`)

Validates that input CSVs have required columns and valid values.

**Key Tests:**
- `test_mock_data_structure`: Mock data fixture is correct
- `test_real_train_data_structure`: Real training CSV has required columns
- `test_real_test_data_structure`: Real test CSV has required columns
- `test_diagnosis_labels`: Labels are only {'AD', 'CN'}

**Why Important:**
- Catches missing columns early
- Ensures downstream code can safely access expected columns
- Validates data integrity before expensive preprocessing

### 2. Data Loader Tests (`TestProteinDataLoader`)

Validates preprocessing pipeline correctness.

**Key Tests:**
- `test_feature_extraction_excludes_metadata`: RID, VISCODE, research_group, subject_age excluded
- `test_label_encoding`: AD=1, CN=0 (not reversed!)
- `test_feature_scaling`: StandardScaler applied (mean≈0, std≈1)
- `test_missing_value_handling`: NaN values imputed with median
- `test_zero_variance_removal`: Constant features removed

**Why Important:**
- **Prevents label leakage**: Metadata columns must be excluded from features
- **Ensures correct encoding**: Swapped labels would invert all predictions
- **Validates scaling**: Unscaled features hurt model performance
- **Handles missing data**: NaN values crash models

### 3. Train/Test Split Tests (`TestTrainTestSplit`)

Validates split properties and prevents data leakage.

**Key Tests:**
- `test_no_data_leakage`: No RID overlap between train/test
- `test_diagnosis_balance_preserved`: AD/CN ratio similar in train/test
- `test_age_distribution_similar`: Age distribution balanced
- `test_split_ratio`: Split is approximately 85/15

**Why Important:**
- **Data leakage is catastrophic**: Overlapping subjects inflate test metrics
- **Class imbalance**: Unbalanced splits bias evaluation
- **Distribution shift**: Age mismatch affects generalization
- **Reproducibility**: Validates documented split ratio

### 4. Cross-Validation Tests (`TestCrossValidation`)

Validates CV fold properties.

**Key Tests:**
- `test_cv_fold_integrity`: No overlap, all samples in validation once
- `test_cv_stratification`: Class balance maintained across folds

**Why Important:**
- **Fold overlap**: Would leak information and inflate CV scores
- **Missing samples**: Biases CV estimates
- **Class imbalance**: Non-stratified CV gives unreliable estimates

### 5. Model Input/Output Tests (`TestModelInputOutput`)

Validates model interfaces.

**Key Tests:**
- `test_classifier_input_shape`: Models accept preprocessed data
- `test_predict_proba_output`: Probabilities are valid [0,1] and sum to 1

**Why Important:**
- **Shape mismatches**: Catch dimension errors before training
- **Invalid probabilities**: NaN or out-of-range probabilities break AUC calculation
- **Interface contracts**: Ensures sklearn API compliance

### 6. Integration Tests (`TestIntegration`)

End-to-end pipeline tests.

**Key Tests:**
- `test_full_pipeline_mock_data`: Full pipeline works with mock data
- `test_real_data_pipeline`: Full pipeline works with real data

**Why Important:**
- **Catches integration issues**: Components may work individually but fail together
- **Validates assumptions**: Real data may violate assumptions tested with mock data
- **Smoke testing**: Quick check that nothing is catastrophically broken

## Common Test Patterns

### Test Fixtures

```python
@pytest.fixture
def mock_protein_data():
    """Reusable mock data for fast tests"""
    # Returns DataFrame with known structure

@pytest.fixture
def real_train_data():
    """Load real data if available, else skip test"""
    if not Path("data.csv").exists():
        return None
    return pd.read_csv("data.csv")
```

### Conditional Skipping

```python
def test_real_data_feature(self, real_train_data):
    if real_train_data is None:
        pytest.skip("Real data not available")
    # Test logic...
```

### Assertions

```python
# Exact equality
assert X.shape == (100, 50)

# Set membership
assert set(y) == {0, 1}

# Approximate equality (for floats)
assert np.allclose(mean, 0, atol=1e-10)

# Informative error messages
assert len(overlap) == 0, f"Found {len(overlap)} overlapping subjects"
```

## What to Test When Adding Features

### Adding New Metadata Column

```python
def test_new_metadata_excluded(self, mock_data):
    """Test that new metadata column is excluded from features"""
    loader = ProteinDataLoader("data.csv")
    X, y = loader.prepare_features(mock_data, fit=True)
    assert X.shape[1] == expected_n_features
```

### Adding New Preprocessing Step

```python
def test_new_preprocessing_applied(self):
    """Test that new preprocessing is correctly applied"""
    # Create data that needs preprocessing
    # Apply preprocessing
    # Assert expected transformation occurred
```

### Adding New Model

```python
def test_new_model_interface(self):
    """Test that new model follows sklearn interface"""
    clf = NewModel()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    assert y_proba.shape == (len(X_test), 2)
```

## Debugging Failed Tests

### Test Fails: "Cannot convert ['bl' 'bl'...] to numeric"

**Cause**: VISCODE column not excluded from features

**Fix**: Add 'VISCODE' to exclude_cols in `dataset.py`

### Test Fails: "Data leakage: N subjects in both train and test"

**Cause**: RID overlap between splits

**Fix**: Regenerate train/test split with proper RID-based splitting

### Test Fails: "Label encoding mismatch"

**Cause**: AD/CN encoding flipped

**Fix**: Check label encoding logic in `prepare_features()`

### Test Fails: "Features not centered"

**Cause**: StandardScaler not applied or applied incorrectly

**Fix**: Verify `scaler.fit_transform()` called in training, `scaler.transform()` in test

## Best Practices

1. **Run tests before committing**: Catch bugs early
2. **Add tests for bugs**: When you fix a bug, add a test to prevent regression
3. **Test edge cases**: Empty data, single class, missing values, etc.
4. **Use fixtures**: Reduce code duplication
5. **Descriptive test names**: `test_label_encoding_ad_is_1_cn_is_0`
6. **Informative assertions**: Include context in error messages

## Continuous Integration

Add to `.github/workflows/test.yml`:

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      - name: Install dependencies
        run: uv sync --extra dev
      - name: Run tests
        run: uv run pytest src/protein/test_data_pipeline.py -v
```

## Expected Test Output

```
======================== test session starts ========================
src/protein/test_data_pipeline.py::TestDataStructure::test_mock_data_structure PASSED
src/protein/test_data_pipeline.py::TestDataStructure::test_real_train_data_structure PASSED
src/protein/test_data_pipeline.py::TestDataLoader::test_feature_extraction_excludes_metadata PASSED
src/protein/test_data_pipeline.py::TestDataLoader::test_label_encoding PASSED
src/protein/test_data_pipeline.py::TestDataLoader::test_feature_scaling PASSED
src/protein/test_data_pipeline.py::TestTrainTestSplit::test_no_data_leakage PASSED
src/protein/test_data_pipeline.py::TestTrainTestSplit::test_diagnosis_balance_preserved PASSED
src/protein/test_data_pipeline.py::TestCrossValidation::test_cv_fold_integrity PASSED
src/protein/test_data_pipeline.py::TestCrossValidation::test_cv_stratification PASSED
src/protein/test_data_pipeline.py::TestModelInputOutput::test_classifier_input_shape PASSED
src/protein/test_data_pipeline.py::TestModelInputOutput::test_predict_proba_output PASSED
src/protein/test_data_pipeline.py::TestIntegration::test_full_pipeline_mock_data PASSED
src/protein/test_data_pipeline.py::TestIntegration::test_real_data_pipeline PASSED

====================== 13 passed in 5.23s ======================
```
