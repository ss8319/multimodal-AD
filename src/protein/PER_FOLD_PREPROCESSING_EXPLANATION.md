# Per-Fold Preprocessing in Cross-Validation

## The Critical Issue: Data Leakage in Cross-Validation

### ❌ WRONG: Pre-scaling Before CV (Old Approach)

```python
# Step 1: Fit scaler on FULL training set (e.g., 97 samples)
scaler.fit(X_train)  # Scaler sees ALL data, including validation folds
X_train_scaled = scaler.transform(X_train)

# Step 2: Split into folds
for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X_train_scaled, y_train)):
    # Train on fold
    model.fit(X_train_scaled[train_idx], y_train[train_idx])
    
    # Validate on fold
    predictions = model.predict(X_train_scaled[val_idx])
    # ⚠️ PROBLEM: Validation fold was already "seen" by the scaler!
```

**Why this is wrong:**
1. The scaler was fitted on the **entire** training set (all 97 samples)
2. The validation fold (e.g., 20 samples) **influenced** the scaler's mean/std
3. When we evaluate on the validation fold, it's not truly "unseen" data
4. This creates **data leakage** - information from the validation set leaked into preprocessing

**Consequences:**
- Overly optimistic validation scores
- Model appears to generalize better than it actually does
- Test set performance may be surprisingly different (better or worse)
- Scientific validity is compromised

---

### ✅ CORRECT: Per-Fold Preprocessing (New Approach)

```python
# Step 1: Keep raw (unscaled) training data
X_train_raw = load_raw_data()  # No scaling yet!

# Step 2: For each fold, fit scaler ONLY on training subset
for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X_train_raw, y_train)):
    # Create fresh scaler for this fold
    fold_scaler = StandardScaler()
    
    # Get raw data for this fold
    X_train_fold_raw = X_train_raw[train_idx]  # e.g., 77 samples
    X_val_fold_raw = X_train_raw[val_idx]      # e.g., 20 samples
    
    # Fit scaler ONLY on training fold (validation fold is truly unseen)
    fold_scaler.fit(X_train_fold_raw)  # ✅ Only sees 77 samples
    
    # Transform both with the same scaler
    X_train_fold = fold_scaler.transform(X_train_fold_raw)
    X_val_fold = fold_scaler.transform(X_val_fold_raw)
    
    # Train and evaluate
    model.fit(X_train_fold, y_train[train_idx])
    predictions = model.predict(X_val_fold)
    # ✅ Validation fold is truly unseen - no data leakage!
```

**Why this is correct:**
1. Each fold gets its **own** scaler fitted only on its training subset
2. Validation fold is **never** seen during preprocessing
3. Simulates real-world scenario: preprocess on training, apply to unseen test
4. Prevents data leakage
5. Provides honest estimates of generalization performance

---

## Implementation in Our Code

### Before: Data Leakage
```python
# main.py (OLD)
X_train, y_train, X_test, y_test, train_df = data_loader.get_train_test_split(
    train_path=args.train_data,
    test_path=args.test_data
)  # X_train is already scaled!

# utils.py (OLD)
for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X_train, y_train)):
    fold_clf.fit(X_train[train_idx], y_train[train_idx])
    # ❌ Using pre-scaled X_train - validation fold was seen by scaler
```

### After: Proper Per-Fold Preprocessing
```python
# main.py (NEW)
X_train_raw, y_train, X_test, y_test, train_df = data_loader.get_train_test_split(
    train_path=args.train_data,
    test_path=args.test_data,
    return_raw=True  # ✅ Get raw (unscaled) features
)

# utils.py (NEW)
for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X_train_raw, y_train)):
    # Create fresh scaler for this fold
    fold_scaler = StandardScaler()
    
    # Get raw data for this fold
    X_train_fold_raw = X_train_raw.iloc[train_idx]
    X_val_fold_raw = X_train_raw.iloc[val_idx]
    
    # Fit scaler ONLY on training fold
    fold_scaler.fit(X_train_fold_raw)
    
    # Transform both
    X_train_fold = fold_scaler.transform(X_train_fold_raw)
    X_val_fold = fold_scaler.transform(X_val_fold_raw)
    
    # Train and evaluate
    fold_clf.fit(X_train_fold, y_train[train_idx])
    # ✅ No data leakage!
```

---

## Key Changes Made

### 1. `dataset.py`: Added `get_raw_features()` method
```python
def get_raw_features(self, df):
    """Extract raw features WITHOUT scaling (for per-fold preprocessing)"""
    X = df[self.feature_cols].fillna(df[self.feature_cols].median())
    # Remove zero-variance features
    # Encode labels
    # But DO NOT scale!
    return X, y_encoded  # DataFrame, not scaled array
```

### 2. `dataset.py`: Updated `get_train_test_split()`
```python
def get_train_test_split(self, train_path, test_path=None, return_raw=False):
    if return_raw:
        X_train, y_train = self.get_raw_features(train_df)  # Raw DataFrame
        
        # IMPORTANT: Still fit scaler on full training set for test set transformation
        # This is ONLY used for test set, NOT for CV (each fold gets its own scaler)
        _ = self.scaler.fit(X_train)
    else:
        X_train, y_train = self.prepare_features(train_df, fit=True)  # Scaled array
    
    # Test set is ALWAYS pre-scaled (using full training set statistics)
    X_test, y_test = self.prepare_features(test_df, fit=False)
```

### 3. `utils.py`: Updated `evaluate_model_cv()`
```python
def evaluate_model_cv(clf, X_train_raw, y_train, X_test, y_test, cv_splitter, 
                      clf_name="Model", compute_test_confusion=False, data_loader=None):
    
    use_per_fold_preprocessing = data_loader is not None and hasattr(X_train_raw, 'columns')
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X_train_raw, y_train)):
        if use_per_fold_preprocessing:
            # Per-fold preprocessing
            fold_scaler = StandardScaler()
            X_train_fold_raw = X_train_raw.iloc[train_idx]
            X_val_fold_raw = X_train_raw.iloc[val_idx]
            fold_scaler.fit(X_train_fold_raw)
            X_train_fold = fold_scaler.transform(X_train_fold_raw)
            X_val_fold = fold_scaler.transform(X_val_fold_raw)
        else:
            # Backward compatibility: use pre-scaled data
            X_train_fold = X_train_raw[train_idx]
            X_val_fold = X_train_raw[val_idx]
```

### 4. `main.py`: Enable per-fold preprocessing
```python
# Get raw features
X_train_raw, y_train, X_test, y_test, train_df = data_loader.get_train_test_split(
    train_path=args.train_data,
    test_path=args.test_data,
    return_raw=True  # Enable per-fold preprocessing
)

# Pass raw features and data_loader to enable per-fold preprocessing
result = evaluate_model_cv(
    clf=clf,
    X_train_raw=X_train_raw,  # Raw DataFrame
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    cv_splitter=cv_splitter,
    clf_name=clf_name,
    compute_test_confusion=args.show_confusion,
    data_loader=data_loader  # Enable per-fold preprocessing
)
```

---

## Why Test Performance Can Differ From Validation

With proper per-fold preprocessing, you may observe:

1. **Test performance better than validation:**
   - Test set may be inherently more separable
   - Model trained on full training set (more data) for test evaluation
   - Different data distribution

2. **Test performance worse than validation:**
   - This is now the **honest** scenario
   - Validation was overly optimistic due to data leakage (old approach)
   - Test set is truly unseen

3. **Similar performance:**
   - Ideal scenario
   - Model generalizes well
   - Data is well-distributed

---

## Important: Different Scalers for CV vs Test Set

Our implementation uses **two different scaling strategies**:

### For Cross-Validation (Training Data)
- Each fold gets its **own scaler**
- Fitted only on that fold's training subset
- Prevents data leakage within CV
- Provides honest validation estimates

```python
# Fold 1: Scaler fitted on samples [20-96]
# Fold 2: Scaler fitted on samples [0-19, 40-96]
# Fold 3: Scaler fitted on samples [0-39, 60-96]
# etc.
```

### For Test Set (Hold-out Data)
- Uses a **single scaler** fitted on the **full training set**
- This is the correct approach because:
  1. Test set is completely separate (not part of CV)
  2. In production, we'd scale new data using training set statistics
  3. Simulates real-world deployment scenario

```python
# Test set: Scaler fitted on ALL training samples [0-96]
# This is OK because test set was never part of the training/CV process
```

### Why This Makes Sense

```
Training Set (97 samples)
├─ Cross-Validation: Per-fold scalers (prevents leakage)
│  ├─ Fold 1: Scaler_1 (77 train samples)
│  ├─ Fold 2: Scaler_2 (77 train samples)
│  └─ Fold 3: Scaler_3 (77 train samples)
│
└─ Test Set Evaluation: Single scaler (97 train samples)
   └─ Applied to Test Set (18 samples)
```

**Key Point:** The test set is treated differently because it's:
- Completely held out (never seen during training or CV)
- Representative of production/deployment scenario
- Used to estimate final model performance

---

## Summary

| Aspect | Old Approach ❌ | New Approach ✅ |
|--------|----------------|----------------|
| **Scaler fitted on** | Full training set (97 samples) | Each fold's training subset (77 samples) |
| **Validation fold** | Seen by scaler | Truly unseen |
| **Data leakage** | Yes | No |
| **Validation scores** | Overly optimistic | Honest |
| **Scientific validity** | Questionable | Valid |
| **Real-world simulation** | Poor | Accurate |

**Bottom line:** Always use per-fold preprocessing for honest cross-validation!


