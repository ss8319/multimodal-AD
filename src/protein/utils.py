"""
Utility functions for model evaluation and cross-validation
"""
import numpy as np
import pandas as pd
import pickle
import torch
import json
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


def save_cv_fold_indices(X, y, df, cv_splitter, output_path, id_col='RID'):
    """
    Generate and save CV fold assignments for reproducibility
    
    Args:
        X: Feature matrix
        y: Labels
        df: Original DataFrame with metadata
        cv_splitter: sklearn CV splitter (e.g., StratifiedKFold)
        output_path: Path to save fold assignments
        id_col: Column name for subject IDs
    """
    fold_data = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y)):
        # Record train split assignments
        for idx in train_idx:
            fold_data.append({
                'original_index': df.index[idx],
                id_col: df.iloc[idx][id_col],
                'fold': fold_idx + 1,
                'split_type': 'train'
            })
        
        # Record validation split assignments
        for idx in val_idx:
            fold_data.append({
                'original_index': df.index[idx],
                id_col: df.iloc[idx][id_col],
                'fold': fold_idx + 1,
                'split_type': 'val'
            })
    
    fold_df = pd.DataFrame(fold_data)
    fold_df.to_csv(output_path, index=False)
    print(f"💾 Saved CV fold indices to: {output_path}")
    return fold_df


def compute_confusion_metrics(y_true, y_pred):
    """
    Compute confusion matrix and derived metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        dict with confusion matrix metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    return {
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv
    }


def preprocess_fold(X_train_raw, X_val_raw, train_idx, val_idx, use_per_fold_preprocessing):
    """
    Preprocess a single CV fold with per-fold scaling
    
    Args:
        X_train_raw: Raw training features (DataFrame or array)
        X_val_raw: Raw validation features (unused if extracting from X_train_raw)
        train_idx: Training indices for this fold
        val_idx: Validation indices for this fold
        use_per_fold_preprocessing: Whether to use per-fold preprocessing
        
    Returns:
        X_train_fold, X_val_fold, y_train_fold, y_val_fold
    """
    if use_per_fold_preprocessing:
        fold_scaler = StandardScaler()
        X_train_fold_raw = X_train_raw.iloc[train_idx]
        X_val_fold_raw = X_train_raw.iloc[val_idx]
        
        fold_scaler.fit(X_train_fold_raw)
        X_train_fold = fold_scaler.transform(X_train_fold_raw)
        X_val_fold = fold_scaler.transform(X_val_fold_raw)
    else:
        X_train_fold = X_train_raw[train_idx]
        X_val_fold = X_train_raw[val_idx]
    
    return X_train_fold, X_val_fold


def evaluate_fold(fold_clf, X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test, y_test):
    """
    Evaluate a single fold on validation and optional test set
    
    Args:
        fold_clf: Trained classifier for this fold
        X_train_fold: Training features for this fold
        y_train_fold: Training labels for this fold
        X_val_fold: Validation features for this fold
        y_val_fold: Validation labels for this fold
        X_test: Test features (can be None)
        y_test: Test labels (can be None)
        
    Returns:
        dict with cv_auc, cv_acc, test_auc, test_acc (test metrics are None if no test set)
    """
    # Validation metrics
    val_pred = fold_clf.predict(X_val_fold)
    cv_acc = accuracy_score(y_val_fold, val_pred)
    
    cv_auc = np.nan
    if hasattr(fold_clf, 'predict_proba'):
        val_proba = fold_clf.predict_proba(X_val_fold)
        if val_proba is not None and not np.isnan(val_proba).any():
            cv_auc = roc_auc_score(y_val_fold, val_proba[:, 1])
    
    # Test metrics (if available)
    test_acc, test_auc = np.nan, np.nan
    if X_test is not None and y_test is not None:
        test_pred = fold_clf.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        
        if hasattr(fold_clf, 'predict_proba'):
            test_proba = fold_clf.predict_proba(X_test)
            if test_proba is not None and not np.isnan(test_proba).any():
                test_auc = roc_auc_score(y_test, test_proba[:, 1])
    
    return {
        'cv_auc': cv_auc,
        'cv_acc': cv_acc,
        'test_auc': test_auc,
        'test_acc': test_acc
    }


def evaluate_model_cv(clf, X_train_raw, y_train, X_test, y_test, cv_splitter, clf_name="Model", compute_test_confusion=False, data_loader=None):
    """
    Evaluate a classifier using cross-validation and optional test set
    
    Args:
        clf: sklearn-compatible classifier
        X_train_raw: Raw training features (UNSCALED) - if None, assumes X_train is pre-scaled
        y_train: Training labels
        X_test: Test features (can be None)
        y_test: Test labels (can be None)
        cv_splitter: sklearn CV splitter
        clf_name: Name of classifier for logging
        compute_test_confusion: If True, compute confusion matrix on test set
        data_loader: ProteinDataLoader instance for per-fold preprocessing (if X_train_raw is DataFrame)
    
    Returns:
        dict with CV and test metrics
    """
    cv_auc_scores = []
    cv_acc_scores = []
    test_auc_scores = []
    test_acc_scores = []
    
    test_available = X_test is not None and y_test is not None
    use_per_fold_preprocessing = data_loader is not None and hasattr(X_train_raw, 'columns')
    
    # === CROSS-VALIDATION LOOP ===
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X_train_raw, y_train)):
        try:
            # Preprocess fold
            X_train_fold, X_val_fold = preprocess_fold(
                X_train_raw, None, train_idx, val_idx, use_per_fold_preprocessing
            )
            y_train_fold = y_train[train_idx]
            y_val_fold = y_train[val_idx]
            
            # Train model on fold
            fold_clf = clone(clf)
            fold_clf.fit(X_train_fold, y_train_fold)
            
            # Evaluate fold
            fold_metrics = evaluate_fold(
                fold_clf, X_train_fold, y_train_fold,
                X_val_fold, y_val_fold, X_test, y_test
            )
            
            # Collect metrics
            cv_auc_scores.append(fold_metrics['cv_auc'])
            cv_acc_scores.append(fold_metrics['cv_acc'])
            if test_available:
                test_auc_scores.append(fold_metrics['test_auc'])
                test_acc_scores.append(fold_metrics['test_acc'])
            
        except Exception as e:
            print(f"   ⚠️  Fold {fold_idx + 1} failed: {str(e)[:50]}")
            cv_auc_scores.append(np.nan)
            cv_acc_scores.append(np.nan)
            if test_available:
                test_auc_scores.append(np.nan)
                test_acc_scores.append(np.nan)
    
    # Calculate statistics
    cv_auc_scores = np.array(cv_auc_scores)
    cv_acc_scores = np.array(cv_acc_scores)
    
    results = {
        'classifier': clf_name,
        'cv_auc_mean': np.nanmean(cv_auc_scores),
        'cv_auc_std': np.nanstd(cv_auc_scores),
        'cv_acc_mean': np.nanmean(cv_acc_scores),
        'cv_acc_std': np.nanstd(cv_acc_scores),
        'cv_auc_scores': cv_auc_scores.tolist(),
        'cv_acc_scores': cv_acc_scores.tolist(),
    }
    
    if test_available:
        test_auc_scores = np.array(test_auc_scores)
        test_acc_scores = np.array(test_acc_scores)
        results.update({
            'test_auc_mean': np.nanmean(test_auc_scores),
            'test_auc_std': np.nanstd(test_auc_scores),
            'test_acc_mean': np.nanmean(test_acc_scores),
            'test_acc_std': np.nanstd(test_acc_scores),
            'test_auc_scores': test_auc_scores.tolist(),
            'test_acc_scores': test_acc_scores.tolist(),
        })
        
        # Optionally compute confusion matrix on full test set
        if compute_test_confusion:
            final_clf = train_final_model(clf, X_train_raw, y_train, use_per_fold_preprocessing)
            test_pred_final = final_clf.predict(X_test)
            confusion_metrics = compute_confusion_metrics(y_test, test_pred_final)
            results.update(confusion_metrics)
    
    return results


def train_final_model(clf, X_train_raw, y_train, use_per_fold_preprocessing):
    """
    Train a model on the full training set (after CV)
    
    Args:
        clf: sklearn-compatible classifier
        X_train_raw: Raw training features
        y_train: Training labels
        use_per_fold_preprocessing: Whether to scale features
        
    Returns:
        Trained classifier
    """
    final_clf = clone(clf)
    
    if use_per_fold_preprocessing:
        # Scale the full training set
        final_scaler = StandardScaler()
        X_train_scaled = final_scaler.fit_transform(X_train_raw)
        final_clf.fit(X_train_scaled, y_train)
    else:
        final_clf.fit(X_train_raw, y_train)
    
    return final_clf


def print_results_summary(results_df, test_available=False, show_confusion=False):
    """
    Print formatted results summary
    
    Args:
        results_df: DataFrame with evaluation results
        test_available: Whether test set results are available
        show_confusion: Whether to show confusion matrix metrics
    """
    print(f"\n🏆 FINAL RESULTS SUMMARY")
    print("=" * 60)
    
    if len(results_df) == 0:
        print("   • No successful results")
        return
    
    # Sort by CV AUC
    results_sorted = results_df.fillna({'cv_auc_mean': 0}).sort_values('cv_auc_mean', ascending=False)
    
    print("📊 Ranking by CV AUC (Mean ± Std):")
    for idx, (_, row) in enumerate(results_sorted.iterrows(), 1):
        cv_auc_str = f"{row['cv_auc_mean']:.3f}±{row['cv_auc_std']:.3f}" if not np.isnan(row['cv_auc_mean']) else "NaN±NaN"
        print(f"   {idx}. {row['classifier']:<20}: {cv_auc_str}")
    
    # Show test results if available
    if test_available and 'test_auc_mean' in results_df.columns:
        print(f"\n🎯 Test Set Performance:")
        test_sorted = results_df.fillna({'test_auc_mean': 0}).sort_values('test_auc_mean', ascending=False)
        for idx, (_, row) in enumerate(test_sorted.iterrows(), 1):
            test_auc_str = f"{row['test_auc_mean']:.3f}±{row['test_auc_std']:.3f}" if not np.isnan(row['test_auc_mean']) else "NaN±NaN"
            print(f"   {idx}. {row['classifier']:<20}: {test_auc_str}")
    
    # Best model details
    best_model = results_sorted.iloc[0]
    print(f"\n🏆 Best Model: {best_model['classifier']}")
    print(f"   • CV AUC: {best_model['cv_auc_mean']:.3f} ± {best_model['cv_auc_std']:.3f}")
    print(f"   • CV Accuracy: {best_model['cv_acc_mean']:.3f} ± {best_model['cv_acc_std']:.3f}")
    
    if test_available and 'test_auc_mean' in best_model:
        print(f"   • Test AUC: {best_model['test_auc_mean']:.3f} ± {best_model['test_auc_std']:.3f}")
        print(f"   • Test Accuracy: {best_model['test_acc_mean']:.3f} ± {best_model['test_acc_std']:.3f}")
    
    # Show confusion matrix if requested and available
    if show_confusion and 'tn' in best_model:
        print(f"\n📊 Confusion Matrix (Test Set):")
        print(f"                Predicted")
        print(f"              CN (0)  AD (1)")
        print(f"Actual CN (0)   {best_model['tn']:4.0f}    {best_model['fp']:4.0f}")
        print(f"       AD (1)   {best_model['fn']:4.0f}    {best_model['tp']:4.0f}")
        print(f"\n   Sensitivity: {best_model['sensitivity']:.3f}")
        print(f"   Specificity: {best_model['specificity']:.3f}")
        print(f"   PPV:         {best_model['ppv']:.3f}")
        print(f"   NPV:         {best_model['npv']:.3f}")


def save_results(results_dict, output_dir):
    """
    Save evaluation results to pickle file
    
    Args:
        results_dict: Dictionary of results
        output_dir: Directory to save results
    """
    output_path = Path(output_dir) / "cv_fold_detailed_results.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(results_dict, f)
    print(f"💾 Saved detailed results to: {output_path}")


def create_run_directory():
    """
    Create a timestamped directory for this run
    
    Returns:
        Path to the run directory
    """
    runs_dir = Path("src/protein/runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Created run directory: {run_dir}")
    return run_dir


def save_model(model, model_name, run_dir):
    """
    Save a trained model to disk
    
    Args:
        model: Trained model (sklearn or PyTorch wrapper)
        model_name: Name of the model
        run_dir: Directory to save the model
    """
    model_dir = Path(run_dir) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if it's a PyTorch model
    is_pytorch = hasattr(model, 'model') and hasattr(model.model, 'state_dict')
    
    if is_pytorch:
        # Save PyTorch model state dict
        model_path = model_dir / f"{model_name}.pth"
        torch.save({
            'model_state_dict': model.model.state_dict(),
            'model_config': {
                'd_model': model.d_model,
                'n_heads': model.n_heads,
                'n_layers': model.n_layers,
                'dropout': model.dropout,
                'n_features': model.model.n_features
            }
        }, model_path)
        print(f"   💾 Saved PyTorch model: {model_path.name}")
    else:
        # Save sklearn model with pickle
        model_path = model_dir / f"{model_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"   💾 Saved sklearn model: {model_path.name}")


def save_all_models(classifiers, results_df, X_train_raw, y_train, use_per_fold_preprocessing, run_dir):
    """
    Train and save all classifiers on full training set
    
    Args:
        classifiers: Dict of classifier name -> classifier instance
        results_df: DataFrame with CV results
        X_train_raw: Raw training features
        y_train: Training labels
        use_per_fold_preprocessing: Whether to use scaling
        run_dir: Directory to save models
    """
    print(f"\n💾 TRAINING AND SAVING FINAL MODELS")
    print("-" * 70)
    
    for clf_name, clf in classifiers.items():
        try:
            # Train on full training set
            final_model = train_final_model(clf, X_train_raw, y_train, use_per_fold_preprocessing)
            
            # Save model
            # Sanitize filename
            safe_name = clf_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
            save_model(final_model, safe_name, run_dir)
            
        except Exception as e:
            print(f"   ⚠️  Failed to save {clf_name}: {str(e)[:50]}")
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'n_models': len(classifiers),
        'model_names': list(classifiers.keys()),
        'best_model': results_df.sort_values('cv_auc_mean', ascending=False).iloc[0]['classifier'],
        'best_cv_auc': results_df.sort_values('cv_auc_mean', ascending=False).iloc[0]['cv_auc_mean']
    }
    
    metadata_path = Path(run_dir) / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   📄 Saved metadata: {metadata_path.name}")
    
    print(f"\n✅ All models saved to: {run_dir / 'models'}")