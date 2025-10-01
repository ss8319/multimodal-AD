"""
Utility functions for model evaluation and cross-validation
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


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
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X_train_raw if use_per_fold_preprocessing else X_train_raw, y_train)):
        try:
            # === PER-FOLD PREPROCESSING (if enabled) ===
            if use_per_fold_preprocessing:
                # Create a fresh loader for this fold
                from sklearn.preprocessing import StandardScaler
                fold_scaler = StandardScaler()
                
                # Get raw data for this fold
                X_train_fold_raw = X_train_raw.iloc[train_idx]
                X_val_fold_raw = X_train_raw.iloc[val_idx]
                
                # Fit scaler ONLY on training fold
                fold_scaler.fit(X_train_fold_raw)
                
                # Transform both training and validation folds
                X_train_fold = fold_scaler.transform(X_train_fold_raw)
                X_val_fold = fold_scaler.transform(X_val_fold_raw)
                
                y_train_fold = y_train[train_idx]
                y_val_fold = y_train[val_idx]
            else:
                # Use pre-scaled data (backward compatibility)
                X_train_fold = X_train_raw[train_idx]
                X_val_fold = X_train_raw[val_idx]
                y_train_fold = y_train[train_idx]
                y_val_fold = y_train[val_idx]
            
            # Clone and train model on fold
            fold_clf = clone(clf)
            fold_clf.fit(X_train_fold, y_train_fold)
            
            # === CV Validation Evaluation ===
            val_pred = fold_clf.predict(X_val_fold)
            cv_acc = accuracy_score(y_val_fold, val_pred)
            cv_acc_scores.append(cv_acc)
            
            # Compute AUC if probabilities available
            if hasattr(fold_clf, 'predict_proba'):
                val_proba = fold_clf.predict_proba(X_val_fold)
                if val_proba is not None and not np.isnan(val_proba).any():
                    cv_auc = roc_auc_score(y_val_fold, val_proba[:, 1])
                    cv_auc_scores.append(cv_auc)
                else:
                    cv_auc_scores.append(np.nan)
            else:
                cv_auc_scores.append(np.nan)
            
            # === Test Set Evaluation ===
            if test_available:
                test_pred = fold_clf.predict(X_test)
                test_acc = accuracy_score(y_test, test_pred)
                test_acc_scores.append(test_acc)
                
                if hasattr(fold_clf, 'predict_proba'):
                    test_proba = fold_clf.predict_proba(X_test)
                    if test_proba is not None and not np.isnan(test_proba).any():
                        test_auc = roc_auc_score(y_test, test_proba[:, 1])
                        test_auc_scores.append(test_auc)
                    else:
                        test_auc_scores.append(np.nan)
                else:
                    test_auc_scores.append(np.nan)
            
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
            # Train on full training set for final test evaluation
            final_clf = clone(clf)
            
            if use_per_fold_preprocessing:
                # Need to scale the full training set before fitting
                from sklearn.preprocessing import StandardScaler
                final_scaler = StandardScaler()
                X_train_scaled = final_scaler.fit_transform(X_train_raw)
                final_clf.fit(X_train_scaled, y_train)
            else:
                final_clf.fit(X_train_raw, y_train)
            
            test_pred_final = final_clf.predict(X_test)
            confusion_metrics = compute_confusion_metrics(y_test, test_pred_final)
            results.update(confusion_metrics)
    
    return results


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