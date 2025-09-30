"""
Utility functions for model evaluation and cross-validation
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score, roc_auc_score


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


def evaluate_model_cv(clf, X_train, y_train, X_test, y_test, cv_splitter, clf_name="Model"):
    """
    Evaluate a classifier using cross-validation and optional test set
    
    Args:
        clf: sklearn-compatible classifier
        X_train: Training features
        y_train: Training labels
        X_test: Test features (can be None)
        y_test: Test labels (can be None)
        cv_splitter: sklearn CV splitter
        clf_name: Name of classifier for logging
    
    Returns:
        dict with CV and test metrics
    """
    cv_auc_scores = []
    cv_acc_scores = []
    test_auc_scores = []
    test_acc_scores = []
    
    test_available = X_test is not None and y_test is not None
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X_train, y_train)):
        try:
            # Clone and train model on fold
            fold_clf = clone(clf)
            fold_clf.fit(X_train[train_idx], y_train[train_idx])
            
            # === CV Validation Evaluation ===
            val_pred = fold_clf.predict(X_train[val_idx])
            cv_acc = accuracy_score(y_train[val_idx], val_pred)
            cv_acc_scores.append(cv_acc)
            
            # Compute AUC if probabilities available
            if hasattr(fold_clf, 'predict_proba'):
                val_proba = fold_clf.predict_proba(X_train[val_idx])
                if val_proba is not None and not np.isnan(val_proba).any():
                    cv_auc = roc_auc_score(y_train[val_idx], val_proba[:, 1])
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
    
    return results


def print_results_summary(results_df, test_available=False):
    """
    Print formatted results summary
    
    Args:
        results_df: DataFrame with evaluation results
        test_available: Whether test set results are available
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
