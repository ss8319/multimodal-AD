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
    print(f"Saved CV fold indices to: {output_path}")
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

def evaluate_model_cv(clf, X_train_raw, y_train, X_test, y_test, cv_splitter, clf_name="Model"):
    """
    Evaluate a classifier using cross-validation and optional test set
    Per-fold preprocessing:
    - Each fold fits its own StandardScaler on the fold's training data
    - Validation data is scaled using the fold's scaler
    - Test data (if provided) is scaled using each fold's scaler for evaluation
    Args:
        clf: sklearn-compatible classifier
        X_train_raw: DataFrame with preprocessed features (NOT scaled)
        y_train: Encoded training labels
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

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X_train_raw, y_train)):
        try:
            # Per-fold preprocessing: fit scaler on fold's training data only
            fold_scaler = StandardScaler()
            X_train_fold_raw = X_train_raw.iloc[train_idx]
            X_val_fold_raw = X_train_raw.iloc[val_idx]

            # Fit scaler ONLY on training fold (no data leakage)
            X_train_fold = fold_scaler.fit_transform(X_train_fold_raw)
            X_val_fold = fold_scaler.transform(X_val_fold_raw)
            
            y_train_fold = y_train[train_idx]
            y_val_fold = y_train[val_idx]

            # Train model on fold (convert to numpy to avoid feature name warnings)
            fold_clf = clone(clf)
            fold_clf.fit(np.asarray(X_train_fold), y_train_fold)

            # === CV Validation Evaluation ===
            val_pred = fold_clf.predict(np.asarray(X_val_fold))
            cv_acc = accuracy_score(y_val_fold, val_pred)
            cv_acc_scores.append(cv_acc)

            # Compute AUC if probabilities available
            if hasattr(fold_clf, 'predict_proba'):
                val_proba = fold_clf.predict_proba(np.asarray(X_val_fold))
                if val_proba is not None and not np.isnan(val_proba).any():
                    # Check AUC conditions
                    n_unique_labels = len(np.unique(y_val_fold))
                    n_unique_preds = len(np.unique(val_pred))
                    
                    if n_unique_labels < 2:
                        cv_auc_scores.append(float('nan'))
                        print("  Warning: Only one class in validation labels - AUC undefined")
                    elif n_unique_preds < 2:
                        cv_auc_scores.append(0.5)
                        print("  Warning: Model predicting only ONE class - Using AUC = 0.5")
                    else:
                        try:
                            cv_auc = roc_auc_score(y_val_fold, val_proba[:, 1])
                            cv_auc_scores.append(cv_auc)
                        except ValueError:
                            cv_auc_scores.append(float('nan'))
                            print("  Warning: AUC calculation failed")
                else:
                    cv_auc_scores.append(np.nan)
            else:
                cv_auc_scores.append(np.nan)

            # === Test Set Evaluation ===
            if test_available:
                # Scale test set using the current fold's scaler
                X_test_scaled = fold_scaler.transform(X_test)
                test_pred = fold_clf.predict(np.asarray(X_test_scaled))
                test_acc = accuracy_score(y_test, test_pred)
                test_acc_scores.append(test_acc)
                
                if hasattr(fold_clf, 'predict_proba'):
                    test_proba = fold_clf.predict_proba(np.asarray(X_test_scaled))
                    if test_proba is not None and not np.isnan(test_proba).any():
                        # Check AUC conditions
                        n_unique_labels = len(np.unique(y_test))
                        n_unique_preds = len(np.unique(test_pred))
                        
                        if n_unique_labels < 2:
                            test_auc_scores.append(float('nan'))
                            print("  Warning: Only one class in test labels - AUC undefined")
                        elif n_unique_preds < 2:
                            test_auc_scores.append(0.5)
                            print("  Warning: Model predicting only ONE class - Using AUC = 0.5")
                        else:
                            try:
                                test_auc = roc_auc_score(y_test, test_proba[:, 1])
                                test_auc_scores.append(test_auc)
                            except ValueError:
                                test_auc_scores.append(float('nan'))
                                print("  Warning: AUC calculation failed")
                    else:
                        test_auc_scores.append(np.nan)
                else:
                    test_auc_scores.append(np.nan)

        except Exception as e:
            print(f"   Warning: Fold {fold_idx + 1} failed: {str(e)[:50]}")
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

    print(f"\nFINAL RESULTS SUMMARY")
    print("=" * 60)
    if len(results_df) == 0:
        print("   No successful results")
        return

    # Sort by CV AUC
    results_sorted = results_df.fillna({'cv_auc_mean': 0}).sort_values('cv_auc_mean', ascending=False)
    
    print("\nRanking by CV AUC (Mean ± Std):")
    for idx, (_, row) in enumerate(results_sorted.iterrows(), 1):
        cv_auc_str = f"{row['cv_auc_mean']:.3f}±{row['cv_auc_std']:.3f}" if not np.isnan(row['cv_auc_mean']) else "NaN±NaN"
        print(f"   {idx}. {row['classifier']:<20}: {cv_auc_str}")
    
    # Show test results if available
    if test_available and 'test_auc_mean' in results_df.columns:
        print(f"\nTest Set Performance:")
        test_sorted = results_df.fillna({'test_auc_mean': 0}).sort_values('test_auc_mean', ascending=False)
        for idx, (_, row) in enumerate(test_sorted.iterrows(), 1):
            test_auc_str = f"{row['test_auc_mean']:.3f}±{row['test_auc_std']:.3f}" if not np.isnan(row['test_auc_mean']) else "NaN±NaN"
            print(f"   {idx}. {row['classifier']:<20}: {test_auc_str}")
    
    # Best model details
    best_model = results_sorted.iloc[0]
    print(f"\nBest Model: {best_model['classifier']}")
    print(f"   CV AUC: {best_model['cv_auc_mean']:.3f} ± {best_model['cv_auc_std']:.3f}")
    print(f"   CV Accuracy: {best_model['cv_acc_mean']:.3f} ± {best_model['cv_acc_std']:.3f}")
    if test_available and 'test_auc_mean' in best_model:
        print(f"   Test AUC: {best_model['test_auc_mean']:.3f} ± {best_model['test_auc_std']:.3f}")
        print(f"   Test Accuracy: {best_model['test_acc_mean']:.3f} ± {best_model['test_acc_std']:.3f}")

def save_results(results_dict, output_dir):
    """
    Save detailed CV evaluation results to pickle file
    Args:
        results_dict: Dictionary of detailed CV results
        output_dir: Directory to save results
    """
    output_path = Path(output_dir) / "cv_detailed_results.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(results_dict, f)
    print(f"Saved detailed CV results to: {output_path}")

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
    print(f"Created run directory: {run_dir}")
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
    # Check if it's a PyTorch model (has .model attribute with state_dict)
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
        print(f"   Saved PyTorch model: {model_path.name}")
    else:
        # Save sklearn model with pickle
        model_path = model_dir / f"{model_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"   Saved sklearn model: {model_path.name}")

def save_all_models(classifiers, results_df, X_train_raw, y_train, X_test, y_test, run_dir):
    """
    Train and save all classifiers on full training set
    Args:
        classifiers: Dict of classifier name -> classifier instance
        results_df: DataFrame with CV results
        X_train_raw: DataFrame with preprocessed features (NOT scaled)
        y_train: Encoded training labels (AD=1, CN=0)
        X_test: Test features (can be None)
        y_test: Test labels (can be None)
        run_dir: Directory to save models
    """
    print(f"\nTRAINING AND SAVING FINAL MODELS")
    print("-" * 70)
    print(f"   Training on FULL training set ({len(X_train_raw)} samples)")
    
    test_available = X_test is not None and y_test is not None
    final_test_results = []
    
    # Fit scaler on full training set (shared across all models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    
    # Save the scaler for later use in inference
    scaler_path = Path(run_dir) / "scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"   Saved scaler: {scaler_path.name}")
    
    # Also save the exact training feature order next to the scaler
    try:
        feature_columns = list(X_train_raw.columns)
        scaler_features_path = Path(run_dir) / "scaler_features.json"
        with open(scaler_features_path, 'w') as f:
            json.dump({
                'feature_columns': feature_columns
            }, f, indent=2)
        print(f"   Saved scaler features: {scaler_features_path.name} ({len(feature_columns)} columns)")
    except Exception as e:
        print(f"   Warning: Could not save scaler feature columns: {e}")
    
    for clf_name, clf in classifiers.items():
        try:
            print(f"\n   Training {clf_name}...")

            # Clone and train on scaled data
            final_clf = clone(clf)
            final_clf.fit(np.asarray(X_train_scaled), y_train)

            # Evaluate on test set if available
            test_metrics = {}
            if test_available:
                X_test_scaled = scaler.transform(X_test)
                test_pred = final_clf.predict(np.asarray(X_test_scaled))
                test_acc = accuracy_score(y_test, test_pred)
                test_metrics['test_accuracy'] = float(test_acc)
                
                if hasattr(final_clf, 'predict_proba'):
                    test_proba = final_clf.predict_proba(np.asarray(X_test_scaled))
                    if test_proba is not None and not np.isnan(test_proba).any():
                        # Check AUC conditions
                        n_unique_labels = len(np.unique(y_test))
                        n_unique_preds = len(np.unique(test_pred))
                        
                        if n_unique_labels < 2:
                            test_metrics['test_auc'] = float('nan')
                            print("  Warning: Only one class in test labels - AUC undefined")
                        elif n_unique_preds < 2:
                            test_metrics['test_auc'] = 0.5
                            print("  Warning: Model predicting only ONE class - Using AUC = 0.5")
                        else:
                            try:
                                test_auc = roc_auc_score(y_test, test_proba[:, 1])
                                test_metrics['test_auc'] = float(test_auc)
                            except ValueError:
                                test_metrics['test_auc'] = float('nan')
                                print("  Warning: AUC calculation failed")
                
                print(f"      Test AUC: {test_metrics.get('test_auc', 'N/A'):.3f}" if 'test_auc' in test_metrics else "      Test AUC: N/A")
                print(f"      Test Accuracy: {test_metrics['test_accuracy']:.3f}")
                
                final_test_results.append({
                    'model': clf_name,
                    **test_metrics
                })

            # Save model
            safe_name = clf_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
            save_model(final_clf, safe_name, run_dir)
        except Exception as e:
            print(f"   Failed to save {clf_name}: {str(e)[:50]}")
   
    # Save final test results as simple CSV (no redundant metadata.json)
    if final_test_results:
        final_results_df = pd.DataFrame(final_test_results)
        final_results_path = Path(run_dir) / "final_test_results.csv"
        final_results_df.to_csv(final_results_path, index=False)
        print(f"\n   Saved final test results: {final_results_path.name}")
        
        # Show best final model
        best_test = max(final_test_results, key=lambda x: x.get('test_auc', 0))
        print(f"   Best final model: {best_test['model']} (AUC: {best_test.get('test_auc', 'N/A'):.3f})")
    
    print(f"\nAll models saved to: {run_dir / 'models'}")
    print(f"Scaler saved to: {scaler_path}")
