"""
Evaluate protein-only baseline using pre-trained model on fusion CV splits
This allows fair comparison between unimodal (protein-only) and multimodal (protein+MRI) approaches

Key differences from fusion evaluation:
- Uses pre-trained protein model (no retraining per fold)
- Uses pre-fitted scaler from training run
- Only evaluates on test set (no validation set)
- Same CV splits as fusion for fair comparison
"""

import argparse
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    confusion_matrix, 
    precision_recall_fscore_support,
    balanced_accuracy_score
)

# Feature alignment helper
import sys as _sys
from pathlib import Path as _P
_protein_mod_path = str(_P(__file__).parent.parent.parent / "protein")
if _protein_mod_path not in _sys.path:
    _sys.path.insert(0, _protein_mod_path)
from feature_utils import align_features_to_scaler, load_scaler_feature_columns


def normalize_path(path_str):
    """Centralized path normalization helper
    
    Handles relative paths that might be prefixed with project name.
    Strips duplicate 'multimodal-AD/' prefix if already in project directory.
    
    Args:
        path_str: Path string (can be relative or absolute)
        
    Returns:
        Path: Normalized absolute path
    """
    if isinstance(path_str, str) and path_str.startswith('multimodal-AD/'):
        # If path starts with project name but we're already in project dir
        cwd = Path.cwd()
        if cwd.name == 'multimodal-AD' or str(cwd).endswith('/multimodal-AD'):
            # Strip the project prefix to avoid duplication
            path_str = path_str.replace('multimodal-AD/', '', 1)
            print(f"  Removed duplicate project prefix from path")
    
    # Normalize to absolute path
    normalized_path = Path(path_str).expanduser().resolve()
    print(f"  Using normalized path: {normalized_path}")
    return normalized_path


def load_cv_splits(cv_splits_path):
    """Load CV splits from fusion experiment"""
    # Normalize path using centralized utility
    cv_splits_path = normalize_path(cv_splits_path)
    
    try:
        with open(cv_splits_path, 'r') as f:
            cv_splits = json.load(f)
        
        print(f"  Loaded {len(cv_splits)} CV folds from {cv_splits_path}")
        return cv_splits
    except FileNotFoundError:
        print(f"  ERROR: CV splits file not found: {cv_splits_path}")
        print(f"  Current working directory: {Path.cwd()}")
        raise


def load_protein_model(model_path, model_type='nn'):
    """Load pre-trained protein model
    
    Args:
        model_path: Path to model file (.pth for PyTorch models)
        model_type: 'nn' (PyTorch Neural Network) or 'transformer'
    """
    model_path = Path(model_path)
    
    # Import PyTorch and model definitions
    import torch
    import sys
    from pathlib import Path as P
    
    # Add protein module to path for model import
    protein_path = str(P(__file__).parent.parent.parent / "protein")
    if protein_path not in sys.path:
        sys.path.insert(0, protein_path)
    
    if model_type == 'nn':
        # Load PyTorch Neural Network model
        from model import NeuralNetwork
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Recreate model with saved config
        if 'model_config' not in checkpoint:
            raise ValueError("Model checkpoint missing 'model_config'. Cannot load Neural Network model.")
            
        model_config = checkpoint['model_config']
        
        # Extract architecture parameters
        n_features = model_config.get('n_features')
        hidden_sizes = model_config.get('hidden_sizes', (128, 64))
        dropout = model_config.get('dropout', 0.2)
        
        if n_features is None:
            raise ValueError(
                "model_config missing 'n_features'. Cannot reconstruct Neural Network.\n"
                "Please retrain the protein model with the updated code to include n_features in the checkpoint."
            )
        
        # Recreate model
        model = NeuralNetwork(
            n_features=n_features,
            hidden_sizes=hidden_sizes,
            dropout=dropout
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Loaded pre-trained Neural Network model from {model_path}")
        print(f"  Model architecture: n_features={n_features}, hidden_sizes={hidden_sizes}, dropout={dropout}")
        
        return model
    
    elif model_type == 'transformer':
        # Load PyTorch Transformer model
        from model import ProteinTransformer
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Recreate model with saved config
        if 'model_config' not in checkpoint:
            raise ValueError("Model checkpoint missing 'model_config'. Cannot load transformer model.")
            
        model_config = checkpoint['model_config']
        model = ProteinTransformer(**model_config)
        
        # For logging
        n_features = model_config.get('n_features')
        d_model = model_config.get('d_model')
        n_layers = model_config.get('n_layers')
        n_heads = model_config.get('n_heads')
        
        print(f"Loaded model config from checkpoint")
        
        # Load weights
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        model.eval()
        
        print(f"Loaded pre-trained Transformer model from {model_path}")
        print(f"  Model architecture: n_features={n_features}, d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}")
        
        return model
    
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Use 'nn' or 'transformer'")


def load_scaler(scaler_path):
    """Load pre-fitted scaler"""
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"Loaded pre-fitted scaler from {scaler_path}")
    return scaler


def evaluate_fold_test(model, scaler, X_test, y_test, fold_idx, model_type='nn'):
    """Evaluate pre-trained model on test set for a single fold
    
    Args:
        model: Pre-trained model (PyTorch Neural Network or Transformer)
        scaler: Pre-fitted StandardScaler
        X_test: Test features
        y_test: Test labels
        fold_idx: Fold index
        model_type: 'nn' or 'transformer'
    """
    
    # Apply pre-fitted scaler to test data
    X_test_scaled = scaler.transform(X_test)
    
    # Run inference (no training)
    if model_type in ['nn', 'transformer']:
        # PyTorch model (both Neural Network and Transformer)
        import torch
        import torch.nn.functional as F
        
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test_scaled)
            logits = model(X_test_tensor)
            test_probs = F.softmax(logits, dim=1)[:, 1].numpy()
            test_preds = torch.argmax(logits, dim=1).numpy()
    
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Use 'nn' or 'transformer'")
    
    # Calculate metrics
    test_acc = accuracy_score(y_test, test_preds)
    test_balanced_acc = balanced_accuracy_score(y_test, test_preds)
    
    # Calculate test AUC with proper handling
    n_unique_preds = len(np.unique(test_preds))
    n_unique_labels = len(np.unique(y_test))
    
    if n_unique_labels < 2:
        test_auc = float('nan')
        print("  Warning: Only one class in test labels - AUC undefined")
    elif n_unique_preds < 2:
        test_auc = 0.5
        print("  Warning: Model predicting only ONE class - Using AUC = 0.5")
    else:
        try:
            test_auc = roc_auc_score(y_test, test_probs)
        except ValueError:
            test_auc = float('nan')
            print("  Warning: AUC calculation failed")
    
    test_cm = confusion_matrix(y_test, test_preds, labels=[0, 1])
    
    # Calculate precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, test_preds, average='binary', zero_division=0
    )
    
    results = {
        'fold': fold_idx,
        'test_acc': test_acc,
        'test_balanced_acc': test_balanced_acc,
        'test_auc': test_auc,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1': f1,
        'test_cm': test_cm.tolist(),
        'n_test': len(y_test)
    }
    
    return results


def print_fold_results(results, fold_idx):
    """Print results for a single fold"""
    print(f"\nFold {fold_idx} Test Results:")
    print(f"  Accuracy: {results['test_acc']:.4f}")
    print(f"  Balanced Accuracy: {results['test_balanced_acc']:.4f}")
    if np.isnan(results['test_auc']):
        print(f"  AUC: undefined")
    else:
        print(f"  AUC: {results['test_auc']:.4f}")
    print(f"  Precision: {results['test_precision']:.4f}")
    print(f"  Recall: {results['test_recall']:.4f}")
    print(f"  F1: {results['test_f1']:.4f}")
    
    cm = np.array(results['test_cm'])
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}, TP={cm[1,1]}")


def aggregate_results(fold_results):
    """Aggregate results across all folds"""
    metrics_to_aggregate = ['test_acc', 'test_balanced_acc', 'test_auc', 'test_precision', 'test_recall', 'test_f1']
    aggregated = {}
    
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS SUMMARY (PROTEIN-ONLY BASELINE)")
    print("="*60)
    
    for metric in metrics_to_aggregate:
        values = [f[metric] for f in fold_results]
        
        # Handle NaN values
        if metric == 'test_auc' and any(np.isnan(v) for v in values):
            aggregated[metric] = {
                'mean': float('nan'),
                'std': float('nan'),
                'values': values
            }
            metric_name = metric.replace('test_', '').replace('_', ' ').upper()
            print(f"{metric_name}: undefined (some folds had undefined AUC)")
        else:
            valid_values = [v for v in values if not np.isnan(v)]
            if valid_values:
                aggregated[metric] = {
                    'mean': np.mean(valid_values),
                    'std': np.std(valid_values),
                    'values': values
                }
                metric_name = metric.replace('test_', '').replace('_', ' ').upper()
                print(f"{metric_name}: {aggregated[metric]['mean']:.4f} ± {aggregated[metric]['std']:.4f}")
            else:
                aggregated[metric] = {
                    'mean': float('nan'),
                    'std': float('nan'),
                    'values': values
                }
                metric_name = metric.replace('test_', '').replace('_', ' ').upper()
                print(f"{metric_name}: undefined (all folds had undefined values)")
    
    # Aggregate confusion matrix
    total_cm = np.sum([np.array(f['test_cm']) for f in fold_results], axis=0)
    print(f"\nAggregated Confusion Matrix:")
    print(f"  TN={total_cm[0,0]}, FP={total_cm[0,1]}")
    print(f"  FN={total_cm[1,0]}, TP={total_cm[1,1]}")
    
    return aggregated, total_cm


def main(args):
    """Main evaluation function"""
    print("="*60)
    print("PROTEIN-ONLY BASELINE EVALUATION (PRE-TRAINED MODEL)")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Scaler: {args.scaler_path}")
    print(f"Data: {args.data_csv}")
    print(f"CV Splits: {args.cv_splits_json}")
    print()
    
    # Load pre-trained model and scaler (once)
    # Normalize paths using centralized utility
    model_path = normalize_path(args.model_path)
    model = load_protein_model(model_path, model_type=args.model_type)
    
    scaler_path = normalize_path(args.scaler_path)
    scaler = load_scaler(scaler_path)
    print()
    
    # Load CV splits from fusion experiment
    cv_splits = load_cv_splits(args.cv_splits_json)
    print()
    
    # Load protein data
    print("Loading protein data...")
    df = pd.read_csv(args.data_csv)
    
    # Get protein features (exclude metadata columns)
    metadata_cols = ['RID', 'Subject', 'VISCODE', 'Visit', 'research_group', 'Group', 
                     'Sex', 'Age', 'subject_age', 'Image Data ID', 'Description', 
                     'Type', 'Modality', 'Format', 'Acq Date', 'Downloaded', 'MRI_acquired',
                     'mri_source_path', 'mri_path']
    protein_cols = [col for col in df.columns if col not in metadata_cols]
    
    # Impute missing values with column medians and remove zero-variance features
    protein_df = df[protein_cols].copy()
    
    # Imputation (median per column)
    protein_df = protein_df.fillna(protein_df.median(numeric_only=True))
    
    # Zero-variance removal
    zero_var_cols = protein_df.columns[(protein_df.var(axis=0) == 0)].tolist()
    if zero_var_cols:
        print(f"  Removing {len(zero_var_cols)} zero-variance features")
        protein_df = protein_df.drop(columns=zero_var_cols)
    
    # Align to scaler's training feature order if available
    scaler_features = load_scaler_feature_columns(Path(args.model_path).parents[1]) if args.model_type == 'mlp' else load_scaler_feature_columns(Path(args.scaler_path).parents[0])
    # Note: For both models, scaler is saved in the run dir; attempt to resolve from provided paths
    if scaler_features:
        protein_df = align_features_to_scaler(protein_df, scaler, scaler_features)
    
    X = protein_df.values
    y = (df['research_group'] == 'AD').astype(int).values
    
    print(f"  Loaded {len(X)} samples with {X.shape[1]} protein features")
    print(f"  Class distribution: AD={y.sum()}, CN={(1-y).sum()}")
    print()
    
    # Evaluate on each fold's test set
    fold_results = []
    
    for fold_idx, split in enumerate(cv_splits):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{len(cv_splits)}")
        print(f"{'='*60}")
        
        test_idx = np.array(split['test'])
        
        X_test, y_test = X[test_idx], y[test_idx]
        
        print(f"  Test:  {len(test_idx)} samples (AD={y_test.sum()}, CN={(1-y_test).sum()})")
        
        # Evaluate fold (inference only, no training)
        results = evaluate_fold_test(
            model=model,
            scaler=scaler,
            X_test=X_test,
            y_test=y_test,
            fold_idx=fold_idx,
            model_type=args.model_type
        )
        
        fold_results.append(results)
        print_fold_results(results, fold_idx)
    
    # Aggregate results
    aggregated, total_cm = aggregate_results(fold_results)
    
    # Save results
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save aggregated results
        results_dict = {
            'model_type': args.model_type,
            'model_path': str(args.model_path),
            'scaler_path': str(args.scaler_path),
            'data_csv': args.data_csv,
            'cv_splits_json': args.cv_splits_json,
            'fold_results': fold_results,
            'aggregated_metrics': {k: {'mean': v['mean'], 'std': v['std']} for k, v in aggregated.items()},
            'aggregated_cm': total_cm.tolist()
        }
        
        with open(save_dir / 'protein_baseline_results.json', 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\n✅ Results saved to: {save_dir / 'protein_baseline_results.json'}")
    
    print("\n✅ Protein baseline evaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate pre-trained protein model on fusion CV splits (inference only, no training)"
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        choices=['nn', 'transformer'],
        default='nn',
        help="Type of protein model: 'nn' (PyTorch Neural Network) or 'transformer'"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="src/protein/runs/run_20251016_205054/models/neural_network.pth",
        help="Path to pre-trained protein model (.pth for PyTorch models)"
    )
    parser.add_argument(
        "--scaler-path",
        type=str,
        default="src/protein/runs/run_20251016_205054/scaler.pkl",
        help="Path to pre-fitted scaler (.pkl file)"
    )
    parser.add_argument(
        "--data-csv",
        type=str,
        default="/home/ssim0068/data/multimodal-dataset/all.csv",
        help="Path to CSV with protein data (same as used for fusion)"
    )
    parser.add_argument(
        "--cv-splits-json",
        type=str,
        required=True,
        help="Path to cv_splits.json from fusion experiment"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save baseline results"
    )
    
    args = parser.parse_args()
    main(args)
