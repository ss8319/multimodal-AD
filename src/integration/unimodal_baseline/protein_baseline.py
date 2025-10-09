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


def load_cv_splits(cv_splits_path):
    """Load CV splits from fusion experiment"""
    with open(cv_splits_path, 'r') as f:
        cv_splits = json.load(f)
    
    print(f"Loaded {len(cv_splits)} CV folds from {cv_splits_path}")
    return cv_splits


def load_protein_model(model_path, model_type='mlp'):
    """Load pre-trained protein model
    
    Args:
        model_path: Path to model file (.pkl for MLP, .pth for Transformer)
        model_type: 'mlp' or 'transformer'
    """
    model_path = Path(model_path)
    
    if model_type == 'mlp':
        # Load sklearn MLP model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Loaded pre-trained MLP model from {model_path}")
        print(f"  Model type: {type(model).__name__}")
        return model
    
    elif model_type == 'transformer':
        # Load PyTorch Transformer model
        import torch
        import sys
        from pathlib import Path as P
        
        # Add protein module to path for model import
        protein_path = str(P(__file__).parent.parent.parent / "protein")
        if protein_path not in sys.path:
            sys.path.insert(0, protein_path)
        
        from model import ProteinTransformer
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Recreate model with saved config (consistent with extract_latents.py and runs/README.md)
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
        raise ValueError(f"Unsupported model_type: {model_type}. Use 'mlp' or 'transformer'")


def load_scaler(scaler_path):
    """Load pre-fitted scaler"""
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"Loaded pre-fitted scaler from {scaler_path}")
    return scaler


def evaluate_fold_test(model, scaler, X_test, y_test, fold_idx, model_type='mlp'):
    """Evaluate pre-trained model on test set for a single fold
    
    Args:
        model: Pre-trained model (sklearn or PyTorch)
        scaler: Pre-fitted StandardScaler
        X_test: Test features
        y_test: Test labels
        fold_idx: Fold index
        model_type: 'mlp' or 'transformer'
    """
    
    # Apply pre-fitted scaler to test data
    X_test_scaled = scaler.transform(X_test)
    
    # Run inference (no training)
    if model_type == 'mlp':
        # sklearn model
        test_preds = model.predict(X_test_scaled)
        test_probs = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else test_preds
    
    elif model_type == 'transformer':
        # PyTorch model
        import torch
        
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test_scaled)
            logits = model(X_test_tensor)
            test_probs = torch.softmax(logits, dim=1)[:, 1].numpy()
            test_preds = torch.argmax(logits, dim=1).numpy()
    
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    
    # Calculate metrics
    test_acc = accuracy_score(y_test, test_preds)
    test_balanced_acc = balanced_accuracy_score(y_test, test_preds)
    
    # Calculate test AUC with proper handling
    n_unique_preds = len(np.unique(test_preds))
    n_unique_labels = len(np.unique(y_test))
    
    if n_unique_labels < 2:
        test_auc = float('nan')
        print(f"  ⚠️  Warning: Only one class in test labels - AUC undefined")
    elif n_unique_preds < 2:
        test_auc = 0.5
        print(f"  ⚠️  Warning: Model predicting only ONE class - Using AUC = 0.5")
    else:
        try:
            test_auc = roc_auc_score(y_test, test_probs)
        except ValueError:
            test_auc = float('nan')
            print(f"  ⚠️  Warning: AUC calculation failed")
    
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
    model_path = Path(args.model_path)
    model = load_protein_model(model_path, model_type=args.model_type)
    
    scaler_path = Path(args.scaler_path)
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
    
    X = df[protein_cols].values
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
        choices=['mlp', 'transformer'],
        default='mlp',
        help="Type of protein model: 'mlp' or 'transformer'"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="src/protein/runs/run_20251003_133215/models/neural_network.pkl",
        help="Path to pre-trained protein model (.pkl for MLP, .pth for Transformer)"
    )
    parser.add_argument(
        "--scaler-path",
        type=str,
        default="src/protein/runs/run_20251003_133215/scaler.pkl",
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
