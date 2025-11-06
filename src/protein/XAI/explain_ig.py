"""
Integrated Gradients (IG) explanations for protein classification models
Simple vanilla IG implementation using Captum for Neural Network and Logistic Regression

This module uses Captum's IntegratedGradients as documented at:
- https://captum.ai/api/integrated_gradients.html
- Captum version: 0.6.0+

Key API patterns used:
1. IntegratedGradients: captum.attr.IntegratedGradients(model)
   - attribute() returns: torch.Tensor of shape (n_samples, n_features)
   
2. For binary classification, we compute attributions for class 1 (AD)
"""

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

# Import project modules (relative imports from parent package)
# Fallback to absolute imports if running as script
try:
    from ..dataset import ProteinDataLoader
    from ..model import NeuralNetworkClassifier, NeuralNetwork
    from ..feature_utils import load_scaler_feature_columns, align_features_to_scaler
except ImportError:
    # Fallback for when running as script directly
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dataset import ProteinDataLoader
    from model import NeuralNetworkClassifier, NeuralNetwork
    from feature_utils import load_scaler_feature_columns, align_features_to_scaler


def load_model(run_dir, model_name):
    """Load a saved model from the run directory"""
    models_dir = Path(run_dir) / "models"
    
    if model_name == "neural_network":
        # Load PyTorch model
        model_path = models_dir / "neural_network.pth"
        checkpoint = torch.load(model_path, map_location='cpu')
        model_config = checkpoint['model_config']
        
        # Recreate the wrapper
        n_features = model_config.get('n_features')
        hidden_sizes = model_config.get('hidden_sizes', (128, 64))
        dropout = model_config.get('dropout', 0.2)
        
        wrapper = NeuralNetworkClassifier(
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            random_state=model_config.get('random_state', 42)
        )
        wrapper.n_features = n_features
        
        # Set classes_ (required for sklearn compatibility)
        wrapper.classes_ = np.array([0, 1])  # CN=0, AD=1
        wrapper.n_classes_ = 2
        
        # Recreate and load the neural network
        wrapper.model = NeuralNetwork(n_features=n_features, hidden_sizes=hidden_sizes, dropout=dropout)
        wrapper.model.load_state_dict(checkpoint['model_state_dict'])
        wrapper.model.eval()
        
        return wrapper
    else:
        # Load sklearn model
        model_path = models_dir / f"{model_name}.pkl"
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model


def create_pytorch_lr_model(sklearn_lr, n_features):
    """
    Convert sklearn LogisticRegression to a PyTorch model for Captum
    
    Args:
        sklearn_lr: Trained sklearn LogisticRegression model
        n_features: Number of input features
    
    Returns:
        PyTorch nn.Module equivalent of the logistic regression
    """
    class LogisticRegressionPyTorch(nn.Module):
        def __init__(self, n_features, n_classes=2):
            super().__init__()
            self.linear = nn.Linear(n_features, n_classes)
            
        def forward(self, x):
            return self.linear(x)
    
    pytorch_model = LogisticRegressionPyTorch(n_features, n_classes=2)
    
    # Copy weights from sklearn model
    # sklearn LR: coef_ shape (n_classes, n_features), intercept_ shape (n_classes,)
    with torch.no_grad():
        pytorch_model.linear.weight.data = torch.FloatTensor(sklearn_lr.coef_)
        pytorch_model.linear.bias.data = torch.FloatTensor(sklearn_lr.intercept_)
    
    pytorch_model.eval()
    return pytorch_model


def prepare_data_for_ig(run_dir, csv_path, label_col='research_group', id_col='RID', is_background=False):
    """
    Prepare data using the same preprocessing pipeline as training
    
    Args:
        run_dir: Directory with saved scaler and feature order
        csv_path: Path to CSV file (train or test)
        label_col: Label column name
        id_col: Subject ID column name
        is_background: If True, this is background data (train set)
    
    Returns:
        X_scaled: Scaled features ready for IG (numpy array)
        feature_names: List of feature names in correct order
        df: Original dataframe with metadata
    """
    # Load scaler and feature order
    scaler_path = Path(run_dir) / "scaler.pkl"
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    feature_columns = load_scaler_feature_columns(run_dir)
    if feature_columns is None:
        raise ValueError("Could not load feature columns from scaler_features.json")
    
    # Load data
    data_loader = ProteinDataLoader(
        data_path=csv_path,
        label_col=label_col,
        id_col=id_col,
        random_state=42
    )
    
    df = data_loader.load_data()
    
    # Manually extract features (same logic as prepare_features but skip label encoding)
    exclude_cols = [id_col, label_col, 'VISCODE', 'subject_age', 'Sex', 'Age', 'Education']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Extract features and impute missing values
    X_raw = df[feature_cols].fillna(df[feature_cols].median())
    
    # Align features to training order (critical!)
    X_aligned = align_features_to_scaler(X_raw, scaler, feature_columns)
    
    # Scale using training scaler
    X_scaled = scaler.transform(X_aligned)
    
    data_type = "background (train)" if is_background else "explained (test)"
    print(f"Prepared {data_type} data: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
    return X_scaled, feature_columns, df


def compute_baseline(background, method='mean'):
    """
    Compute baseline for Integrated Gradients
    
    The baseline is the reference point for the path integral in IG.
    IG computes: Attribution = ∫[baseline→input] ∇F(x) · (input - baseline) dx
    
    Common baseline choices:
    - 'mean': Mean of training data (represents "average" patient)
    - 'median': Median of training data (robust to outliers)
    - 'zero': Zero vector (represents "no features")
    
    Args:
        background: Background samples (numpy array) - typically training data
        method: 'mean' (default), 'zero', or 'median'
    
    Returns:
        Baseline tensor (1, n_features) - will broadcast to (n_samples, n_features) in Captum
    """
    if method == 'mean':
        baseline = np.mean(background, axis=0, keepdims=True)
    elif method == 'median':
        baseline = np.median(background, axis=0, keepdims=True)
    elif method == 'zero':
        baseline = np.zeros((1, background.shape[1]))
    else:
        raise ValueError(f"Unknown baseline method: {method}")
    
    # Convert to tensor and ensure it's float32 for Captum compatibility
    baseline_tensor = torch.FloatTensor(baseline)
    
    # Ensure baseline has requires_grad=False (Captum will handle gradient computation)
    baseline_tensor.requires_grad_(False)
    
    return baseline_tensor


def explain_neural_network(model, background, explained, feature_names, output_dir, 
                          baseline_method='mean', n_steps=50, target_class=1):
    """
    Generate Integrated Gradients explanations for Neural Network using Captum
    
    Args:
        model: NeuralNetworkClassifier instance
        background: Background samples (used for baseline computation)
        explained: Samples to explain
        feature_names: List of feature names
        output_dir: Directory to save outputs
        baseline_method: Method for baseline ('mean', 'median', 'zero')
        n_steps: Number of integration steps (default: 50)
        target_class: Which class to explain (1 for AD)
    
    Captum API Reference:
        - IntegratedGradients: https://captum.ai/api/integrated_gradients.html
        - attribute() returns: torch.Tensor of shape (n_samples, n_features)
    """
    print("\n" + "="*60)
    print("INTEGRATED GRADIENTS: Neural Network")
    print("="*60)
    
    # Get the PyTorch model (not the wrapper)
    # IG requires: (1) model outputs logits (not probabilities), (2) model in eval mode
    if not hasattr(model, 'model') or not isinstance(model.model, torch.nn.Module):
        raise ValueError("Model must have a PyTorch .model attribute for Integrated Gradients")
    
    pytorch_model = model.model
    pytorch_model.eval()  # Ensure model is in eval mode (disables dropout, batch norm updates)
    
    # Verify model outputs logits (required for IG gradient computation)
    # NeuralNetwork outputs logits directly (no softmax in forward), which is correct
    
    # Convert to tensors
    background_tensor = torch.FloatTensor(background)
    explained_tensor = torch.FloatTensor(explained)
    
    # Compute baseline
    baseline = compute_baseline(background, method=baseline_method)
    print(f"   Baseline method: {baseline_method}")
    print(f"   Baseline shape: {baseline.shape}")
    print(f"   Explained samples: {len(explained)}")
    
    # Initialize Integrated Gradients explainer
    # Captum API: IntegratedGradients(model)
    ig = IntegratedGradients(pytorch_model)
    
    # Compute attributions
    # Captum API: attribute(inputs, baselines, target=None, n_steps=50, ...)
    # For binary classification, target=1 means we're explaining class 1 (AD)
    # 
    # IG Theory: Attribution = ∫[baseline→input] ∇F(x) · (input - baseline) dx
    # where F(x) is the model output (logit) for the target class
    # 
    # The completeness axiom states: sum(attributions) ≈ F(input) - F(baseline)
    print(f"   Computing IG attributions for class {target_class} (AD)...")
    print(f"   Using {n_steps} integration steps...")
    
    # Enable convergence delta to validate approximation quality
    # Convergence delta measures approximation error (should be small for good approximation)
    attributions, convergence_delta = ig.attribute(
        explained_tensor,
        baselines=baseline,
        target=target_class,
        n_steps=n_steps,
        return_convergence_delta=True  # Validate approximation quality
    )
    
    # Convert to numpy
    if isinstance(attributions, torch.Tensor):
        attributions_np = attributions.detach().cpu().numpy()
    else:
        attributions_np = np.asarray(attributions)
    
    if isinstance(convergence_delta, torch.Tensor):
        convergence_delta_np = convergence_delta.detach().cpu().numpy()
    else:
        convergence_delta_np = np.asarray(convergence_delta)
    
    print(f"   Attributions shape: {attributions_np.shape} ✓")
    print(f"   (Expected: {explained.shape[0]} samples × {explained.shape[1]} features)")
    print(f"   Convergence delta (approximation error): mean={convergence_delta_np.mean():.6f}, "
          f"max={convergence_delta_np.max():.6f}")
    print(f"   (Smaller is better; typically < 0.01 indicates good approximation)")
    
    # Validate completeness axiom: sum(attributions) ≈ F(input) - F(baseline)
    # This is a sanity check that IG is working correctly
    with torch.no_grad():
        input_logits = pytorch_model(explained_tensor)[:, target_class]
        baseline_logits = pytorch_model(baseline)[:, target_class]
        predicted_diff = (input_logits - baseline_logits).cpu().numpy()
        attribution_sums = attributions_np.sum(axis=1)
        completeness_error = np.abs(predicted_diff - attribution_sums)
        print(f"   Completeness check: mean error={completeness_error.mean():.6f}, "
              f"max error={completeness_error.max():.6f}")
        print(f"   (Should be close to 0; validates IG implementation)")
    
    # Create plots
    plot_dir = Path(output_dir) / "ig" / "neural_network"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary plot (beeswarm-like, similar to SHAP)
    print("   Creating summary plot (beeswarm)...")
    plt.figure(figsize=(10, 12))
    
    # Compute mean absolute attributions for feature importance
    mean_abs_attr = np.abs(attributions_np).mean(axis=0)
    feature_order = np.argsort(mean_abs_attr)[::-1]
    
    # Create beeswarm-like plot
    fig, ax = plt.subplots(figsize=(10, 12))
    y_pos = np.arange(len(feature_order[:20]))  # Top 20 features
    colors = ['red' if mean_abs_attr[i] > 0 else 'blue' for i in feature_order[:20]]
    
    ax.barh(y_pos, mean_abs_attr[feature_order[:20]], color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in feature_order[:20]], fontsize=9)
    ax.set_xlabel('Mean |Attribution|', fontsize=11)
    ax.set_title('Integrated Gradients: Top 20 Feature Importances', fontsize=12)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(plot_dir / "summary_beeswarm.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Summary plot (bar)
    print("   Creating summary plot (bar)...")
    plt.figure(figsize=(10, 8))
    top_indices = np.argsort(mean_abs_attr)[-20:][::-1]
    top_names = [feature_names[i] for i in top_indices]
    top_values = mean_abs_attr[top_indices]
    
    plt.barh(range(len(top_names)), top_values, alpha=0.7)
    plt.yticks(range(len(top_names)), top_names, fontsize=9)
    plt.xlabel('Mean |Attribution|', fontsize=11)
    plt.title('Integrated Gradients: Top 20 Feature Importances (Bar)', fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(plot_dir / "summary_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Dependence plots for top 5 features
    print("   Creating dependence plots for top 5 features...")
    top_5_indices = top_indices[:5]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, feat_idx in enumerate(top_5_indices):
        ax = axes[idx]
        feat_values = explained[:, feat_idx]
        feat_attr = attributions_np[:, feat_idx]
        
        ax.scatter(feat_values, feat_attr, alpha=0.6, s=30)
        ax.set_xlabel(feature_names[feat_idx], fontsize=9)
        ax.set_ylabel('Attribution', fontsize=9)
        ax.set_title(f"{feature_names[feat_idx]}", fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Remove extra subplot
    fig.delaxes(axes[5])
    plt.tight_layout()
    plt.savefig(plot_dir / "dependence_plots_top5.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save attributions and validation metrics
    np.save(plot_dir / "ig_attributions.npy", attributions_np)
    np.save(plot_dir / "convergence_delta.npy", convergence_delta_np)
    
    # Save completeness validation metrics
    completeness_metrics = {
        'mean_error': float(completeness_error.mean()),
        'max_error': float(completeness_error.max()),
        'mean_convergence_delta': float(convergence_delta_np.mean()),
        'max_convergence_delta': float(convergence_delta_np.max())
    }
    import json
    with open(plot_dir / "validation_metrics.json", 'w') as f:
        json.dump(completeness_metrics, f, indent=2)
    
    print(f"\n   Saved IG plots, attributions, and validation metrics to: {plot_dir}")
    
    return attributions_np


def explain_logistic_regression(model, background, explained, feature_names, output_dir,
                                baseline_method='mean', n_steps=50, target_class=1):
    """
    Generate Integrated Gradients explanations for Logistic Regression using Captum
    
    Args:
        model: sklearn LogisticRegression instance
        background: Background samples (used for baseline computation)
        explained: Samples to explain
        feature_names: List of feature names
        output_dir: Directory to save outputs
        baseline_method: Method for baseline ('mean', 'median', 'zero')
        n_steps: Number of integration steps (default: 50)
        target_class: Which class to explain (1 for AD)
    
    Note: Logistic Regression is converted to a PyTorch model for Captum compatibility
    """
    print("\n" + "="*60)
    print("INTEGRATED GRADIENTS: Logistic Regression")
    print("="*60)
    
    # Convert sklearn LR to PyTorch model
    # IG requires: (1) model outputs logits (not probabilities), (2) model in eval mode
    n_features = explained.shape[1]
    print(f"   Converting sklearn LogisticRegression to PyTorch model...")
    pytorch_model = create_pytorch_lr_model(model, n_features)
    pytorch_model.eval()  # Ensure model is in eval mode
    
    # LogisticRegressionPyTorch outputs logits directly (linear layer only), which is correct for IG
    
    # Convert to tensors
    background_tensor = torch.FloatTensor(background)
    explained_tensor = torch.FloatTensor(explained)
    
    # Compute baseline
    baseline = compute_baseline(background, method=baseline_method)
    print(f"   Baseline method: {baseline_method}")
    print(f"   Baseline shape: {baseline.shape}")
    print(f"   Explained samples: {len(explained)}")
    
    # Initialize Integrated Gradients explainer
    ig = IntegratedGradients(pytorch_model)
    
    # Compute attributions
    # IG Theory: Attribution = ∫[baseline→input] ∇F(x) · (input - baseline) dx
    # where F(x) is the model output (logit) for the target class
    print(f"   Computing IG attributions for class {target_class} (AD)...")
    print(f"   Using {n_steps} integration steps...")
    
    # Enable convergence delta to validate approximation quality
    attributions, convergence_delta = ig.attribute(
        explained_tensor,
        baselines=baseline,
        target=target_class,
        n_steps=n_steps,
        return_convergence_delta=True  # Validate approximation quality
    )
    
    # Convert to numpy
    if isinstance(attributions, torch.Tensor):
        attributions_np = attributions.detach().cpu().numpy()
    else:
        attributions_np = np.asarray(attributions)
    
    if isinstance(convergence_delta, torch.Tensor):
        convergence_delta_np = convergence_delta.detach().cpu().numpy()
    else:
        convergence_delta_np = np.asarray(convergence_delta)
    
    print(f"   Attributions shape: {attributions_np.shape} ✓")
    print(f"   Convergence delta (approximation error): mean={convergence_delta_np.mean():.6f}, "
          f"max={convergence_delta_np.max():.6f}")
    print(f"   (Smaller is better; typically < 0.01 indicates good approximation)")
    
    # Validate completeness axiom: sum(attributions) ≈ F(input) - F(baseline)
    with torch.no_grad():
        input_logits = pytorch_model(explained_tensor)[:, target_class]
        baseline_logits = pytorch_model(baseline)[:, target_class]
        predicted_diff = (input_logits - baseline_logits).cpu().numpy()
        attribution_sums = attributions_np.sum(axis=1)
        completeness_error = np.abs(predicted_diff - attribution_sums)
        print(f"   Completeness check: mean error={completeness_error.mean():.6f}, "
              f"max error={completeness_error.max():.6f}")
        print(f"   (Should be close to 0; validates IG implementation)")
    
    # Create plots
    plot_dir = Path(output_dir) / "ig" / "logistic_regression"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary plot (beeswarm-like)
    print("   Creating summary plot (beeswarm)...")
    mean_abs_attr = np.abs(attributions_np).mean(axis=0)
    feature_order = np.argsort(mean_abs_attr)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 12))
    y_pos = np.arange(len(feature_order[:20]))
    colors = ['red' if mean_abs_attr[i] > 0 else 'blue' for i in feature_order[:20]]
    
    ax.barh(y_pos, mean_abs_attr[feature_order[:20]], color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in feature_order[:20]], fontsize=9)
    ax.set_xlabel('Mean |Attribution|', fontsize=11)
    ax.set_title('Integrated Gradients: Top 20 Feature Importances', fontsize=12)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(plot_dir / "summary_beeswarm.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Summary plot (bar)
    print("   Creating summary plot (bar)...")
    plt.figure(figsize=(10, 8))
    top_indices = np.argsort(mean_abs_attr)[-20:][::-1]
    top_names = [feature_names[i] for i in top_indices]
    top_values = mean_abs_attr[top_indices]
    
    plt.barh(range(len(top_names)), top_values, alpha=0.7)
    plt.yticks(range(len(top_names)), top_names, fontsize=9)
    plt.xlabel('Mean |Attribution|', fontsize=11)
    plt.title('Integrated Gradients: Top 20 Feature Importances (Bar)', fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(plot_dir / "summary_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Dependence plots for top 5 features
    print("   Creating dependence plots for top 5 features...")
    top_5_indices = top_indices[:5]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, feat_idx in enumerate(top_5_indices):
        ax = axes[idx]
        feat_values = explained[:, feat_idx]
        feat_attr = attributions_np[:, feat_idx]
        
        ax.scatter(feat_values, feat_attr, alpha=0.6, s=30)
        ax.set_xlabel(feature_names[feat_idx], fontsize=9)
        ax.set_ylabel('Attribution', fontsize=9)
        ax.set_title(f"{feature_names[feat_idx]}", fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Remove extra subplot
    fig.delaxes(axes[5])
    plt.tight_layout()
    plt.savefig(plot_dir / "dependence_plots_top5.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save attributions and validation metrics
    np.save(plot_dir / "ig_attributions.npy", attributions_np)
    np.save(plot_dir / "convergence_delta.npy", convergence_delta_np)
    
    # Save completeness validation metrics
    completeness_metrics = {
        'mean_error': float(completeness_error.mean()),
        'max_error': float(completeness_error.max()),
        'mean_convergence_delta': float(convergence_delta_np.mean()),
        'max_convergence_delta': float(convergence_delta_np.max())
    }
    import json
    with open(plot_dir / "validation_metrics.json", 'w') as f:
        json.dump(completeness_metrics, f, indent=2)
    
    print(f"\n   Saved IG plots, attributions, and validation metrics to: {plot_dir}")
    
    return attributions_np


def main():
    """Main function to generate Integrated Gradients explanations"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Integrated Gradients explanations for protein classification models")
    parser.add_argument("--run-dir", type=str, 
                       default="src/protein/runs/run_20251016_205054",
                       help="Path to run directory with saved models")
    parser.add_argument("--train-data", type=str,
                       default="src/data/protein/proteomic_encoder_train.csv",
                       help="Path to training data CSV (used as background/reference)")
    parser.add_argument("--test-data", type=str,
                       default="src/data/protein/proteomic_encoder_test.csv",
                       help="Path to test data CSV (samples to explain)")
    parser.add_argument("--label-col", type=str, default="research_group",
                       help="Label column name")
    parser.add_argument("--id-col", type=str, default="RID",
                       help="Subject ID column name")
    parser.add_argument("--background-size", type=int, default=None,
                       help="Number of samples for background (default: use full training set)")
    parser.add_argument("--explained-size", type=int, default=None,
                       help="Number of samples to explain (default: use full test set, use -1 for all)")
    parser.add_argument("--random-state", type=int, default=42,
                       help="Random seed for sampling")
    parser.add_argument("--baseline-method", type=str, default="mean", 
                       choices=["mean", "median", "zero"],
                       help="Baseline computation method: 'mean' (default), 'median', or 'zero'")
    parser.add_argument("--n-steps", type=int, default=50,
                       help="Number of integration steps (default: 50, higher = more accurate but slower)")
    parser.add_argument("--target-class", type=int, default=1,
                       help="Target class to explain (1 for AD, 0 for CN)")
    
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise ValueError(f"Run directory not found: {run_dir}")
    
    print("="*70)
    print("INTEGRATED GRADIENTS EXPLANATIONS FOR PROTEIN CLASSIFICATION MODELS")
    print("="*70)
    print(f"Run directory: {run_dir}")
    print(f"Train data (background): {args.train_data}")
    print(f"Test data (explained): {args.test_data}")
    print(f"Baseline method: {args.baseline_method}")
    print(f"Integration steps: {args.n_steps}")
    print(f"Target class: {args.target_class} ({'AD' if args.target_class == 1 else 'CN'})")
    
    # Prepare data
    print("\nPREPARING DATA")
    print("-"*70)
    
    # Prepare training data for background (reference distribution)
    print("Loading training data for background...")
    X_train_scaled, feature_names, train_df = prepare_data_for_ig(
        run_dir, args.train_data, args.label_col, args.id_col, is_background=True
    )
    
    # Prepare test data for explanations
    print("\nLoading test data for explanations...")
    X_test_scaled, _, test_df = prepare_data_for_ig(
        run_dir, args.test_data, args.label_col, args.id_col, is_background=False
    )
    
    # Determine background and explained sets
    np.random.seed(args.random_state)
    n_train = len(X_train_scaled)
    n_test = len(X_test_scaled)
    
    # Background set (from training data)
    if args.background_size is not None:
        background_size = min(args.background_size, n_train)
        background_indices = np.random.choice(n_train, background_size, replace=False)
        background = X_train_scaled[background_indices]
        print(f"   Background: {background_size} samples (randomly sampled from {n_train} training samples)")
    else:
        # Default: use full training set (recommended for stable baseline)
        background = X_train_scaled
        print(f"   Background: ALL {n_train} training samples ✓ (recommended)")
    
    # Explained set (from test data)
    if args.explained_size is not None:
        if args.explained_size == -1:
            explained = X_test_scaled
            print(f"   Explained: ALL {n_test} test samples")
        else:
            explained_size = min(args.explained_size, n_test)
            explained_indices = np.random.choice(n_test, explained_size, replace=False)
            explained = X_test_scaled[explained_indices]
            print(f"   Explained: {explained_size} samples (randomly sampled from {n_test} test samples)")
    else:
        # Default: use full test set (recommended)
        explained = X_test_scaled
        print(f"   Explained: ALL {n_test} test samples ✓ (recommended)")
    
    print(f"\n   Final setup:")
    print(f"   Background set: {len(background)} samples (for baseline computation)")
    print(f"   Explained set: {len(explained)} samples (will compute IG attributions for these)")
    
    # Load models
    print("\nLOADING MODELS")
    print("-"*70)
    try:
        lr_model = load_model(run_dir, "logistic_regression")
        print("   ✓ Loaded Logistic Regression")
    except Exception as e:
        print(f"   ✗ Failed to load Logistic Regression: {e}")
        lr_model = None
    
    try:
        nn_model = load_model(run_dir, "neural_network")
        print("   ✓ Loaded Neural Network")
    except Exception as e:
        print(f"   ✗ Failed to load Neural Network: {e}")
        nn_model = None
    
    # Generate IG explanations
    print("\nGENERATING INTEGRATED GRADIENTS EXPLANATIONS")
    print("-"*70)
    
    if lr_model is not None:
        try:
            explain_logistic_regression(
                lr_model, background, explained, feature_names, run_dir,
                baseline_method=args.baseline_method,
                n_steps=args.n_steps,
                target_class=args.target_class
            )
        except Exception as e:
            print(f"   ✗ Error explaining Logistic Regression: {e}")
            import traceback
            traceback.print_exc()
    
    if nn_model is not None:
        try:
            explain_neural_network(
                nn_model, background, explained, feature_names, run_dir,
                baseline_method=args.baseline_method,
                n_steps=args.n_steps,
                target_class=args.target_class
            )
        except Exception as e:
            print(f"   ✗ Error explaining Neural Network: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("INTEGRATED GRADIENTS EXPLANATION COMPLETE!")
    print("="*70)
    print(f"\nResults saved to:")
    print(f"  {run_dir / 'ig' / 'logistic_regression'}")
    print(f"  {run_dir / 'ig' / 'neural_network'}")

if __name__ == "__main__":
    main()

