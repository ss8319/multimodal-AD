"""
SHAP explanations for protein classification models
Simple vanilla SHAP implementation for Logistic Regression and Neural Network

This module follows SHAP's official API as documented at:
- https://github.com/slundberg/shap
- SHAP version: 0.49.1

Key API patterns used:
1. LinearExplainer: shap.LinearExplainer(model, background_data)
   - shap_values() returns: list[np.ndarray] for binary classification
   
2. DeepExplainer: shap.DeepExplainer(pytorch_model, background_tensor)
   - shap_values() returns: np.ndarray shape (n_samples, n_features, n_classes) for binary classification
   
3. KernelExplainer: shap.KernelExplainer(predict_fn, background_data)
   - shap_values() returns: np.ndarray shape (n_samples, n_features)
"""
import numpy as np
import pandas as pd
import pickle
import torch
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler

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


def prepare_data_for_shap(run_dir, csv_path, label_col='research_group', id_col='RID', is_background=False):
    """
    Prepare data using the same preprocessing pipeline as training
    
    Args:
        run_dir: Directory with saved scaler and feature order
        csv_path: Path to CSV file (train or test)
        label_col: Label column name
        id_col: Subject ID column name
        is_background: If True, this is background data (train set)
    
    Returns:
        X_scaled: Scaled features ready for SHAP (numpy array)
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
    # We don't need labels for SHAP explanations
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


def create_prediction_function(model, class_idx=1):
    """
    Create a prediction function for SHAP that returns probabilities for class_idx
    
    Args:
        model: Trained model (sklearn or NeuralNetworkClassifier)
        class_idx: Which class probability to return (1 for AD)
    
    Returns:
        Function that takes X and returns probabilities for class_idx
    """
    def predict_fn(X):
        # Convert to numpy if it's a tensor (SHAP may pass tensors)
        if isinstance(X, torch.Tensor):
            X_np = X.numpy()
        else:
            X_np = np.asarray(X)
        
        # Use the model's predict_proba method (works for both sklearn and our wrapper)
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_np)
        else:
            # Fallback: use predict and convert to probabilities
            pred = model.predict(X_np)
            proba = np.column_stack([1 - pred, pred])
        
        return proba[:, class_idx]
    
    return predict_fn


def explain_logistic_regression(model, background, explained, feature_names, output_dir):
    """
    Generate SHAP explanations for Logistic Regression using SHAP's LinearExplainer
    
    SHAP API Reference:
        - LinearExplainer: https://github.com/slundberg/shap
        - shap_values() returns numpy.ndarray of shape (n_samples, n_features) for binary classification
    """
    print("\n" + "="*60)
    print("SHAP EXPLANATIONS: Logistic Regression")
    print("="*60)
    
    # Initialize explainer per SHAP API: LinearExplainer(model, data)
    explainer = shap.LinearExplainer(model, background)
    
    # Compute SHAP values per SHAP API: explainer.shap_values(X)
    # SHAP API: For binary classification, returns list [shap_class_0, shap_class_1]
    print(f"Computing SHAP values for {len(explained)} samples...")
    shap_values_raw = explainer.shap_values(explained)
    
    # SHAP API return format for LinearExplainer with binary classification:
    # Returns list of numpy arrays [class_0_SHAP, class_1_SHAP]
    # Each array has shape (n_samples, n_features)
    if isinstance(shap_values_raw, list):
        if len(shap_values_raw) != 2:
            raise ValueError(f"Expected 2 classes for binary classification, got {len(shap_values_raw)}")
        shap_values = shap_values_raw[1]  # Use class 1 (AD) - second element
    else:
        # Regression or unexpected format
        shap_values = np.asarray(shap_values_raw)
    
    # Create plots
    plot_dir = Path(output_dir) / "shap" / "logistic_regression"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary plot (beeswarm)
    print("   Creating summary plot (beeswarm)...")
    plt.figure(figsize=(10, 12))
    shap.summary_plot(shap_values, explained, feature_names=feature_names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(plot_dir / "summary_beeswarm.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Summary plot (bar)
    print("   Creating summary plot (bar)...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, explained, feature_names=feature_names, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(plot_dir / "summary_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Dependence plots for top 5 features
    print("   Creating dependence plots for top 5 features...")
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-5:][::-1]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, feat_idx in enumerate(top_indices[:5]):
        shap.dependence_plot(
            feat_idx, shap_values, explained,
            feature_names=feature_names,
            ax=axes[idx],
            show=False
        )
        axes[idx].set_title(f"{feature_names[feat_idx]}", fontsize=10)
    
    # Remove extra subplot
    fig.delaxes(axes[5])
    plt.tight_layout()
    plt.savefig(plot_dir / "dependence_plots_top5.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save SHAP values
    np.save(plot_dir / "shap_values.npy", shap_values)
    print(f"\n   Saved SHAP plots and values to: {plot_dir}")
    
    return shap_values


def explain_neural_network(model, background, explained, feature_names, output_dir, use_deep=True):
    """
    Generate SHAP explanations for Neural Network using SHAP's official API
    
    Args:
        model: NeuralNetworkClassifier instance
        background: Background samples (full training set recommended)
        explained: Samples to explain (full test set recommended)
        feature_names: List of feature names
        output_dir: Directory to save outputs
        use_deep: If True, use DeepExplainer (recommended), else use KernelExplainer
    
    SHAP API Reference:
        - DeepExplainer: https://github.com/slundberg/shap
        - shap_values() returns:
          * For binary classification: numpy.ndarray of shape (n_samples, n_features, n_classes)
          * For multi-class: list of numpy arrays, one per class
          * For regression: numpy.ndarray of shape (n_samples, n_features)
    """
    print("\n" + "="*60)
    print("SHAP EXPLANATIONS: Neural Network")
    print("="*60)
    
    if use_deep:
        # Use DeepExplainer (recommended for neural networks)
        print("   Using DeepExplainer (recommended for neural networks)...")
        print("   This is faster and more accurate than KernelExplainer")
        
        # Get the PyTorch model (not the wrapper)
        if not hasattr(model, 'model') or not isinstance(model.model, torch.nn.Module):
            raise ValueError("Model must have a PyTorch .model attribute for DeepExplainer")
        
        pytorch_model = model.model
        pytorch_model.eval()  # Ensure model is in eval mode
        
        # Convert background and explained to tensors
        # Use full sets for consistency with other methods
        # SHAP API: DeepExplainer(model, background_data) where background_data is torch.Tensor or numpy array
        background_tensor = torch.FloatTensor(background)
        explained_tensor = torch.FloatTensor(explained)
        
        print(f"   Background: {len(background)} samples (full training set)")
        print(f"   Explained: {len(explained)} samples (full test set)")
        
        # Initialize explainer per SHAP API: DeepExplainer(model, background_data)
        explainer = shap.DeepExplainer(pytorch_model, background_tensor)
        
        # Compute SHAP values per SHAP API: explainer.shap_values(X, ranked_outputs=None, ...)
        # API doc: https://github.com/slundberg/shap
        print(f"   Computing SHAP values for {len(explained)} samples...")
        shap_values_raw = explainer.shap_values(explained_tensor, check_additivity=True)
        
        # SHAP API return format (from SHAP 0.49.1 documentation and source):
        # - For binary classification with 2 output classes: 
        #   Returns numpy.ndarray of shape (n_samples, n_features, n_classes) = (23, 320, 2)
        # - For multi-class (>2 classes):
        #   Returns list of numpy arrays, one per class
        # - For regression (single output):
        #   Returns numpy.ndarray of shape (n_samples, n_features)
        
        # Convert to numpy array if needed (SHAP may return tensors)
        if isinstance(shap_values_raw, torch.Tensor):
            shap_values_raw = shap_values_raw.cpu().numpy()
        elif not isinstance(shap_values_raw, np.ndarray) and not isinstance(shap_values_raw, list):
            shap_values_raw = np.asarray(shap_values_raw)
        
        # Handle SHAP API return formats according to official behavior
        if isinstance(shap_values_raw, list):
            # Multi-class format (>2 classes): list of arrays
            # Each element is shape (n_samples, n_features)
            if len(shap_values_raw) != 2:
                raise ValueError(f"Expected 2 classes for binary classification, got {len(shap_values_raw)}")
            shap_values = shap_values_raw[1]  # Class 1 (AD) - second element
        elif isinstance(shap_values_raw, np.ndarray):
            if shap_values_raw.ndim == 3:
                # Binary classification: shape (n_samples, n_features, n_classes)
                # Extract class 1 (AD) - last dimension index 1
                if shap_values_raw.shape[2] != 2:
                    raise ValueError(f"Expected 2 classes, got shape {shap_values_raw.shape}")
                shap_values = shap_values_raw[:, :, 1]  # Class 1 (AD)
            elif shap_values_raw.ndim == 2:
                # Regression or single output: shape (n_samples, n_features)
                shap_values = shap_values_raw
            else:
                raise ValueError(f"Unexpected SHAP values ndim: {shap_values_raw.ndim}, shape: {shap_values_raw.shape}")
        else:
            raise TypeError(f"SHAP returned unexpected type: {type(shap_values_raw)}")
        
        # Verify output shape matches expected format per SHAP API
        expected_shape = (explained.shape[0], explained.shape[1])
        if shap_values.shape != expected_shape:
            raise ValueError(
                f"SHAP values shape mismatch. Got {shap_values.shape}, expected {expected_shape}. "
                f"This may indicate an issue with SHAP API usage or model output format."
            )
        
        print(f"   SHAP values shape: {shap_values.shape} ✓ (per SHAP API: {explained.shape[0]} samples × {explained.shape[1]} features)")
        print("   ✓ DeepExplainer completed successfully")
        
    else:
        # Use KernelExplainer (model-agnostic fallback)
        # SHAP API: KernelExplainer(model.predict_fn, background_data)
        print("   Using KernelExplainer (model-agnostic)...")
        
        # Create prediction function for class 1 (AD)
        # SHAP API requires: predict_fn(X) -> array of predictions
        predict_fn = create_prediction_function(model, class_idx=1)
        
        # Use full background set for consistency
        print(f"   Background: {len(background)} samples (full training set)")
        print(f"   Explained: {len(explained)} samples (full test set)")
        
        # Initialize explainer per SHAP API: KernelExplainer(model_fn, data)
        explainer = shap.KernelExplainer(predict_fn, background)
        
        # Compute SHAP values per SHAP API: explainer.shap_values(X, nsamples=100)
        # API doc: Returns numpy.ndarray of shape (n_samples, n_features)
        print(f"   Computing SHAP values for {len(explained)} samples...")
        print("   (This may take longer than DeepExplainer)")
        shap_values = explainer.shap_values(explained, nsamples=200)
        
        # KernelExplainer always returns numpy.ndarray of shape (n_samples, n_features)
        # No need to handle different formats
        shap_values = np.asarray(shap_values)
    
    # Ensure shap_values is numpy array
    if isinstance(shap_values, torch.Tensor):
        shap_values = shap_values.cpu().numpy()
    
    # Create plots
    plot_dir = Path(output_dir) / "shap" / "neural_network"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary plot (beeswarm)
    print("   Creating summary plot (beeswarm)...")
    plt.figure(figsize=(10, 12))
    shap.summary_plot(shap_values, explained, feature_names=feature_names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(plot_dir / "summary_beeswarm.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Summary plot (bar)
    print("   Creating summary plot (bar)...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, explained, feature_names=feature_names, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(plot_dir / "summary_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Dependence plots for top 5 features
    print("   Creating dependence plots for top 5 features...")
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-5:][::-1]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, feat_idx in enumerate(top_indices[:5]):
        shap.dependence_plot(
            feat_idx, shap_values, explained,
            feature_names=feature_names,
            ax=axes[idx],
            show=False
        )
        axes[idx].set_title(f"{feature_names[feat_idx]}", fontsize=10)
    
    # Remove extra subplot
    fig.delaxes(axes[5])
    plt.tight_layout()
    plt.savefig(plot_dir / "dependence_plots_top5.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save SHAP values
    np.save(plot_dir / "shap_values.npy", shap_values)
    print(f"\n   Saved SHAP plots and values to: {plot_dir}")
    
    return shap_values


def main():
    """Main function to generate SHAP explanations"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate SHAP explanations for protein classification models")
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
    parser.add_argument("--nn-explainer", type=str, default="deep", choices=["deep", "kernel"],
                       help="SHAP explainer for Neural Network: 'deep' (recommended, faster) or 'kernel' (slower but model-agnostic)")
    
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise ValueError(f"Run directory not found: {run_dir}")
    
    print("="*70)
    print("SHAP EXPLANATIONS FOR PROTEIN CLASSIFICATION MODELS")
    print("="*70)
    print(f"Run directory: {run_dir}")
    print(f"Train data (background): {args.train_data}")
    print(f"Test data (explained): {args.test_data}")
    
    # Prepare data
    print("\nPREPARING DATA")
    print("-"*70)
    
    # Prepare training data for background (reference distribution)
    print("Loading training data for background...")
    X_train_scaled, feature_names, train_df = prepare_data_for_shap(
        run_dir, args.train_data, args.label_col, args.id_col, is_background=True
    )
    
    # Prepare test data for explanations
    print("\nLoading test data for explanations...")
    X_test_scaled, _, test_df = prepare_data_for_shap(
        run_dir, args.test_data, args.label_col, args.id_col, is_background=False
    )
    
    # Determine background and explained sets
    np.random.seed(args.random_state)
    n_train = len(X_train_scaled)
    n_test = len(X_test_scaled)
    
    # Background set (from training data)
    # Default: use full training set (best practice)
    if args.background_size is not None:
        # User specified size
        background_size = min(args.background_size, n_train)
        background_indices = np.random.choice(n_train, background_size, replace=False)
        background = X_train_scaled[background_indices]
        print(f"   Background: {background_size} samples (randomly sampled from {n_train} training samples)")
    else:
        # Default: use full training set (recommended for stable reference)
        background = X_train_scaled
        print(f"   Background: ALL {n_train} training samples ✓ (recommended)")
    
    # Explained set (from test data)
    # Default: use full test set (best practice)
    if args.explained_size is not None:
        # User specified size
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
    print(f"   Background set: {len(background)} samples (reference distribution)")
    print(f"   Explained set: {len(explained)} samples (will compute SHAP values for these)")
    
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
    
    # Generate SHAP explanations
    print("\nGENERATING SHAP EXPLANATIONS")
    print("-"*70)
    
    if lr_model is not None:
        try:
            explain_logistic_regression(lr_model, background, explained, feature_names, run_dir)
        except Exception as e:
            print(f"   ✗ Error explaining Logistic Regression: {e}")
            import traceback
            traceback.print_exc()
    
    if nn_model is not None:
        try:
            use_deep = args.nn_explainer == "deep"
            explain_neural_network(nn_model, background, explained, feature_names, run_dir, 
                                  use_deep=use_deep)
        except Exception as e:
            print(f"   ✗ Error explaining Neural Network: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("SHAP EXPLANATION COMPLETE!")
    print("="*70)
    print(f"\nResults saved to:")
    print(f"  {run_dir / 'shap' / 'logistic_regression'}")
    print(f"  {run_dir / 'shap' / 'neural_network'}")


if __name__ == "__main__":
    main()

