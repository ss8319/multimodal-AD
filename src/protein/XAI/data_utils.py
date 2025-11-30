"""
Shared data preparation utilities for XAI modules

This module provides common data loading and preprocessing functions
used by both SHAP and Integrated Gradients explainers.
"""
import pickle
import numpy as np
from pathlib import Path

# Import project modules (relative imports from parent package)
# Fallback to absolute imports if running as script
try:
    from ..dataset import ProteinDataLoader
    from ..feature_utils import load_scaler_feature_columns, align_features_to_scaler
except ImportError:
    import sys
    # When running as script, add src/protein to path
    script_dir = Path(__file__).parent.parent  # src/protein/XAI -> src/protein
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    # Also add src to path for package-style imports
    src_dir = script_dir.parent  # src/protein -> src
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from protein.dataset import ProteinDataLoader
    from protein.feature_utils import load_scaler_feature_columns, align_features_to_scaler


class DataPreparationError(Exception):
    """Custom exception for data preparation errors"""
    pass


def prepare_data(run_dir: str, csv_path: str, label_col: str = 'research_group', 
                 id_col: str = 'RID', is_background: bool = False):
    """
    Prepare data using the same preprocessing pipeline as training.
    
    This function ensures that data used for XAI explanations is preprocessed
    exactly the same way as during training, which is critical for valid explanations.
    
    Args:
        run_dir: Directory with saved scaler and feature order
        csv_path: Path to CSV file (train or test)
        label_col: Label column name
        id_col: Subject ID column name
        is_background: If True, this is background data (train set)
    
    Returns:
        tuple: (X_scaled, feature_names, df)
            - X_scaled: Scaled features ready for XAI (numpy array)
            - feature_names: List of feature names in correct order
            - df: Original dataframe with metadata
    
    Raises:
        DataPreparationError: If scaler or feature columns not found
        FileNotFoundError: If csv_path doesn't exist
    """
    run_dir = Path(run_dir)
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Load scaler
    scaler_path = run_dir / "scaler.pkl"
    if not scaler_path.exists():
        raise DataPreparationError(
            f"Scaler not found: {scaler_path}. "
            f"Make sure you're using a valid run directory with saved models."
        )
    
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    except Exception as e:
        raise DataPreparationError(f"Failed to load scaler from {scaler_path}: {e}") from e
    
    # Load feature columns (order matters!)
    feature_columns = load_scaler_feature_columns(run_dir)
    if feature_columns is None:
        raise DataPreparationError(
            f"Feature columns not found in {run_dir}. "
            f"Expected file: {run_dir / 'scaler_features.json'}"
        )
    
    # Load data
    try:
        data_loader = ProteinDataLoader(
            data_path=str(csv_path),
            label_col=label_col,
            id_col=id_col,
            random_state=42
        )
        df = data_loader.load_data()
    except Exception as e:
        raise DataPreparationError(f"Failed to load data from {csv_path}: {e}") from e
    
    # Extract features (same logic as training)
    exclude_cols = [id_col, label_col, 'VISCODE', 'subject_age', 'Sex', 'Age', 'Education']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Extract features and impute missing values
    X_raw = df[feature_cols].fillna(df[feature_cols].median())
    
    # Align features to training order (critical!)
    try:
        X_aligned = align_features_to_scaler(X_raw, scaler, feature_columns)
    except Exception as e:
        raise DataPreparationError(
            f"Failed to align features to training order: {e}. "
            f"This usually means the test data has different features than training data."
        ) from e
    
    # Scale using training scaler
    try:
        X_scaled = scaler.transform(X_aligned)
    except Exception as e:
        raise DataPreparationError(f"Failed to scale features: {e}") from e
    
    data_type = "background (train)" if is_background else "explained (test)"
    print(f"Prepared {data_type} data: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
    
    return X_scaled, feature_columns, df

