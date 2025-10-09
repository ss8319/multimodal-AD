import json
import pandas as pd
from pathlib import Path


def load_scaler_feature_columns(run_dir):
    """Load saved training feature order from scaler_features.json.
    Returns list[str] or None if missing.
    """
    run_dir = Path(run_dir)
    features_path = run_dir / "scaler_features.json"
    if not features_path.exists():
        return None
    try:
        with open(features_path, 'r') as f:
            data = json.load(f)
        cols = data.get('feature_columns')
        return cols if isinstance(cols, list) else None
    except Exception:
        return None


def align_features_to_scaler(df, scaler, feature_columns):
    """Align df's columns to the scaler's training feature order.
    - Reorders to feature_columns
    - Drops extras
    - Adds missing columns filled with scaler means (standardizes to 0)
    - Fills NaNs per column with scaler means
    Returns a DataFrame with columns in feature_columns order.
    """
    if feature_columns is None or len(feature_columns) == 0:
        # Nothing to align to; return as-is
        return df

    # Build per-column means from the scaler
    # Assumes scaler.mean_ matches feature_columns order
    if not hasattr(scaler, 'mean_'):
        # Fallback: no means; just reorder and drop extras if possible
        aligned = pd.DataFrame({col: df[col] for col in feature_columns if col in df.columns})
        return aligned.reindex(columns=feature_columns)

    means = {col: scaler.mean_[i] for i, col in enumerate(feature_columns)}

    # Construct aligned DataFrame
    aligned = {}
    for col in feature_columns:
        if col in df.columns:
            series = df[col]
            series = series.fillna(means[col])
            aligned[col] = series
        else:
            # Missing column: fill with mean
            aligned[col] = pd.Series([means[col]] * len(df), index=df.index)

    aligned_df = pd.DataFrame(aligned, index=df.index)
    return aligned_df[feature_columns]


