"""
Pre-extract protein latents for multimodal fusion training
Run this in the 'multimodal' uv environment

This script:
1. Loads the trained protein model (MLP or Transformer)
2. Extracts latents for all samples in train.csv and test.csv
3. Saves latents as .npy files for later use in the brainiac environment
"""

import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path

# Add protein module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "protein"))

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


def extract_mlp_latents_batch(model, X_scaled, layer_name='hidden_layer_2'):
    """
    Extract latents from MLPClassifier for batch of samples
    
    Args:
        model: Trained MLPClassifier
        X_scaled: Scaled protein features [n_samples, n_features]
        layer_name: Which hidden layer to extract
    
    Returns:
        Latent features [n_samples, latent_dim]
    """
    if layer_name == 'hidden_layer_1':
        # First hidden layer
        layer_output = np.dot(X_scaled, model.coefs_[0]) + model.intercepts_[0]
        layer_output = np.maximum(0, layer_output)  # ReLU
    elif layer_name == 'hidden_layer_2':
        # Second hidden layer
        layer1_output = np.dot(X_scaled, model.coefs_[0]) + model.intercepts_[0]
        layer1_output = np.maximum(0, layer1_output)  # ReLU
        
        layer_output = np.dot(layer1_output, model.coefs_[1]) + model.intercepts_[1]
        layer_output = np.maximum(0, layer_output)  # ReLU
    else:
        raise ValueError(f"Unknown layer: {layer_name}")
    
    return layer_output.astype(np.float32)


def preextract_latents(
    train_csv,
    test_csv,
    protein_model_path,
    output_dir,
    layer_name='hidden_layer_2',
    protein_columns=None
):
    """
    Pre-extract protein latents for all samples
    
    Args:
        train_csv: Path to train.csv
        test_csv: Path to test.csv
        protein_model_path: Path to trained protein model (.pkl)
        output_dir: Directory to save extracted latents
        layer_name: Which layer to extract from
        protein_columns: List of protein column names (auto-detect if None)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("PRE-EXTRACTING PROTEIN LATENTS")
    print("="*60)
    print(f"Model: {protein_model_path}")
    print(f"Layer: {layer_name}")
    print(f"Output: {output_dir}")
    print()
    
    # Load trained model
    print("Loading protein model...")
    with open(protein_model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"  Model type: {type(model).__name__}")
    print(f"  Architecture: {model.hidden_layer_sizes}")
    
    # Load scaler if available
    scaler_path = Path(protein_model_path).parent.parent / "scaler.pkl"
    scaler = None
    if scaler_path.exists():
        print(f"Loading scaler from: {scaler_path}")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"  Scaler loaded successfully")
    else:
        print(f"  Warning: No scaler found at {scaler_path}")
        print(f"  Using raw protein values (no scaling)")
    print()
    
    # Process train and test sets
    for split_name, csv_path in [('train', train_csv), ('test', test_csv)]:
        print(f"Processing {split_name} set...")
        
        # Load CSV
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df)} samples")
        
        # Auto-detect protein columns
        if protein_columns is None:
            metadata_cols = ['RID', 'Subject', 'VISCODE', 'Visit', 'research_group', 
                           'Group', 'Sex', 'Age', 'subject_age', 'Image Data ID',
                           'Description', 'Type', 'Modality', 'Format', 'Acq Date',
                           'Downloaded', 'MRI_acquired', 'mri_source_path', 'mri_path']
            protein_cols = [col for col in df.columns if col not in metadata_cols]
        else:
            protein_cols = protein_columns
        
        print(f"  Protein features: {len(protein_cols)} columns")
        
        # Extract protein values
        X_raw = df[protein_cols].values.astype(np.float32)
        
        # Apply scaling if scaler is available
        if scaler is not None:
            print(f"  Applying StandardScaler...")
            X_scaled = scaler.transform(X_raw)
        else:
            print(f"  Using raw values (no scaling)")
            X_scaled = X_raw
        
        # Extract latents
        print(f"  Extracting latents from {layer_name}...")
        latents = extract_mlp_latents_batch(model, X_scaled, layer_name)
        print(f"  Latent shape: {latents.shape}")
        
        # Save latents
        latents_file = output_dir / f"{split_name}_protein_latents.npy"
        np.save(latents_file, latents)
        print(f"  Saved: {latents_file}")
        
        # Save subject IDs for reference
        subjects_file = output_dir / f"{split_name}_subjects.npy"
        np.save(subjects_file, df['Subject'].values)
        print(f"  Saved: {subjects_file}")
        
        # Save labels for reference
        labels = (df['research_group'] == 'AD').astype(int).values
        labels_file = output_dir / f"{split_name}_labels.npy"
        np.save(labels_file, labels)
        print(f"  Saved: {labels_file}")
        print()
    
    # Save metadata
    metadata = {
        'model_path': str(protein_model_path),
        'scaler_path': str(scaler_path) if scaler_path.exists() else None,
        'layer_name': layer_name,
        'latent_dim': latents.shape[1],
        'protein_columns': protein_cols,
        'scaling_applied': scaler is not None
    }
    
    import json
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {metadata_file}")
    
    print("\n" + "="*60)
    print("✅ PRE-EXTRACTION COMPLETE")
    print("="*60)
    print(f"Latent dimension: {latents.shape[1]}")
    print(f"Scaling applied: {scaler is not None}")
    print(f"Files saved in: {output_dir}")
    print("\nNext steps:")
    print("1. Use these pre-extracted latents in multimodal_dataset.py")
    print("2. Run training in 'multimodal' environment")


if __name__ == "__main__":
    # Configuration
    train_csv = "/home/ssim0068/data/multimodal-dataset/train.csv"
    test_csv = "/home/ssim0068/data/multimodal-dataset/test.csv"
    protein_model_path = "/home/ssim0068/multimodal-AD/src/protein/runs/run_20251003_133215/models/neural_network.pkl"
    output_dir = "/home/ssim0068/data/multimodal-dataset/protein_latents"
    layer_name = "hidden_layer_2"
    
    # Run extraction
    preextract_latents(
        train_csv=train_csv,
        test_csv=test_csv,
        protein_model_path=protein_model_path,
        output_dir=output_dir,
        layer_name=layer_name
    )


