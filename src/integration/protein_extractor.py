"""
Protein feature extractor for multimodal dataset
Extracts latents from trained protein models (MLPClassifier or ProteinTransformer)
"""

import torch
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# Import ProteinTransformer (will be loaded on demand)
try:
    from src.protein.model import ProteinTransformer
except ImportError:
    ProteinTransformer = None

# Feature alignment utilities
try:
    from src.protein.feature_utils import align_features_to_scaler, load_scaler_feature_columns
except ImportError:
    from protein.feature_utils import align_features_to_scaler, load_scaler_feature_columns

class ProteinLatentExtractor:
    """
    Extract latent representations from trained protein models
    Adapted from extract_latents.py for multimodal dataset
    """
    
    def __init__(self, run_dir, device='cpu'):
        """
        Args:
            run_dir: Path to protein model run directory
            device: Device for PyTorch models
        """
        self.run_dir = Path(run_dir)
        self.device = device
        
        # Load preprocessing scaler (if available)
        scaler_path = self.run_dir / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"  Loaded protein scaler from {scaler_path}")
            # Load saved training feature order for alignment
            self.scaler_feature_columns = load_scaler_feature_columns(self.run_dir)
            if self.scaler_feature_columns:
                print(f"  Loaded scaler feature columns ({len(self.scaler_feature_columns)})")
        else:
            print(f"  No scaler found - will use raw protein values")
            self.scaler = None
            self.scaler_feature_columns = None
        
        # Initialize models (will be loaded on demand)
        self.mlp_model = None
        self.transformer_model = None
        
    def load_mlp_model(self):
        """Load MLPClassifier model"""
        if self.mlp_model is not None:
            return self.mlp_model
            
        model_path = self.run_dir / "models" / "neural_network.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"MLP model not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            self.mlp_model = pickle.load(f)
        
        print(f"  Loaded MLP model from {model_path}")
        return self.mlp_model
    
    def _infer_model_config(self, state_dict):
        """
        DEPRECATED: We now require checkpoints to include 'model_config'.
        This method is kept only to preserve the interface; it will raise.
        """
        raise ValueError("Checkpoint missing 'model_config'. Please retrain/save models with model_config.")
    
    def load_transformer_model(self):
        """Load ProteinTransformer model"""
        if self.transformer_model is not None:
            return self.transformer_model
            
        model_path = self.run_dir / "models" / "protein_transformer.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Transformer model not found: {model_path}")
        
        # Check if ProteinTransformer is available
        if ProteinTransformer is None:
            raise ImportError("ProteinTransformer not available. Make sure src.protein.model is accessible.")
        
        # Load model state
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Require model_config for deterministic reconstruction
        if 'model_config' not in checkpoint:
            raise ValueError("Model checkpoint missing 'model_config'. Cannot load Transformer model.")
        config = checkpoint['model_config']
        
        self.transformer_model = ProteinTransformer(**config)
        self.transformer_model.load_state_dict(checkpoint['model_state_dict'])
        self.transformer_model.to(self.device)
        self.transformer_model.eval()
        
        print(f"  Loaded Transformer model from {model_path}")
        return self.transformer_model
    
    def preprocess_protein_data(self, protein_values, feature_names=None):
        """
        Preprocess protein data using the same scaler used during training
        
        Args:
            protein_values: Raw protein values [n_features]
            feature_names: Optional list of feature names for alignment
        
        Returns:
            Scaled protein values
        """
        if self.scaler is None:
            # No scaler available - return raw values
            print("  Warning: No scaler found, using raw protein values")
            return protein_values
        
        # If we have saved training feature columns and (ideally) names, align
        if self.scaler_feature_columns is not None and len(self.scaler_feature_columns) > 0:
            # Build a single-row DataFrame with provided names if available
            if feature_names is not None and len(feature_names) == len(protein_values):
                df = pd.DataFrame([protein_values], columns=list(feature_names))
            else:
                # Fallback: create DataFrame with current columns; if lengths mismatch, revert to direct transform
                try:
                    df = pd.DataFrame([protein_values], columns=self.scaler_feature_columns[:len(protein_values)])
                except Exception:
                    df = None
            if df is not None:
                aligned = align_features_to_scaler(df, self.scaler, self.scaler_feature_columns)
                scaled_values = self.scaler.transform(aligned)
                return scaled_values.flatten()
        
        # Final fallback: Reshape and transform without alignment
        protein_values_2d = protein_values.reshape(1, -1)
        scaled_values = self.scaler.transform(protein_values_2d)
        return scaled_values.flatten()
    
    def extract_mlp_latents(self, protein_values, layer_name='hidden_layer_2', feature_names=None):
        """
        Extract latents from MLPClassifier model
        
        Args:
            protein_values: Raw protein values [n_features]
            layer_name: Which hidden layer to extract ('hidden_layer_1' or 'hidden_layer_2')
        
        Returns:
            Latent features from specified layer
        """
        # Load model if not already loaded
        model = self.load_mlp_model()
        
        # Preprocess data
        X_scaled = self.preprocess_protein_data(protein_values, feature_names)
        
        # Extract features from hidden layers
        # MLPClassifier stores weights in coefs_ and intercepts_
        if layer_name == 'hidden_layer_1':
            # First hidden layer
            layer_output = np.dot(X_scaled, model.coefs_[0]) + model.intercepts_[0]
            # Apply activation function (ReLU for MLPClassifier)
            layer_output = np.maximum(0, layer_output)
        elif layer_name == 'hidden_layer_2':
            # Second hidden layer
            layer1_output = np.dot(X_scaled, model.coefs_[0]) + model.intercepts_[0]
            layer1_output = np.maximum(0, layer1_output)  # ReLU
            
            layer_output = np.dot(layer1_output, model.coefs_[1]) + model.intercepts_[1]
            layer_output = np.maximum(0, layer_output)  # ReLU
        else:
            raise ValueError(f"Unknown layer: {layer_name}")
        
        return layer_output.astype(np.float32)
    
    def extract_transformer_latents(self, protein_values, layer_name='transformer_embeddings', feature_names=None):
        """
        Extract latents from ProteinTransformer model
        
        Args:
            protein_values: Raw protein values [n_features]
            layer_name: Which layer to extract from
        
        Returns:
            Latent features from specified layer
        """
        # Load model if not already loaded
        model = self.load_transformer_model()
        
        # Preprocess data
        X_scaled = self.preprocess_protein_data(protein_values, feature_names)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0).to(self.device)  # [1, n_features]
        
        # Extract features using hooks
        activations = {}
        
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                activations[name] = output.detach().cpu().numpy().flatten()
            return hook
        
        # Register hooks based on layer_name
        if layer_name == 'input_projection':
            hook = model.input_projection.register_forward_hook(get_activation('input_projection'))
        elif layer_name == 'transformer_embeddings':
            hook = model.transformer.register_forward_hook(get_activation('transformer_embeddings'))
        elif layer_name == 'layernorm_output':
            hook = model.layer_norm.register_forward_hook(get_activation('layernorm_output'))
        else:
            raise ValueError(f"Unknown layer: {layer_name}")
        
        # Forward pass
        with torch.no_grad():
            _ = model(X_tensor)
        
        # Remove hook
        hook.remove()
        
        # Return the requested layer's activations
        if layer_name in activations:
            return activations[layer_name].astype(np.float32)
        else:
            raise RuntimeError(f"Failed to extract activations for {layer_name}")


if __name__ == "__main__":
    # Test the protein extractor
    print("Testing ProteinLatentExtractor...")
    
    # Paths
    run_dir = "/home/ssim0068/multimodal-AD/src/protein/runs/run_20251003_133215"
    
    # Create extractor
    extractor = ProteinLatentExtractor(run_dir, device='cpu')
    
    # Test with dummy protein data (320 features)
    dummy_protein = np.random.randn(320).astype(np.float32)
    
    print("\nTesting MLP latent extraction...")
    try:
        mlp_latents = extractor.extract_mlp_latents(dummy_protein, 'hidden_layer_2')
        print(f"  MLP hidden_layer_2 shape: {mlp_latents.shape}")
        print(f"  Sample values: {mlp_latents[:5]}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\nTesting Transformer latent extraction...")
    try:
        transformer_latents = extractor.extract_transformer_latents(dummy_protein, 'transformer_embeddings')
        print(f"  Transformer embeddings shape: {transformer_latents.shape}")
        print(f"  Sample values: {transformer_latents[:5]}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n✅ Protein extractor test completed!")
