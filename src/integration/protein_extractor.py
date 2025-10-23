"""
Protein feature extractor for multimodal dataset
Extracts latents from trained protein models (PyTorch NeuralNetwork or ProteinTransformer)
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

from sklearn.preprocessing import StandardScaler

# Import PyTorch models (will be loaded on demand)
try:
    from src.protein.model import NeuralNetwork, ProteinTransformer
except ImportError:
    NeuralNetwork = None
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
        # Handle relative paths that might be prefixed with project name
        if isinstance(run_dir, str) and run_dir.startswith('multimodal-AD/'):
            # If path starts with project name but we're already in project dir
            cwd = Path.cwd()
            if cwd.name == 'multimodal-AD' or str(cwd).endswith('/multimodal-AD'):
                # Strip the project prefix to avoid duplication
                run_dir = run_dir.replace('multimodal-AD/', '', 1)
                print(f"  Removed duplicate project prefix from path")
        
        # Normalize to absolute path to avoid CWD-dependent resolution in SLURM/jobs
        self.run_dir = Path(run_dir).expanduser().resolve()
        print(f"  ProteinLatentExtractor using run_dir: {self.run_dir}")
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
        self.nn_model = None
        self.transformer_model = None
        
    def load_nn_model(self):
        """Load PyTorch NeuralNetwork model"""
        if self.nn_model is not None:
            return self.nn_model
            
        model_path = self.run_dir / "models" / "neural_network.pth"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Neural Network model not found: {model_path} (cwd={Path.cwd()})"
            )
        
        # Check if NeuralNetwork is available
        if NeuralNetwork is None:
            raise ImportError("NeuralNetwork not available. Make sure src.protein.model is accessible.")
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Require model_config for deterministic reconstruction
        if 'model_config' not in checkpoint:
            raise ValueError("Model checkpoint missing 'model_config'. Cannot load Neural Network model.")
        config = checkpoint['model_config']
        
        # Extract architecture parameters
        n_features = config.get('n_features')
        hidden_sizes = config.get('hidden_sizes', (128, 64))
        dropout = config.get('dropout', 0.2)
        
        if n_features is None:
            raise ValueError(
                "model_config missing 'n_features'. Cannot reconstruct Neural Network.\n"
                "Please retrain the protein model with the updated code to include n_features in the checkpoint."
            )
        
        # Reconstruct model
        self.nn_model = NeuralNetwork(
            n_features=n_features,
            hidden_sizes=hidden_sizes,
            dropout=dropout
        )
        self.nn_model.load_state_dict(checkpoint['model_state_dict'])
        self.nn_model.to(self.device)
        self.nn_model.eval()
        
        print(f"  Loaded Neural Network model from {model_path}")
        print(f"    Architecture: n_features={n_features}, hidden_sizes={hidden_sizes}, dropout={dropout}")
        return self.nn_model
    
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
                # Step 1: Align features to match training data order/columns
                aligned = align_features_to_scaler(df, self.scaler, self.scaler_feature_columns)
                
                # Step 2: Validate that alignment worked correctly
                # Check 1: Did we get the right number of features?
                expected_n_features = len(self.scaler_feature_columns)
                actual_n_features = aligned.shape[1]
                if actual_n_features != expected_n_features:
                    raise ValueError(
                        f"Feature alignment failed: expected {expected_n_features} features, "
                        f"got {actual_n_features}. Scaler columns: {self.scaler_feature_columns[:5]}..."
                    )
                
                # Check 2: Are the columns in the correct order?
                # This prevents subtle bugs where features get reordered
                if not all(aligned.columns == self.scaler_feature_columns):
                    raise ValueError(
                        f"Feature alignment failed: column order mismatch. "
                        f"Expected: {self.scaler_feature_columns[:5]}..., "
                        f"Got: {list(aligned.columns)[:5]}..."
                    )
                
                # Step 3: Apply scaling (now safe to do)
                scaled_values = self.scaler.transform(aligned)
                return scaled_values.flatten()
        
        # Final fallback: Direct scaling without alignment
        # This happens when we don't have feature column names or alignment fails
        protein_values_2d = protein_values.reshape(1, -1)
        
        # Safety check: Make sure dimensions match what the scaler expects
        # The scaler was trained on n_features_in_ features, so we need exactly that many
        expected_n_features = self.scaler.n_features_in_
        actual_n_features = protein_values_2d.shape[1]
        if actual_n_features != expected_n_features:
            raise ValueError(
                f"Feature dimension mismatch: scaler expects {expected_n_features} features, "
                f"got {actual_n_features}. Cannot proceed without feature alignment."
            )
        
        # Apply scaling directly (assumes features are already in correct order)
        scaled_values = self.scaler.transform(protein_values_2d)
        return scaled_values.flatten()
    
    def extract_nn_latents(self, protein_values, layer_name='last_hidden_layer', feature_names=None):
        """
        Extract latents from PyTorch NeuralNetwork model
        
        Args:
            protein_values: Raw protein values [n_features]
            layer_name: Which hidden layer to extract from
                       Options: 'hidden_layer_1', 'hidden_layer_2', 'last_hidden_layer' (default)
                       'last_hidden_layer' extracts from the final hidden layer before classification
        
        Returns:
            Latent features from specified layer
        """
        # Load model if not already loaded
        model = self.load_nn_model()
        
        # Preprocess data
        X_scaled = self.preprocess_protein_data(protein_values, feature_names)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0).to(self.device)  # [1, n_features]
        
        # The NeuralNetwork architecture is:
        # Sequential(
        #     Linear(n_features, hidden_sizes[0]),  # Layer 0
        #     ReLU,                                  # Layer 1
        #     Dropout,                              # Layer 2
        #     Linear(hidden_sizes[0], hidden_sizes[1]),  # Layer 3
        #     ReLU,                                  # Layer 4
        #     Dropout,                              # Layer 5
        #     ... (repeat for more hidden layers)
        #     Linear(hidden_sizes[-1], 2)           # Final classification layer
        # )
        
        # Map layer names to indices
        # For hidden_sizes=(128, 64):
        # - hidden_layer_1: after layer 2 (first ReLU+Dropout) → output (128,)
        # - hidden_layer_2: after layer 5 (second ReLU+Dropout) → output (64,)
        # - last_hidden_layer: same as the last hidden layer (layer 5 for 2 hidden layers)
        
        # Calculate layer indices based on architecture
        n_hidden_layers = len(model.net) // 3 - 1  # Each hidden layer is 3 modules (Linear, ReLU, Dropout)
        
        if layer_name == 'last_hidden_layer':
            # Extract from last hidden layer (before classification head)
            target_layer_idx = len(model.net) - 2  # Second to last (before classification Linear)
        elif layer_name.startswith('hidden_layer_'):
            # Extract layer number from name
            try:
                layer_num = int(layer_name.split('_')[-1])
                if layer_num < 1 or layer_num > n_hidden_layers:
                    raise ValueError(f"Invalid layer number {layer_num}. Model has {n_hidden_layers} hidden layers.")
                # Each hidden layer group is 3 modules (Linear, ReLU, Dropout)
                # Layer 1 ends at index 2, Layer 2 ends at index 5, etc.
                target_layer_idx = (layer_num * 3) - 1
            except (IndexError, ValueError) as e:
                raise ValueError(f"Invalid layer name format: {layer_name}. Use 'hidden_layer_N' where N is 1-{n_hidden_layers}")
        else:
            raise ValueError(f"Unknown layer: {layer_name}. Use 'hidden_layer_1', 'hidden_layer_2', or 'last_hidden_layer'")
        
        # Extract features using forward hook
        activations = {}
        
        def get_activation(name):
            def hook(module, input, output):
                activations[name] = output.detach().cpu().numpy().flatten()
            return hook
        
        # Register hook at target layer
        hook_handle = model.net[target_layer_idx].register_forward_hook(get_activation('target_layer'))
        
        # Forward pass
        with torch.no_grad():
            _ = model(X_tensor)
        
        # Remove hook
        hook_handle.remove()
        
        # Return the extracted activations
        if 'target_layer' in activations:
            return activations['target_layer'].astype(np.float32)
        else:
            raise RuntimeError(f"Failed to extract activations for {layer_name} at index {target_layer_idx}")
    
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
    
    # Paths - update this to your actual run directory
    run_dir = "/home/ssim0068/multimodal-AD/src/protein/runs/run_20251016_194651"
    
    # Create extractor
    extractor = ProteinLatentExtractor(run_dir, device='cpu')
    
    # Test with dummy protein data (320 features)
    dummy_protein = np.random.randn(320).astype(np.float32)
    
    print("\nTesting Neural Network latent extraction...")
    try:
        # Test different layer names
        for layer_name in ['hidden_layer_1', 'hidden_layer_2', 'last_hidden_layer']:
            nn_latents = extractor.extract_nn_latents(dummy_protein, layer_name)
            print(f"  NN {layer_name} shape: {nn_latents.shape}")
            print(f"  Sample values: {nn_latents[:5]}")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTesting Transformer latent extraction...")
    try:
        transformer_latents = extractor.extract_transformer_latents(dummy_protein, 'transformer_embeddings')
        print(f"  Transformer embeddings shape: {transformer_latents.shape}")
        print(f"  Sample values: {transformer_latents[:5]}")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Protein extractor test completed!")
