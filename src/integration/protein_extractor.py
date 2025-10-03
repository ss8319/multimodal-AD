"""
Protein feature extractor for multimodal dataset
Extracts latents from trained protein models (MLPClassifier or ProteinTransformer)
"""

import torch
import numpy as np
import pickle
from pathlib import Path
import sys

# Add protein module path
sys.path.append(str(Path(__file__).parent.parent / "protein"))

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


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
        else:
            print(f"  No scaler found - will use raw protein values")
            self.scaler = None
        
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
    
    def load_transformer_model(self):
        """Load ProteinTransformer model"""
        if self.transformer_model is not None:
            return self.transformer_model
            
        model_path = self.run_dir / "models" / "protein_transformer.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Transformer model not found: {model_path}")
        
        # Import here to avoid circular imports
        from model import ProteinTransformer
        
        # Load model state
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get n_features from checkpoint or config
        if 'config' in checkpoint:
            n_features = checkpoint['config']['n_features']
        else:
            # Try to infer from model state dict
            # Look for input layer weights to determine input size
            for key, value in checkpoint['model_state_dict'].items():
                if 'input_projection.weight' in key:
                    n_features = value.shape[1]
                    break
            else:
                raise ValueError("Could not determine n_features from checkpoint")
        
        self.transformer_model = ProteinTransformer(n_features=n_features)
        self.transformer_model.load_state_dict(checkpoint['model_state_dict'])
        self.transformer_model.to(self.device)
        self.transformer_model.eval()
        
        print(f"  Loaded Transformer model from {model_path}")
        return self.transformer_model
    
    def preprocess_protein_data(self, protein_values):
        """
        Preprocess protein data using the same scaler used during training
        
        Args:
            protein_values: Raw protein values [n_features]
        
        Returns:
            Scaled protein values
        """
        if self.scaler is None:
            # No scaler available - return raw values
            return protein_values
        
        # Reshape for scaler (expects 2D array)
        protein_values_2d = protein_values.reshape(1, -1)
        scaled_values = self.scaler.transform(protein_values_2d)
        return scaled_values.flatten()
    
    def extract_mlp_latents(self, protein_values, layer_name='hidden_layer_2'):
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
        X_scaled = self.preprocess_protein_data(protein_values)
        
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
    
    def extract_transformer_latents(self, protein_values, layer_name='transformer_embeddings'):
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
        X_scaled = self.preprocess_protein_data(protein_values)
        
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
