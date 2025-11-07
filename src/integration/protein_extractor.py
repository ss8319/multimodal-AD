"""
Protein feature extractor for multimodal dataset
Extracts latents from trained protein models (any PyTorch model architecture)

Uses centralized model loading for model-agnostic support.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

from sklearn.preprocessing import StandardScaler

# Import centralized model loader
try:
    from src.protein.model_loader import load_pytorch_model_generic
except ImportError:
    # Fallback for relative imports
    sys.path.append(str(Path(__file__).parent.parent / "protein"))
    from model_loader import load_pytorch_model_generic

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
        # Store models by their class name for generic access
        self.loaded_models = {}  # {model_class_name: model_instance}
        
    def _load_model_generic(self, model_name):
        """
        Generic model loader that works with any PyTorch model architecture.
        
        Args:
            model_name: Name of model file (e.g., 'neural_network.pth', 'custom_transformer.pth')
            
        Returns:
            Loaded PyTorch model in eval mode
        """
        model_path = self.run_dir / "models" / model_name
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path} (cwd={Path.cwd()})"
            )
        
        # Use centralized generic loader (accepts file path directly)
        model = load_pytorch_model_generic(model_path, device=self.device)
        
        # Store model by class name for later reference
        model_class_name = model.__class__.__name__
        self.loaded_models[model_class_name] = model
        
        return model
    
    def load_nn_model(self):
        """
        Load PyTorch NeuralNetwork model (backward compatibility wrapper).
        
        Uses generic loader internally.
        """
        model_name = "neural_network.pth"
        model = self._load_model_generic(model_name)
        return model
    
    def load_transformer_model(self):
        """
        Load ProteinTransformer model (backward compatibility wrapper).
        
        Uses generic loader internally.
        """
        model_name = "protein_transformer.pth"
        model = self._load_model_generic(model_name)
        return model
    
    def load_custom_transformer_model(self):
        """
        Load CustomTransformerEncoder model.
        
        Uses generic loader internally.
        """
        model_name = "custom_transformer.pth"
        model = self._load_model_generic(model_name)
        return model
    
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
    
    def extract_custom_transformer_latents(self, protein_values, layer_name='attention_output', feature_names=None):
        """
        Extract latents from CustomTransformerEncoder model
        
        Args:
            protein_values: Raw protein values [n_features]
            layer_name: Which layer to extract from
                       Options:
                       - 'conv_output': After convolutional layer (n_features × d_model)
                       - 'attention_output': After attention blocks (n_features × d_model)
                       - 'classifier_input': Before classifier (flattened, n_features * d_model)
        
        Returns:
            Latent features from specified layer
        """
        # Load model if not already loaded
        model = self.load_custom_transformer_model()
        
        # Preprocess data
        X_scaled = self.preprocess_protein_data(protein_values, feature_names)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0).to(self.device)  # [1, n_features]
        
        # Extract features using hooks
        activations = {}
        
        def get_activation(name):
            def hook(module, input, output):
                # Handle different output shapes
                if isinstance(output, tuple):
                    output = output[0]
                # Flatten for consistent output format
                output_np = output.detach().cpu().numpy()
                if output_np.ndim > 1:
                    activations[name] = output_np.flatten()
                else:
                    activations[name] = output_np
            return hook
        
        # Register hooks based on layer_name
        if layer_name == 'conv_output':
            # After conv layer: (batch, d_model, n_features) -> transpose -> (batch, n_features, d_model)
            hook = model.conv_layer.register_forward_hook(get_activation('conv_output'))
        elif layer_name == 'attention_output':
            # After attention blocks: (batch, n_features, d_model)
            # Hook on the last attention block
            if len(model.attention_blocks) > 0:
                hook = model.attention_blocks[-1].register_forward_hook(get_activation('attention_output'))
            else:
                raise ValueError("Model has no attention blocks")
        elif layer_name == 'classifier_input':
            # Before classifier: need to flatten (batch, n_features, d_model) -> (batch, n_features * d_model)
            # Hook on classifier's Flatten layer
            hook = model.classifier[0].register_forward_hook(get_activation('classifier_input'))
        else:
            raise ValueError(
                f"Unknown layer: {layer_name}. "
                f"Options: 'conv_output', 'attention_output', 'classifier_input'"
            )
        
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
    
    def extract_latents(self, protein_values, model_name='neural_network.pth', 
                        layer_name='auto', feature_names=None):
        """
        Generic latent extraction that automatically routes to model-specific methods.
        
        Args:
            protein_values: Raw protein values [n_features]
            model_name: Name of model file (e.g., 'neural_network.pth', 'custom_transformer.pth')
            layer_name: Which layer to extract from (model-specific)
            feature_names: Optional feature names for alignment
        
        Returns:
            Latent features from specified layer
        """
        # Load model to get its class name
        model = self._load_model_generic(model_name)
        model_class_name = model.__class__.__name__
        
        # Route to appropriate extraction method based on model class
        if model_class_name == 'NeuralNetwork':
            if layer_name == 'auto':
                layer_name = 'last_hidden_layer'
            return self.extract_nn_latents(protein_values, layer_name, feature_names)
        elif model_class_name == 'ProteinTransformer':
            if layer_name == 'auto':
                layer_name = 'transformer_embeddings'
            return self.extract_transformer_latents(protein_values, layer_name, feature_names)
        elif model_class_name == 'CustomTransformerEncoder':
            if layer_name == 'auto':
                layer_name = 'attention_output'
            return self.extract_custom_transformer_latents(protein_values, layer_name, feature_names)
        else:
            raise ValueError(
                f"Unknown model class: {model_class_name}. "
                f"Supported: NeuralNetwork, ProteinTransformer, CustomTransformerEncoder"
            )


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
