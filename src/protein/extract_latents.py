"""
Extract latent representations from trained protein classification models
"""
import pandas as pd
import numpy as np
import pickle
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# Import our modules
from dataset import ProteinDataLoader
from model import ProteinTransformer


class ProteinLatentExtractor:
    """Extract latent representations from trained protein models"""
    
    def __init__(self, run_dir, train_csv_path, random_state=42):
        """
        Initialize latent extractor
        
        Args:
            run_dir: Path to saved model run directory
            train_csv_path: Path to training CSV (needed for scaler fitting)
            random_state: Random seed
        """
        self.run_dir = Path(run_dir)
        self.models_dir = self.run_dir / "models"
        self.train_csv_path = Path(train_csv_path)
        self.random_state = random_state
        
        # Load and preprocess training data (needed for scaler)
        self.data_loader = ProteinDataLoader(
            data_path=train_csv_path,
            random_state=random_state
        )
        
        print(f"Loading training data for scaler fitting...")
        train_df = self.data_loader.load_data()
        self.X_train_raw, self.y_train = self.data_loader.prepare_features(train_df, fit=True)
        
        # Fit scaler on training data (same as in save_all_models)
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train_raw)
        
        print(f"   Training data: {self.X_train_scaled.shape}")
        print(f"   Available models: {list(self.models_dir.glob('*.pkl'))} + {list(self.models_dir.glob('*.pth'))}")
    
    def load_sklearn_model(self, model_name):
        """Load sklearn model from pickle file"""
        model_path = self.models_dir / f"{model_name}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Loaded {model_name}: {type(model).__name__}")
        return model
    
    def load_pytorch_model(self, model_name="protein_transformer"):
        """Load PyTorch model from state dict"""
        model_path = self.models_dir / f"{model_name}.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Recreate model with the same configuration
        model_config = checkpoint['model_config']
        model = ProteinTransformer(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Loaded {model_name}: ProteinTransformer with config {model_config}")
        return model, model_config
    
    def preprocess_data(self, data_path):
        """
        Preprocess new data using the same pipeline as training
        
        Args:
            data_path: Path to CSV file
            
        Returns:
            X_scaled: Scaled features ready for model input
            df: Original dataframe with metadata
        """
        # Load and preprocess data
        self.data_loader.data_path = Path(data_path)
        df = self.data_loader.load_data() #reuse function from ProteinDataLoader
        X_raw, y = self.data_loader.prepare_features(df, fit=False)  # use function from ProteinDataLoader
        
        # Scale using training scaler
        X_scaled = self.scaler.transform(X_raw)
        
        print(f"Preprocessed data: {X_scaled.shape}")
        return X_scaled, df
       
    def extract_neural_network_latents(self, X_scaled, model_name="neural_network"):
        """
        Extract hidden layer activations from Neural Network by manually
        computing the forward pass using the model's weights and biases.
        This ensures we get exactly the hidden layer activations we expect.
        """
        model = self.load_sklearn_model(model_name)
        
        if not isinstance(model, MLPClassifier):
            raise TypeError(f"Expected MLPClassifier, got {type(model).__name__}")
        
        latents = {}
        
        # Get model parameters
        print(f"Model architecture: {model.hidden_layer_sizes}")
        
        # Manual forward pass to extract activations
        activation = X_scaled
        
        # Process each hidden layer
        for i, (coef, intercept) in enumerate(zip(model.coefs_[:-1], model.intercepts_[:-1])):
            # Linear transformation
            activation = np.dot(activation, coef) + intercept
            
            # Apply activation function (ReLU for MLPClassifier)
            activation = np.maximum(0, activation)  # ReLU
            
            # Store the activation
            latents[f'hidden_layer_{i+1}'] = activation.copy()
            print(f"   Hidden layer {i+1}: {activation.shape}")
        
        # Final layer (before output activation)
        final_coef, final_intercept = model.coefs_[-1], model.intercepts_[-1]
        final_activation = np.dot(activation, final_coef) + final_intercept
        latents['pre_output'] = final_activation
        print(f"   Pre-output layer: {final_activation.shape}")
        
        return latents
    
    def extract_transformer_latents(self, X_scaled, model_name="protein_transformer"):
        """
        Extract embeddings from Protein Transformer using hooks to capture
        intermediate activations at specific points in the network.
        """
        model, config = self.load_pytorch_model(model_name)
        
        latents = {}
        
        # Create hooks to capture intermediate activations
        activations = {}
        
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach().cpu()
            return hook
        
        # Register hooks at key points in the network
        hooks = [
            model.input_projection.register_forward_hook(get_activation('input_projection')),
            model.transformer.register_forward_hook(get_activation('transformer_output')),
            model.classifier[0].register_forward_hook(get_activation('layernorm_output')),
            model.classifier[2].register_forward_hook(get_activation('first_linear_output')),
            model.classifier[3].register_forward_hook(get_activation('relu_output'))
        ]
        
        # Forward pass
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled)
            _ = model(X_tensor)  # Run forward pass
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Process and store activations
        latents['input_projection'] = activations['input_projection'].numpy()
        print(f"   Input projection: {latents['input_projection'].shape}")
        
        # Process transformer output (squeeze out the sequence dimension which is 1)
        transformer_out = activations['transformer_output'].squeeze(1).numpy()
        latents['transformer_embeddings'] = transformer_out
        print(f"   Transformer embeddings: {transformer_out.shape}")
        
        # Process other activations
        latents['layernorm_output'] = activations['layernorm_output'].numpy()
        print(f"   LayerNorm output: {latents['layernorm_output'].shape}")
        
        latents['first_linear_output'] = activations['first_linear_output'].numpy()
        print(f"   First linear output: {latents['first_linear_output'].shape}")
        
        latents['relu_output'] = activations['relu_output'].numpy()
        print(f"   ReLU output: {latents['relu_output'].shape}")
        
        return latents
     
    def save_latents(self, latents, output_path, metadata=None):
        """
        Save latents to files
        
        Args:
            latents: Dictionary of latent representations
            output_path: Base path for saving (will create subdirectory)
            metadata: Optional metadata to save
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each latent type
        for latent_name, latent_data in latents.items():
            if isinstance(latent_data, np.ndarray):
                # Save as numpy array
                np.save(output_dir / f"{latent_name}.npy", latent_data)
                print(f"Saved {latent_name}: {latent_data.shape} -> {output_dir / f'{latent_name}.npy'}")
        
        # Save metadata
        if metadata:
            import json
            with open(output_dir / "latents_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Saved metadata -> {output_dir / 'latents_metadata.json'}")
        
        # Save subject IDs if available
        if hasattr(self, 'current_subjects') and self.current_subjects is not None:
            np.save(output_dir / "subject_ids.npy", self.current_subjects)
            print(f"Saved subject IDs -> {output_dir / 'subject_ids.npy'}")
        
        # Save labels if available
        if hasattr(self, 'current_labels') and self.current_labels is not None:
            np.save(output_dir / "labels.npy", self.current_labels)
            print(f"Saved labels -> {output_dir / 'labels.npy'}")


def main():
    """Example usage"""
    # Configuration
    run_dir = "src/protein/runs/run_20251002_170038"  # Update with your run
    train_csv = "src/data/protein/proteomic_encoder_train.csv"
    test_csv = "src/data/protein/proteomic_encoder_test.csv"
    
    # Initialize extractor
    extractor = ProteinLatentExtractor(run_dir, train_csv)
    
    # Preprocess test data
    X_test_scaled, test_df = extractor.preprocess_data(test_csv)
    
    # Store subject IDs and labels for reference
    if 'RID' in test_df.columns:
        extractor.current_subjects = test_df['RID'].values
    
    if 'research_group' in test_df.columns:
        extractor.current_labels = test_df['research_group'].values
      
    # 1. Neural Network Latents
    print(f"\n1. NEURAL NETWORK LATENTS")
    print("-" * 40)
    try:
        nn_latents = extractor.extract_neural_network_latents(X_test_scaled)
        output_dir = Path(run_dir) / "latents" / "neural_network"
        extractor.save_latents(nn_latents, output_dir, {
            'model_type': 'MLPClassifier',
            'recommended_latent': 'hidden_layer_2'
        })
    except Exception as e:
        print(f"   Error extracting NN latents: {e}")
    
    # 2. Transformer Latents
    print(f"\n2. PROTEIN TRANSFORMER LATENTS")
    print("-" * 40)
    try:
        transformer_latents = extractor.extract_transformer_latents(X_test_scaled)
        output_dir = Path(run_dir) / "latents" / "protein_transformer"
        extractor.save_latents(transformer_latents, output_dir, {
            'model_type': 'ProteinTransformer',
            'recommended_latent': 'transformer_embeddings'
        })
    except Exception as e:
        print(f"   Error extracting Transformer latents: {e}")

if __name__ == "__main__":
    main()