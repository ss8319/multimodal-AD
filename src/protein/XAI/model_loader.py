"""
Centralized model loading for XAI modules

This module provides a generic, extensible way to load models for XAI analysis.
Uses a registry pattern to support different model types.
"""
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Callable, Any, Optional
import importlib

# Import project modules (relative imports from parent package)
# Fallback to absolute imports if running as script
try:
    from ..model import NeuralNetworkClassifier, NeuralNetwork
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from model import NeuralNetworkClassifier, NeuralNetwork


class ModelLoaderError(Exception):
    """Custom exception for model loading errors"""
    pass


class ModelRegistry:
    """
    Registry for model factory functions.
    
    Maps model names to factory functions that create model wrappers from configs.
    """
    def __init__(self):
        self._factories: Dict[str, Callable] = {}
        self._model_classes: Dict[str, str] = {}  # Maps model_name to model_class name
    
    def register(self, model_name: str, factory_fn: Callable, model_class: Optional[str] = None):
        """
        Register a model factory function.
        
        Args:
            model_name: Name of the model (e.g., 'neural_network')
            factory_fn: Function that takes (model_config, n_features) and returns a wrapper instance
            model_class: Optional class name to match against checkpoint's model_class
        """
        self._factories[model_name] = factory_fn
        if model_class:
            self._model_classes[model_name] = model_class
    
    def get_factory(self, model_name: str, model_class: Optional[str] = None) -> Callable:
        """
        Get factory function for a model.
        
        Args:
            model_name: Name of the model
            model_class: Optional model class name from checkpoint (for validation)
        
        Returns:
            Factory function
        
        Raises:
            ModelLoaderError: If model is not registered
        """
        if model_name not in self._factories:
            available = ', '.join(self._factories.keys())
            raise ModelLoaderError(
                f"Model '{model_name}' is not registered. "
                f"Available models: {available}"
            )
        
        # Optional: validate model_class matches expected
        if model_class and model_name in self._model_classes:
            expected_class = self._model_classes[model_name]
            if model_class != expected_class:
                raise ModelLoaderError(
                    f"Model class mismatch for '{model_name}': "
                    f"expected '{expected_class}', got '{model_class}'"
                )
        
        return self._factories[model_name]
    
    def list_models(self) -> list:
        """List all registered model names"""
        return list(self._factories.keys())


# Global registry instance
_registry = ModelRegistry()


def _create_neural_network_wrapper(model_config: dict, n_features: int):
    """
    Factory function to create NeuralNetworkClassifier from config.
    
    Args:
        model_config: Model configuration dict from checkpoint
        n_features: Number of input features
    
    Returns:
        NeuralNetworkClassifier instance with loaded model
    """
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
    
    # Create and load the neural network
    wrapper.model = NeuralNetwork(
        n_features=n_features,
        hidden_sizes=hidden_sizes,
        dropout=dropout
    )
    
    return wrapper


# Register models
_registry.register('neural_network', _create_neural_network_wrapper, 'NeuralNetwork')


def load_model(run_dir: str, model_name: str):
    """
    Load a saved model from the run directory.
    
    This is a generic loader that works with:
    - PyTorch models (saved as .pth files with model_config)
    - sklearn models (saved as .pkl files)
    
    Args:
        run_dir: Path to run directory containing models/
        model_name: Name of the model (e.g., 'neural_network', 'logistic_regression')
    
    Returns:
        Loaded model (wrapper for PyTorch models, sklearn model for sklearn models)
    
    Raises:
        ModelLoaderError: If model file not found or loading fails
        FileNotFoundError: If run_dir doesn't exist
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    models_dir = run_dir / "models"
    if not models_dir.exists():
        raise ModelLoaderError(f"Models directory not found: {models_dir}")
    
    # Try PyTorch model first (.pth file)
    pytorch_model_path = models_dir / f"{model_name}.pth"
    if pytorch_model_path.exists():
        try:
            checkpoint = torch.load(pytorch_model_path, map_location='cpu')
            
            if 'model_config' not in checkpoint:
                raise ModelLoaderError(
                    f"Invalid checkpoint format for {model_name}: missing 'model_config'. "
                    f"Checkpoint keys: {list(checkpoint.keys())}"
                )
            
            model_config = checkpoint['model_config']
            model_class = model_config.get('model_class')
            n_features = model_config.get('n_features')
            
            if n_features is None:
                raise ModelLoaderError(
                    f"Missing 'n_features' in model_config for {model_name}. "
                    f"Config keys: {list(model_config.keys())}"
                )
            
            # Get factory function from registry
            factory_fn = _registry.get_factory(model_name, model_class)
            
            # Create wrapper using factory
            wrapper = factory_fn(model_config, n_features)
            
            # Load state dict
            if 'model_state_dict' not in checkpoint:
                raise ModelLoaderError(
                    f"Invalid checkpoint format for {model_name}: missing 'model_state_dict'. "
                    f"Checkpoint keys: {list(checkpoint.keys())}"
                )
            
            wrapper.model.load_state_dict(checkpoint['model_state_dict'])
            wrapper.model.eval()
            
            return wrapper
            
        except ModelLoaderError:
            raise
        except Exception as e:
            raise ModelLoaderError(
                f"Failed to load PyTorch model '{model_name}' from {pytorch_model_path}: {e}"
            ) from e
    
    # Try sklearn model (.pkl file)
    sklearn_model_path = models_dir / f"{model_name}.pkl"
    if sklearn_model_path.exists():
        try:
            with open(sklearn_model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            raise ModelLoaderError(
                f"Failed to load sklearn model '{model_name}' from {sklearn_model_path}: {e}"
            ) from e
    
    # Model not found
    available_models = []
    if models_dir.exists():
        available_models = [
            f.stem for f in models_dir.iterdir() 
            if f.suffix in ['.pth', '.pkl']
        ]
    
    available_str = ', '.join(available_models) if available_models else 'none'
    raise ModelLoaderError(
        f"Model '{model_name}' not found in {models_dir}. "
        f"Available models: {available_str}"
    )


def list_available_models(run_dir: str) -> list:
    """
    List all available models in a run directory.
    
    Args:
        run_dir: Path to run directory
    
    Returns:
        List of model names (without extensions)
    """
    run_dir = Path(run_dir)
    models_dir = run_dir / "models"
    
    if not models_dir.exists():
        return []
    
    models = set()
    for f in models_dir.iterdir():
        if f.suffix in ['.pth', '.pkl']:
            models.add(f.stem)
    
    return sorted(list(models))

