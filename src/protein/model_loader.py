"""
Centralized model loading utilities for PyTorch models.

This module provides generic model loading that works with any PyTorch model
architecture, as long as the model was saved with model_config using extract_model_config().

Usage:
    from src.protein.model_loader import load_pytorch_model_generic
    
    checkpoint = torch.load('model.pth')
    model = load_pytorch_model_generic(checkpoint, device='cpu')
"""

import torch
import importlib
from pathlib import Path


def load_pytorch_model_generic(checkpoint, device=None):
    """
    Generic loader for any PyTorch model using saved class info.
    
    This function dynamically reconstructs a PyTorch model from a checkpoint that
    contains both 'model_state_dict' and 'model_config'. The model_config must
    include 'model_class' and 'model_module' for dynamic import, plus all
    constructor parameters needed to recreate the model architecture.
    
    Args:
        checkpoint: Dictionary with 'model_config' and 'model_state_dict'
                   Can also be a path to a checkpoint file (str or Path)
        device: torch.device to load model onto (default: 'cpu')
        
    Returns:
        Loaded PyTorch model in eval mode, moved to specified device
        
    Raises:
        ValueError: If checkpoint is missing required model_config fields
        ImportError: If model class cannot be imported
        TypeError: If model cannot be created with provided parameters
        RuntimeError: If state_dict cannot be loaded into model
        
    Example:
        >>> checkpoint = torch.load('models/custom_transformer.pth')
        >>> model = load_pytorch_model_generic(checkpoint, device='cuda')
        >>> # Model is ready for inference
        >>> model.eval()
    """
    if device is None:
        device = torch.device('cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    
    # Handle file path input
    if isinstance(checkpoint, (str, Path)):
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Validate checkpoint structure
    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Checkpoint must be a dictionary, got {type(checkpoint)}. "
            f"If loading from file, ensure it contains 'model_config' and 'model_state_dict'."
        )
    
    if 'model_config' not in checkpoint:
        raise ValueError(
            "Model checkpoint missing 'model_config'. Cannot load model.\n"
            "This checkpoint may have been saved with an older version that didn't include config.\n"
            "Please retrain the model with the updated code that saves model_config."
        )
    
    model_config = checkpoint['model_config']
    
    # Get class name and module from config (required for dynamic loading)
    model_class_name = model_config.get('model_class')
    model_module_path = model_config.get('model_module')
    
    if not model_class_name or not model_module_path:
        raise ValueError(
            "Model config missing 'model_class' or 'model_module'. "
            "This checkpoint may have been saved with an older version. "
            "Please retrain the model."
        )
    
    # Dynamically import the model class
    try:
        # Import the module
        module = importlib.import_module(model_module_path)
        # Get the class
        model_class = getattr(module, model_class_name)
    except ImportError as e:
        raise ImportError(
            f"Failed to import module '{model_module_path}': {e}\n"
            f"Make sure the model definition module is available in the Python path."
        )
    except AttributeError as e:
        raise ImportError(
            f"Failed to find class '{model_class_name}' in module '{model_module_path}': {e}\n"
            f"Make sure the model class is defined in the module."
        )
    
    # Extract constructor parameters (exclude metadata keys)
    metadata_keys = {'model_class', 'model_module'}
    constructor_kwargs = {
        k: v for k, v in model_config.items() 
        if k not in metadata_keys
    }
    
    # Validate required parameters
    if 'n_features' not in constructor_kwargs:
        raise ValueError(
            f"Model config missing 'n_features'. Cannot reconstruct {model_class_name}.\n"
            f"Available config keys: {list(constructor_kwargs.keys())}"
        )
    
    # Recreate model instance with exact architecture parameters
    try:
        model = model_class(**constructor_kwargs)
    except TypeError as e:
        raise TypeError(
            f"Failed to create {model_class_name} with parameters {constructor_kwargs}: {e}\n"
            f"Model may have changed signature. Check model definition.\n"
            f"Expected parameters: {list(constructor_kwargs.keys())}"
        )
    
    # Load state dict (weights)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load state_dict into {model_class_name}: {e}\n"
            f"Model architecture may have changed since training.\n"
            f"Check that the model_config matches the current model definition."
        )
    
    # Move to device and set eval mode
    model.to(device)
    model.eval()
    
    # Print loaded info for debugging
    print(f"Loaded pre-trained model: {model_class_name}")
    print(f"  Module: {model_module_path}")
    print(f"  Architecture parameters:")
    for k, v in constructor_kwargs.items():
        print(f"    {k}: {v}")
    
    return model


def get_device():
    """
    Get the appropriate device for PyTorch operations.
    
    Returns:
        torch.device: CUDA device if available, else CPU
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"  Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print(f"  Using CPU")
    return device

