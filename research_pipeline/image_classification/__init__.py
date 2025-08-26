"""
Image Classification Module for AD/CN Classification

This module provides a clean, academic-focused implementation of 3D MRI classification
using various deep learning architectures including CNNs, Vision Transformers, and BM-MAE.

Modules:
- models: CNN and transformer model architectures
- dataset: MRI dataset loading and preprocessing
- training: Training loops and loss functions
- evaluation: Metrics and visualization
- config: Configuration management
- main: Main training orchestration
"""

try:
    from .models import *
    from .dataset import MRIDataset
    from .training import train_model
    from .evaluation import evaluate_model, visualize_predictions
    from .config import get_config
except ImportError:
    # Fallback for direct module execution
    from models import *
    from dataset import MRIDataset
    from training import train_model
    from evaluation import evaluate_model, visualize_predictions
    from config import get_config

__version__ = "1.0.0"
__author__ = "Shamus"

__all__ = [
    'MRIDataset',
    'train_model', 
    'evaluate_model',
    'visualize_predictions',
    'get_config'
]
