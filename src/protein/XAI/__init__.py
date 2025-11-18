"""
XAI (Explainable AI) module for protein classification models

This module contains implementations of interpretability methods:
- SHAP (SHapley Additive exPlanations)
- Integrated Gradients (IG)

The module uses a centralized model loader and shared data preparation utilities
for consistency and maintainability.
"""

# Import shared utilities
from .model_loader import load_model, list_available_models, ModelLoaderError
from .data_utils import prepare_data, DataPreparationError

# Import explainer functions
from .explain_shap import (
    explain_logistic_regression as explain_shap_lr,
    explain_neural_network as explain_shap_nn,
    prepare_data_for_shap  # Alias for backward compatibility
)

from .explain_ig import (
    explain_logistic_regression as explain_ig_lr,
    explain_neural_network as explain_ig_nn,
    prepare_data_for_ig  # Alias for backward compatibility
)

__all__ = [
    # Shared utilities
    'load_model',
    'list_available_models',
    'prepare_data',
    'ModelLoaderError',
    'DataPreparationError',
    # SHAP explainers
    'explain_shap_lr',
    'explain_shap_nn',
    'prepare_data_for_shap',
    # IG explainers
    'explain_ig_lr',
    'explain_ig_nn',
    'prepare_data_for_ig',
]

