"""
XAI (Explainable AI) module for protein classification models

This module contains implementations of interpretability methods:
- SHAP (SHapley Additive exPlanations)
- Integrated Gradients (IG)
"""

from .explain_shap import (
    explain_logistic_regression as explain_shap_lr,
    explain_neural_network as explain_shap_nn,
    load_model as load_model_shap,
    prepare_data_for_shap
)

from .explain_ig import (
    explain_logistic_regression as explain_ig_lr,
    explain_neural_network as explain_ig_nn,
    load_model as load_model_ig,
    prepare_data_for_ig
)

__all__ = [
    'explain_shap_lr',
    'explain_shap_nn',
    'load_model_shap',
    'prepare_data_for_shap',
    'explain_ig_lr',
    'explain_ig_nn',
    'load_model_ig',
    'prepare_data_for_ig',
]

