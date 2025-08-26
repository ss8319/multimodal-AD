"""
Simple configuration management for image classification experiments.

This module provides YAML-based configuration loading with sensible defaults.
"""

import yaml
import os
from pathlib import Path
from typing import Optional

def get_default_config() -> dict:
    """Get default configuration as a simple dictionary."""
    return {
        'experiment_name': 'mri_classification_baseline',
        'output_dir': r'D:\ADNI\AD_CN\proteomics\Biomarkers Consortium Plasma Proteomics MRM\MRI\splits',
        'save_model': True,
        'save_results': True,
        'enable_visualization': False,
        'enable_quality_control': False,
        'qc_sample_count': 3,
        'training': {
            'methods': ['resnet3d'],
            'epochs': 20,
            'learning_rate': 0.001,
            'batch_size': 16,
            'weight_decay_classifier_head': 1e-4,
            'weight_decay_encoder': 0.0,
            'target_size': [128, 128, 128],
            'train_val_split': 0.8,
            'early_stopping_patience': 8,
            'min_epochs': 5,
            'gradient_clip_norm': 1.0,
            'enable_augmentation': True,
            'rotation_range': 5.0,
            'intensity_scale_range': [0.8, 1.2],
            'noise_std': 0.05
        },
        'model': {
            'bmmae_pretrained_path': 'BM-MAE/pretrained_models/bmmae.pth',
            'cnn_dropout': 0.5,
            'vit_patch_size': 8,
            'vit_embed_dim': 768
        },
        'data': {
            'adni_base_path': r'D:\ADNI\AD_CN\proteomics\Biomarkers Consortium Plasma Proteomics MRM\MRI\ADNI',
            'splits_folder': r'D:\ADNI\AD_CN\proteomics\Biomarkers Consortium Plasma Proteomics MRM\MRI\splits',
            'normalization_method': 'zscore',
            'percentile_range': [1, 99],
            'max_files_per_subject': None
        }
    }

def load_config_from_yaml(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def get_config(config_path: Optional[str] = None) -> dict:
    """Get configuration from file or use defaults."""
    if config_path is None:
        return get_default_config()
    
    try:
        return load_config_from_yaml(config_path)
    except Exception as e:
        print(f"⚠️ Failed to load config from {config_path}: {e}")
        print("📋 Using default configuration instead")
        return get_default_config()

def create_example_config(config_path: str = "example_config.yaml"):
    """Create an example configuration file."""
    config = get_default_config()
    
    # Modify some defaults for the example
    config['experiment_name'] = "example_mri_experiment"
    config['training']['methods'] = ['resnet3d', 'bmmae_frozen']
    config['training']['epochs'] = 10
    config['training']['batch_size'] = 8
    config['enable_visualization'] = True
    
    # Save to YAML
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"📋 Example configuration saved to: {config_path}")
    return config_path
