#!/usr/bin/env python3
"""
Main pipeline controller for protein autoencoder and TabNet training and visualization.
Uses YAML configuration files for easy parameter management.
"""

import argparse
import yaml
import os
from typing import Dict, Any
from pathlib import Path

# Import our modules
from config import AutoencoderConfig, TabNetConfig, DatasetConfig, DATASET_CONFIGS
from protein_model import train_autoencoder_pipeline
from protein_feature_vis import visualize_features_pipeline
from tabnet_model import train_tabnet_pipeline

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_config_from_yaml(config_dict: Dict[str, Any], model_type: str = 'autoencoder') -> AutoencoderConfig:
    """Create AutoencoderConfig from YAML dictionary"""
    return AutoencoderConfig(
        hidden_size=int(config_dict.get('hidden_size', 8)),
        dropout_rate=float(config_dict.get('dropout_rate', 0.1)),
        num_epochs=int(config_dict.get('num_epochs', 200)),
        learning_rate=float(config_dict.get('learning_rate', 0.001)),
        patience=int(config_dict.get('patience', 20)),
        batch_size=int(config_dict.get('batch_size', 16)),
        weight_decay=float(config_dict.get('weight_decay', 1e-5)),
        test_size=float(config_dict.get('test_size', 0.2)),
        random_state=int(config_dict.get('random_state', 42)),
        base_path=str(config_dict.get('base_path', r"D:\ADNI\AD_CN\proteomics\Biomarkers Consortium Plasma Proteomics MRM"))
    )

def create_tabnet_config_from_yaml(config_dict: Dict[str, Any]) -> TabNetConfig:
    """Create TabNetConfig from YAML dictionary"""
    tabnet_params = config_dict.get('tabnet_params', {})
    return TabNetConfig(
        num_features=int(tabnet_params.get('num_features', 8)),
        feature_dim=int(tabnet_params.get('feature_dim', 64)),
        output_dim=int(tabnet_params.get('output_dim', 2)),
        num_decision_steps=int(tabnet_params.get('num_decision_steps', 3)),
        relaxation_factor=float(tabnet_params.get('relaxation_factor', 1.5)),
        sparsity_coefficient=float(tabnet_params.get('sparsity_coefficient', 1e-5)),
        batch_momentum=float(tabnet_params.get('batch_momentum', 0.98)),
        virtual_batch_size=int(tabnet_params.get('virtual_batch_size', 128)),
        mask_type=str(tabnet_params.get('mask_type', 'sparsemax')),
        num_epochs=int(config_dict.get('num_epochs', 200)),
        learning_rate=float(config_dict.get('learning_rate', 0.001)),
        patience=int(config_dict.get('patience', 20)),
        batch_size=int(config_dict.get('batch_size', 16)),
        weight_decay=float(config_dict.get('weight_decay', 1e-5)),
        test_size=float(config_dict.get('test_size', 0.2)),
        random_state=int(config_dict.get('random_state', 42)),
        base_path=str(config_dict.get('base_path', r"D:\ADNI\AD_CN\proteomics\Biomarkers Consortium Plasma Proteomics MRM"))
    )

def create_dataset_config_from_yaml(dataset_dict: Dict[str, Any]) -> DatasetConfig:
    """Create DatasetConfig from YAML dictionary"""
    return DatasetConfig(
        name=dataset_dict['name'],
        metadata_path=dataset_dict['metadata_path'],
        exclude_columns=dataset_dict.get('exclude_columns', 
                                      ['RID', 'VISCODE', 'MRI_acquired', 'research_group', 'subject_age'])
    )

def main():
    """Main pipeline controller"""
    parser = argparse.ArgumentParser(description='Protein Autoencoder and TabNet Pipeline')
    parser.add_argument('--config', type=str, default='configs/default.yml',
                       help='Path to YAML configuration file')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name to process')
    parser.add_argument('--model-type', type=str, choices=['autoencoder', 'tabnet'], default='autoencoder',
                       help='Model type to use (autoencoder or tabnet)')
    parser.add_argument('--train', action='store_true',
                       help='Train model')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize features')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory override')
    parser.add_argument('--experiment-name', type=str,
                       help='Custom experiment name (optional, will auto-generate if not provided)')
    
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Configuration file not found: {args.config}")
        print("Creating default configuration...")
        create_default_config(args.config)
    
    config_dict = load_config(args.config)
    
    # Create config objects based on model type
    if args.model_type == 'autoencoder':
        config = create_config_from_yaml(config_dict)
    else:  # tabnet
        config = create_tabnet_config_from_yaml(config_dict)
    
    # Get dataset configuration
    if args.dataset in config_dict.get('datasets', {}):
        dataset_dict = config_dict['datasets'][args.dataset]
        dataset_config = create_dataset_config_from_yaml(dataset_dict)
    elif args.dataset in DATASET_CONFIGS:
        dataset_config = DATASET_CONFIGS[args.dataset]
    else:
        raise ValueError(f"Dataset '{args.dataset}' not found in configuration")
    
    # Override output directory if specified
    if args.output_dir:
        config.base_path = args.output_dir
    
    print(f"Processing dataset: {args.dataset}")
    print(f"Model type: {args.model_type}")
    print(f"Metadata path: {dataset_config.metadata_path}")
    print(f"Output directory: {config.base_path}")
    if args.experiment_name:
        print(f"Experiment name: {args.experiment_name}")
    else:
        print("Experiment name: Auto-generated with timestamp")
    
    results = {}
    
    if args.train:
        print(f"Training {args.model_type}...")
        if args.model_type == 'autoencoder':
            training_results = train_autoencoder_pipeline(config, dataset_config, args.experiment_name)
        else:  # tabnet
            training_results = train_tabnet_pipeline(config, dataset_config, args.experiment_name)
        
        results['training'] = training_results
        print(f"Training completed! Results saved to: {training_results['experiment_dir']}")
    
    if args.visualize:
        print("Visualizing features...")
        # For visualization, we need to find the most recent experiment directory
        if 'training' in results:
            # Use the experiment directory from training
            experiment_dir = results['training']['experiment_dir']
        else:
            # Find the most recent experiment directory
            dataset_path = os.path.join(config.base_path, args.dataset)
            if os.path.exists(dataset_path):
                experiments = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
                if experiments:
                    # Sort by creation time and use the most recent
                    experiments.sort(key=lambda x: os.path.getctime(os.path.join(dataset_path, x)), reverse=True)
                    experiment_dir = os.path.join(dataset_path, experiments[0])
                    print(f"Using most recent experiment: {experiment_dir}")
                else:
                    raise ValueError(f"No experiment directories found in {dataset_path}")
            else:
                raise ValueError(f"Dataset directory not found: {dataset_path}")
        
        # For now, use the same visualization pipeline for both models
        # TODO: Create TabNet-specific visualization
        visualization_results = visualize_features_pipeline(config, dataset_config, experiment_dir)
        results['visualization'] = visualization_results
        print("Visualization completed!")
    
    return results

def create_default_config(config_path: str):
    """Create a default configuration file"""
    default_config = {
        'hidden_size': 8,
        'dropout_rate': 0.1,
        'num_epochs': 200,
        'learning_rate': 0.001,
        'patience': 20,
        'batch_size': 16,
        'weight_decay': 1e-5,
        'test_size': 0.2,
        'random_state': 42,
        'base_path': r"D:\ADNI\AD_CN\proteomics\Biomarkers Consortium Plasma Proteomics MRM",
        'tabnet_params': {
            'num_features': 8,
            'feature_dim': 64,
            'output_dim': 2,
            'num_decision_steps': 3,
            'relaxation_factor': 1.5,
            'sparsity_coefficient': 1e-5,
            'batch_momentum': 0.98,
            'virtual_batch_size': 128,
            'mask_type': 'sparsemax'
        },
        'datasets': {
            'mrm_small': {
                'name': 'mrm_small',
                'metadata_path': r"D:\ADNI\AD_CN\proteomics\Biomarkers Consortium Plasma Proteomics MRM\metadata.csv",
                'exclude_columns': ['RID', 'VISCODE', 'MRI_acquired', 'research_group', 'subject_age']
            },
            'mrm_large': {
                'name': 'mrm_large',
                'metadata_path': r"D:\ADNI\AD_CN\proteomics\Biomarkers Consortium Plasma Proteomics MRM\metadata_large.csv",
                'exclude_columns': ['RID', 'VISCODE', 'MRI_acquired', 'research_group', 'subject_age']
            }
        }
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)
    
    print(f"Created default configuration at: {config_path}")

# Convenience functions for programmatic use
def train_dataset(dataset_name: str, model_type: str = 'autoencoder', config_path: str = 'configs/default.yml', 
                 experiment_name: str = None) -> Dict[str, Any]:
    """Train model for a specific dataset"""
    config_dict = load_config(config_path)
    
    if model_type == 'autoencoder':
        config = create_config_from_yaml(config_dict)
    else:  # tabnet
        config = create_tabnet_config_from_yaml(config_dict)
    
    if dataset_name in config_dict.get('datasets', {}):
        dataset_dict = config_dict['datasets'][dataset_name]
        dataset_config = create_dataset_config_from_yaml(dataset_dict)
    else:
        dataset_config = DATASET_CONFIGS[dataset_name]
    
    if model_type == 'autoencoder':
        return train_autoencoder_pipeline(config, dataset_config, experiment_name)
    else:  # tabnet
        return train_tabnet_pipeline(config, dataset_config, experiment_name)

def visualize_dataset(dataset_name: str, config_path: str = 'configs/default.yml',
                    experiment_dir: str = None) -> Dict[str, Any]:
    """Visualize features for a specific dataset"""
    config_dict = load_config(config_path)
    config = create_config_from_yaml(config_dict)
    
    if dataset_name in config_dict.get('datasets', {}):
        dataset_dict = config_dict['datasets'][dataset_name]
        dataset_config = create_dataset_config_from_yaml(dataset_dict)
    else:
        dataset_config = DATASET_CONFIGS[dataset_name]
    
    return visualize_features_pipeline(config, dataset_config, experiment_dir)

def process_all_datasets(config_path: str = 'configs/default.yml', model_type: str = 'autoencoder') -> Dict[str, Any]:
    """Process all datasets defined in configuration"""
    config_dict = load_config(config_path)
    
    if model_type == 'autoencoder':
        config = create_config_from_yaml(config_dict)
    else:  # tabnet
        config = create_tabnet_config_from_yaml(config_dict)
    
    results = {}
    datasets = config_dict.get('datasets', {})
    
    for dataset_name in datasets:
        print(f"\nProcessing {dataset_name} with {model_type}...")
        dataset_dict = datasets[dataset_name]
        dataset_config = create_dataset_config_from_yaml(dataset_dict)
        
        # Train
        if model_type == 'autoencoder':
            training_results = train_autoencoder_pipeline(config, dataset_config)
        else:  # tabnet
            training_results = train_tabnet_pipeline(config, dataset_config)
        
        # Visualize
        visualization_results = visualize_features_pipeline(config, dataset_config, 
                                                        training_results['experiment_dir'])
        
        results[dataset_name] = {
            'training': training_results,
            'visualization': visualization_results
        }
    
    return results

if __name__ == "__main__":
    main()
