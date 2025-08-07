import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import os
import yaml
import logging
from typing import Tuple, Dict, Any
from datetime import datetime

def setup_logging(log_path: str, logger_name: str = 'base_model') -> logging.Logger:
    """Setup logging to both file and console"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # Setup logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def save_config_snapshot(config: Any, dataset_config: 'DatasetConfig', 
                        config_path: str, save_path: str, model_type: str = 'autoencoder'):
    """Save a snapshot of the current configuration"""
    if model_type == 'autoencoder':
        config_dict = {
            'model_params': {
                'hidden_size': config.hidden_size,
                'dropout_rate': config.dropout_rate
            },
            'training_params': {
                'num_epochs': config.num_epochs,
                'learning_rate': config.learning_rate,
                'patience': config.patience,
                'batch_size': config.batch_size,
                'weight_decay': config.weight_decay
            },
            'data_params': {
                'test_size': config.test_size,
                'random_state': config.random_state
            },
            'dataset': {
                'name': dataset_config.name,
                'metadata_path': dataset_config.metadata_path,
                'exclude_columns': dataset_config.exclude_columns
            },
            'original_config_path': config_path,
            'timestamp': datetime.now().isoformat()
        }
    else:  # tabnet
        config_dict = {
            'model_params': {
                'num_features': config.num_features,
                'feature_dim': config.feature_dim,
                'output_dim': config.output_dim,
                'num_decision_steps': config.num_decision_steps,
                'relaxation_factor': config.relaxation_factor,
                'sparsity_coefficient': config.sparsity_coefficient,
                'batch_momentum': config.batch_momentum,
                'virtual_batch_size': config.virtual_batch_size,
                'mask_type': config.mask_type
            },
            'training_params': {
                'num_epochs': config.num_epochs,
                'learning_rate': config.learning_rate,
                'patience': config.patience,
                'batch_size': config.batch_size,
                'weight_decay': config.weight_decay
            },
            'data_params': {
                'test_size': config.test_size,
                'random_state': config.random_state
            },
            'dataset': {
                'name': dataset_config.name,
                'metadata_path': dataset_config.metadata_path,
                'exclude_columns': dataset_config.exclude_columns
            },
            'original_config_path': config_path,
            'timestamp': datetime.now().isoformat()
        }
    
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)

def load_and_prepare_data(metadata_path: str, exclude_columns: list, model_type: str = 'autoencoder') -> Tuple[np.ndarray, np.ndarray, object, list]:
    """Load and prepare data for model training"""
    df = pd.read_csv(metadata_path)
    
    # Extract proteomic features
    proteomic_cols = [col for col in df.columns if col not in exclude_columns]
    
    # Prepare features
    X = df[proteomic_cols].values
    print(f"Data shape: {X.shape}")
    
    # Handle missing values
    X = np.nan_to_num(X, nan=0.0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if model_type == 'autoencoder':
        # For autoencoder, we don't need labels
        return X_scaled, scaler, proteomic_cols
    else:  # tabnet
        # For TabNet, we need labels for supervised learning
        y = (df['research_group'] == 'AD').astype(int).values
        return X_scaled, y, scaler, proteomic_cols

def plot_training_curves(history: Dict[str, Any], save_path: str, model_type: str = 'autoencoder'):
    """Plot training curves for both models"""
    if model_type == 'autoencoder':
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Autoencoder Training Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.semilogy(history['train_loss'], label='Train Loss')
        plt.semilogy(history['val_loss'], label='Val Loss')
        plt.title('Autoencoder Training Curves (Log Scale)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log)')
        plt.legend()
        
    else:  # tabnet
        plt.figure(figsize=(15, 5))
        
        # Loss curves
        plt.subplot(1, 3, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('TabNet Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Metrics curves
        if 'train_metrics' in history and history['train_metrics']:
            plt.subplot(1, 3, 2)
            train_metrics = np.array(history['train_metrics'])
            val_metrics = np.array(history['val_metrics'])
            
            plt.plot(train_metrics[:, 0], label='Train Accuracy')
            plt.plot(val_metrics[:, 0], label='Val Accuracy')
            plt.title('TabNet Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
        
        # Feature importance (only if available)
        plt.subplot(1, 3, 3)
        feature_importance = history.get('feature_importance', [])
        if len(feature_importance) > 0:
            plt.bar(range(len(feature_importance)), feature_importance)
            plt.title('Feature Importance')
            plt.xlabel('Feature Index')
            plt.ylabel('Importance')
        else:
            plt.text(0.5, 0.5, 'Feature importance\nnot available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Feature Importance')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_experiment_summary(paths: Dict[str, str], config: Any, dataset_config: 'DatasetConfig', 
                            train_features: np.ndarray, test_features: np.ndarray, 
                            history: Dict[str, Any], logger: logging.Logger, model_type: str = 'autoencoder'):
    """Create a summary of the experiment"""
    if model_type == 'autoencoder':
        summary = f"""
AUTOENCODER EXPERIMENT SUMMARY
==============================

Dataset: {dataset_config.name}
Experiment Directory: {paths['experiment_dir']}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

MODEL PARAMETERS:
- Hidden size: {config.hidden_size}
- Dropout rate: {config.dropout_rate}
- Input features: {train_features.shape[1] if len(train_features.shape) > 1 else 'N/A'}

TRAINING PARAMETERS:
- Epochs: {config.num_epochs}
- Learning rate: {config.learning_rate}
- Batch size: {config.batch_size}
- Patience: {config.patience}
- Weight decay: {config.weight_decay}

DATA:
- Training samples: {train_features.shape[0]}
- Test samples: {test_features.shape[0]}
- Feature dimension: {train_features.shape[1] if len(train_features.shape) > 1 else 'N/A'}

RESULTS:
- Final train loss: {history['train_loss'][-1]:.6f}
- Final validation loss: {history['val_loss'][-1]:.6f}
- Best validation loss: {min(history['val_loss']):.6f}
- Training epochs: {len(history['train_loss'])}

FILES SAVED:
- Model: {paths['best_model']}
- Training curves: {paths['training_curves']}
- Train features: {paths['train_features']}
- Test features: {paths['test_features']}
- Scaler: {paths['scaler']}
- Config snapshot: {paths['config_snapshot']}
- Training log: {paths['training_log']}
"""
    else:  # tabnet
        summary = f"""
TABNET EXPERIMENT SUMMARY
=========================

Dataset: {dataset_config.name}
Experiment Directory: {paths['experiment_dir']}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

MODEL PARAMETERS:
- Number of features: {config.num_features}
- Feature dimension: {config.feature_dim}
- Output dimension: {config.output_dim}
- Decision steps: {config.num_decision_steps}
- Relaxation factor: {config.relaxation_factor}
- Sparsity coefficient: {config.sparsity_coefficient}

TRAINING PARAMETERS:
- Epochs: {config.num_epochs}
- Learning rate: {config.learning_rate}
- Batch size: {config.batch_size}
- Patience: {config.patience}
- Weight decay: {config.weight_decay}

DATA:
- Training samples: {train_features.shape[0]}
- Test samples: {test_features.shape[0]}
- Feature dimension: {train_features.shape[1] if len(train_features.shape) > 1 else 'N/A'}

RESULTS:
- Final train loss: {history['train_loss'][-1]:.6f}
- Final validation loss: {history['val_loss'][-1]:.6f}
- Best validation loss: {min(history['val_loss']):.6f}
- Training epochs: {len(history['train_loss'])}

FILES SAVED:
- Model: {paths['best_model']}
- Training curves: {paths['training_curves']}
- Train features: {paths['train_features']}
- Test features: {paths['test_features']}
- Scaler: {paths['scaler']}
- Config snapshot: {paths['config_snapshot']}
- Training log: {paths['training_log']}
"""
    
    with open(paths['experiment_summary'], 'w') as f:
        f.write(summary)
    
    logger.info(f"{model_type.title()} experiment summary saved")
    print(summary) 