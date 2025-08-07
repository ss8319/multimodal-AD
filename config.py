import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime

@dataclass
class AutoencoderConfig:
    # Model parameters
    hidden_size: int = 8
    dropout_rate: float = 0.1
    
    # Training parameters
    num_epochs: int = 200
    learning_rate: float = 0.001
    patience: int = 20
    batch_size: int = 16
    weight_decay: float = 1e-5
    
    # Data parameters
    test_size: float = 0.2
    random_state: int = 42
    
    # File paths
    base_path: str = r"D:\ADNI\AD_CN\proteomics\Biomarkers Consortium Plasma Proteomics MRM"
    
    def get_paths(self, dataset_name: str, experiment_name: str = None) -> Dict[str, str]:
        """Generate file paths for a specific dataset"""
        if experiment_name is None:
            # Create unique experiment name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"{dataset_name}_{timestamp}"
        
        dataset_path = os.path.join(self.base_path, dataset_name)
        experiment_path = os.path.join(dataset_path, experiment_name)
        
        return {
            'metadata': os.path.join(dataset_path, 'metadata.csv'),
            'experiment_dir': experiment_path,
            'train_features': os.path.join(experiment_path, 'train_features_autoencoder.npy'),
            'test_features': os.path.join(experiment_path, 'test_features_autoencoder.npy'),
            'best_model': os.path.join(experiment_path, 'best_autoencoder.pth'),
            'scaler': os.path.join(experiment_path, 'scaler.pkl'),
            'training_curves': os.path.join(experiment_path, 'training_curves.png'),
            'feature_quality': os.path.join(experiment_path, 'feature_quality_analysis.png'),
            'dim_reduction': os.path.join(experiment_path, 'dimensionality_reduction_analysis.png'),
            'visualization_data': os.path.join(experiment_path, 'visualization_data.npy'),
            'training_log': os.path.join(experiment_path, 'training_log.txt'),
            'config_snapshot': os.path.join(experiment_path, 'config_snapshot.yml'),
            'experiment_summary': os.path.join(experiment_path, 'experiment_summary.txt')
        }

@dataclass
class TabNetConfig:
    # Model parameters
    num_features: int = 8  # Number of features to select
    feature_dim: int = 64  # Dimension of features
    output_dim: int = 2    # Number of classes (AD vs CN)
    num_decision_steps: int = 3
    relaxation_factor: float = 1.5
    sparsity_coefficient: float = 1e-5
    batch_momentum: float = 0.98
    virtual_batch_size: int = 128
    mask_type: str = 'sparsemax'
    
    # Training parameters
    num_epochs: int = 200
    learning_rate: float = 0.001
    patience: int = 20
    batch_size: int = 16
    weight_decay: float = 1e-5
    
    # Data parameters
    test_size: float = 0.2
    random_state: int = 42
    
    # File paths
    base_path: str = r"D:\ADNI\AD_CN\proteomics\Biomarkers Consortium Plasma Proteomics MRM"
    
    def get_paths(self, dataset_name: str, experiment_name: str = None) -> Dict[str, str]:
        """Generate file paths for TabNet experiments"""
        if experiment_name is None:
            # Create unique experiment name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"{dataset_name}_tabnet_{timestamp}"
        
        dataset_path = os.path.join(self.base_path, dataset_name)
        experiment_path = os.path.join(dataset_path, experiment_name)
        
        return {
            'metadata': os.path.join(dataset_path, 'metadata.csv'),
            'experiment_dir': experiment_path,
            'train_features': os.path.join(experiment_path, 'train_features_tabnet.npy'),
            'test_features': os.path.join(experiment_path, 'test_features_tabnet.npy'),
            'best_model': os.path.join(experiment_path, 'best_tabnet.pth'),
            'scaler': os.path.join(experiment_path, 'scaler.pkl'),
            'training_curves': os.path.join(experiment_path, 'training_curves.png'),
            'feature_importance': os.path.join(experiment_path, 'feature_importance.png'),
            'attention_masks': os.path.join(experiment_path, 'attention_masks.png'),
            'visualization_data': os.path.join(experiment_path, 'visualization_data.npy'),
            'training_log': os.path.join(experiment_path, 'training_log.txt'),
            'config_snapshot': os.path.join(experiment_path, 'config_snapshot.yml'),
            'experiment_summary': os.path.join(experiment_path, 'experiment_summary.txt'),
            'feature_selection': os.path.join(experiment_path, 'feature_selection_report.txt')
        }

@dataclass
class DatasetConfig:
    name: str
    metadata_path: str
    exclude_columns: list = None
    
    def __post_init__(self):
        if self.exclude_columns is None:
            self.exclude_columns = ['RID', 'VISCODE', 'MRI_acquired', 'research_group', 'subject_age']

# Predefined dataset configurations
DATASET_CONFIGS = {
    'mrm_small': DatasetConfig(
        name='mrm_small',
        metadata_path=r"D:\ADNI\AD_CN\proteomics\Biomarkers Consortium Plasma Proteomics MRM\metadata.csv"
    ),
    'mrm_large': DatasetConfig(
        name='mrm_large',
        metadata_path=r"D:\ADNI\AD_CN\proteomics\Biomarkers Consortium Plasma Proteomics MRM\metadata_large.csv"
    ),
    # Add more datasets as needed
}