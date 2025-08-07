import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import os
import logging
from typing import Tuple, Dict, Any
from pytorch_tabnet.pretraining import TabNetPretrainer

# Import shared functionality
from base_model import (
    setup_logging, save_config_snapshot, load_and_prepare_data, 
    plot_training_curves, create_experiment_summary
)

def train_tabnet(X_train: np.ndarray, X_test: np.ndarray, 
                 y_train: np.ndarray, y_test: np.ndarray,
                 config: 'TabNetConfig', save_path: str, 
                 logger: logging.Logger = None) -> Tuple[TabNetPretrainer, Dict[str, Any]]:
    """Train TabNet pretrainer for unsupervised feature extraction"""
    if logger is None:
        logger = logging.getLogger('tabnet_classifier')
    
    # Initialize TabNetPretrainer for unsupervised learning
    tabnet = TabNetPretrainer(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=config.learning_rate, weight_decay=config.weight_decay),
        mask_type=config.mask_type
    )
    
    logger.info(f"TabNetPretrainer initialized with {X_train.shape[1]} input features")
    logger.info(f"Training parameters: epochs={config.num_epochs}, lr={config.learning_rate}")
    
    # Train TabNet pretrainer
    tabnet.fit(
        X_train=X_train,
        eval_set=[X_test],
        pretraining_ratio=0.8,
        max_epochs=config.num_epochs,
        patience=config.patience,
        batch_size=config.batch_size,
        virtual_batch_size=config.virtual_batch_size,
        num_workers=0,
        drop_last=False
    )
    
    # Save model
    tabnet.save_model(save_path)
    logger.info(f"TabNet pretrainer saved to: {save_path}")
    
    # Get training history with correct key names for pretrainer
    history = {
        'train_loss': tabnet.history['loss'],
        'val_loss': tabnet.history['val_0_unsup_loss_numpy'],  # Correct key for pretrainer
        'train_metrics': tabnet.history.get('metrics', []) if hasattr(tabnet.history, 'get') else [],
        'val_metrics': tabnet.history.get('val_metrics', []) if hasattr(tabnet.history, 'get') else []
    }
    
    return tabnet, history

def extract_tabnet_features(tabnet: TabNetPretrainer, X: np.ndarray) -> np.ndarray:
    """Extract features from trained TabNet pretrainer using forward pass"""
    # Convert to tensor
    X_tensor = torch.FloatTensor(X)
    
    # Use the pretrainer's forward method to get encoded representations
    tabnet.eval()
    with torch.no_grad():
        # The forward method should give us the encoded representations
        # According to TabNet documentation, this should be simpler
        encoded_features = tabnet(X_tensor)
        
        # Convert to numpy
        features = encoded_features.numpy()
        
        # Ensure 8-dimensional output
        if features.shape[1] > 8:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=8)
            features = pca.fit_transform(features)
        elif features.shape[1] < 8:
            # Pad with zeros if smaller than 8
            padded_features = np.zeros((features.shape[0], 8))
            padded_features[:, :features.shape[1]] = features
            features = padded_features
    
    return features

def train_tabnet_pipeline(config: 'TabNetConfig', dataset_config: 'DatasetConfig', 
                         experiment_name: str = None) -> Dict[str, Any]:
    """Complete TabNet pretraining pipeline with logging"""
    # Get file paths
    paths = config.get_paths(dataset_config.name, experiment_name)
    
    # Create experiment directory
    os.makedirs(paths['experiment_dir'], exist_ok=True)
    
    # Setup logging
    logger = setup_logging(paths['training_log'], 'tabnet_pretrainer')
    logger.info(f"Starting TabNet pretraining experiment: {dataset_config.name}")
    logger.info(f"Experiment directory: {paths['experiment_dir']}")
    
    # Save config snapshot
    save_config_snapshot(config, dataset_config, "configs/default.yml", paths['config_snapshot'], 'tabnet')
    
    # Load and prepare data (for unsupervised learning, we don't need labels)
    logger.info("Loading and preparing data...")
    X_scaled, scaler, proteomic_cols = load_and_prepare_data(
        dataset_config.metadata_path, dataset_config.exclude_columns, 'autoencoder'
    )
    logger.info(f"Data loaded: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
    
    # Split data
    X_train, X_test = train_test_split(
        X_scaled, test_size=config.test_size, random_state=config.random_state
    )
    logger.info(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    # Train TabNet pretrainer
    logger.info("Starting TabNet pretraining...")
    tabnet, history = train_tabnet(
        X_train, X_test, None, None, config, paths['best_model'], logger
    )
    
    # Extract features
    logger.info("Extracting TabNet features...")
    train_features = extract_tabnet_features(tabnet, X_train)
    test_features = extract_tabnet_features(tabnet, X_test)
    
    logger.info(f"Training features shape: {train_features.shape}")
    logger.info(f"Test features shape: {test_features.shape}")
    
    # Save results
    np.save(paths['train_features'], train_features)
    np.save(paths['test_features'], test_features)
    
    with open(paths['scaler'], 'wb') as f:
        pickle.dump(scaler, f)
    
    # Plot training curves
    plot_training_curves(history, paths['training_curves'], 'tabnet')
    
    # Create experiment summary
    create_experiment_summary(paths, config, dataset_config, train_features, test_features, 
                             history, logger, 'tabnet')
    
    logger.info("TabNet pretraining pipeline completed successfully!")
    
    return {
        'train_features': train_features,
        'test_features': test_features,
        'scaler': scaler,
        'proteomic_cols': proteomic_cols,
        'tabnet': tabnet,
        'history': history,
        'experiment_dir': paths['experiment_dir']
    } 