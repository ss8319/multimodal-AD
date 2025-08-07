import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import os
import logging
from typing import Tuple, Dict, Any
from datetime import datetime

# Import shared functionality
from base_model import (
    setup_logging, save_config_snapshot, load_and_prepare_data, 
    plot_training_curves, create_experiment_summary
)

class ProteinAutoencoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 8, dropout_rate: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2), # input_size -> hidden_size * 2
            nn.BatchNorm1d(hidden_size * 2),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, hidden_size), # (BOTTLENECK) hidden_size * 2 -> hidden_size
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, input_size),
        )
        
        print(f"Autoencoder: {input_size} -> {hidden_size} -> {input_size}")

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

    def extract_features(self, x):
        return self.encoder(x)

def train_autoencoder(model: nn.Module, train_loader, val_loader, 
                     num_epochs: int = 200, learning_rate: float = 0.001, 
                     patience: int = 20, weight_decay: float = 1e-5,
                     save_path: str = None, logger: logging.Logger = None) -> Tuple[list, list]:
    """Train autoencoder with early stopping and logging"""
    if logger is None:
        logger = logging.getLogger('protein_autoencoder')
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    logger.info(f"Starting training with {num_epochs} epochs")
    logger.info(f"Learning rate: {learning_rate}, Weight decay: {weight_decay}")
    logger.info(f"Patience: {patience}, Batch size: {len(next(iter(train_loader))[0])}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_x, _ in train_loader:
            optimizer.zero_grad()
            decoded, encoded = model(batch_x)
            loss = criterion(decoded, batch_x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, _ in val_loader:
                decoded, encoded = model(batch_x)
                loss = criterion(decoded, batch_x)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
                logger.info(f"New best model saved at epoch {epoch+1}")
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            logger.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        
        if patience_counter >= patience:
            logger.info(f'Early stopping at epoch {epoch+1}')
            break
    
    logger.info(f"Training completed. Best validation loss: {best_val_loss:.6f}")
    
    return train_losses, val_losses

def extract_features(model: nn.Module, data_loader) -> np.ndarray:
    """Extract features from trained model"""
    model.eval()
    features = []
    with torch.no_grad():
        for batch_x, _ in data_loader:
            hidden_features = model.extract_features(batch_x) # Get the bottleneck features
            features.append(hidden_features.numpy())
    return np.concatenate(features, axis=0)

def create_data_loaders(X_train: np.ndarray, X_test: np.ndarray, 
                       batch_size: int = 16) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create data loaders for training"""
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, torch.zeros(len(X_train_tensor)))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, torch.zeros(len(X_test_tensor)))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_autoencoder_pipeline(config: 'AutoencoderConfig', dataset_config: 'DatasetConfig', 
                              experiment_name: str = None) -> Dict[str, Any]:
    """Complete autoencoder training pipeline with logging"""
    # Get file paths
    paths = config.get_paths(dataset_config.name, experiment_name)
    
    # Create experiment directory
    os.makedirs(paths['experiment_dir'], exist_ok=True)
    
    # Setup logging
    logger = setup_logging(paths['training_log'], 'protein_autoencoder')
    logger.info(f"Starting experiment: {dataset_config.name}")
    logger.info(f"Experiment directory: {paths['experiment_dir']}")
    
    # Save config snapshot
    save_config_snapshot(config, dataset_config, "configs/default.yml", paths['config_snapshot'], 'autoencoder')
    
    # Load and prepare data
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
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(X_train, X_test, config.batch_size)
    
    # Initialize model
    input_size = len(proteomic_cols) # number of features
    model = ProteinAutoencoder(input_size, config.hidden_size, config.dropout_rate)
    logger.info(f"Model initialized: {input_size} -> {config.hidden_size} -> {input_size}")
    
    # Train the autoencoder
    logger.info("Starting training...")
    train_losses, val_losses = train_autoencoder(
        model, train_loader, test_loader, 
        num_epochs=config.num_epochs, 
        learning_rate=config.learning_rate, 
        patience=config.patience,
        weight_decay=config.weight_decay,
        save_path=paths['best_model'],
        logger=logger
    )
    
    # Load best model
    model.load_state_dict(torch.load(paths['best_model']))
    logger.info("Best model loaded for feature extraction")
    
    # Extract features
    logger.info("Extracting features...")
    train_features = extract_features(model, train_loader)
    test_features = extract_features(model, test_loader)
    
    logger.info(f"Training features shape: {train_features.shape}")
    logger.info(f"Test features shape: {test_features.shape}")
    
    # Save results
    np.save(paths['train_features'], train_features)
    np.save(paths['test_features'], test_features)
    
    with open(paths['scaler'], 'wb') as f:
        pickle.dump(scaler, f)
    
    # Create training history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses
    }
    
    # Plot training curves
    plot_training_curves(history, paths['training_curves'], 'autoencoder')
    
    # Create experiment summary
    create_experiment_summary(paths, config, dataset_config, train_features, test_features, 
                             history, logger, 'autoencoder')
    
    logger.info("Training pipeline completed successfully!")
    
    return {
        'train_features': train_features,
        'test_features': test_features,
        'scaler': scaler,
        'proteomic_cols': proteomic_cols,
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'experiment_dir': paths['experiment_dir']
    }
