import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
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

def setup_logging(log_path: str) -> logging.Logger:
    """Setup logging to both file and console"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # Setup logger
    logger = logging.getLogger('protein_autoencoder')
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

def save_config_snapshot(config: 'AutoencoderConfig', dataset_config: 'DatasetConfig', 
                        config_path: str, save_path: str):
    """Save a snapshot of the current configuration"""
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
    
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)

def train_autoencoder(model: nn.Module, train_loader, val_loader, 
                     num_epochs: int = 200, learning_rate: float = 0.001, 
                     patience: int = 20, weight_decay: float = 1e-5,
                     save_path: str = None, logger: logging.Logger = None) -> Tuple[list, list]:
    """Train autoencoder with early stopping and logging"""
    if logger is None:
        logger = logging.getLogger('protein_autoencoder')
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
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
    
    # Plot training curves
    if save_path:
        plot_training_curves(train_losses, val_losses, save_path.replace('.pth', '_curves.png'))
    
    return train_losses, val_losses

def plot_training_curves(train_losses: list, val_losses: list, save_path: str):
    """Plot training curves"""
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.semilogy(train_losses, label='Train Loss')
    plt.semilogy(val_losses, label='Val Loss')
    plt.title('Training Curves (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def extract_features(model: nn.Module, data_loader) -> np.ndarray:
    """Extract features from trained model"""
    model.eval()
    features = []
    with torch.no_grad():
        for batch_x, _ in data_loader:
            hidden_features = model.extract_features(batch_x) # Get the bottleneck features
            features.append(hidden_features.numpy())
    return np.concatenate(features, axis=0)

def load_and_prepare_data(metadata_path: str, exclude_columns: list) -> Tuple[np.ndarray, np.ndarray, list]:
    """Load and prepare data for training"""
    df = pd.read_csv(metadata_path)
    
    # Extract proteomic features
    proteomic_cols = [col for col in df.columns if col not in exclude_columns]
    
    # Prepare data
    X = df[proteomic_cols].values
    print(f"Data shape: {X.shape}")
    
    # Handle missing values
    X = np.nan_to_num(X, nan=0.0)
    
    # Scale features
    # StandardScaler does: (x - mean) / std
    # Result: mean=0, std=1 for each feature
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler, proteomic_cols

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
    logger = setup_logging(paths['training_log'])
    logger.info(f"Starting experiment: {dataset_config.name}")
    logger.info(f"Experiment directory: {paths['experiment_dir']}")
    
    # Save config snapshot
    save_config_snapshot(config, dataset_config, "configs/default.yml", paths['config_snapshot'])
    
    # Load and prepare data
    logger.info("Loading and preparing data...")
    X_scaled, scaler, proteomic_cols = load_and_prepare_data(
        dataset_config.metadata_path, dataset_config.exclude_columns
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
    
    # Create experiment summary
    create_experiment_summary(paths, config, dataset_config, train_features, test_features, 
                             train_losses, val_losses, logger)
    
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

def create_experiment_summary(paths: Dict[str, str], config: 'AutoencoderConfig', 
                            dataset_config: 'DatasetConfig', train_features: np.ndarray,
                            test_features: np.ndarray, train_losses: list, val_losses: list,
                            logger: logging.Logger):
    """Create a summary of the experiment"""
    summary = f"""
EXPERIMENT SUMMARY
==================

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
- Final train loss: {train_losses[-1]:.6f}
- Final validation loss: {val_losses[-1]:.6f}
- Best validation loss: {min(val_losses):.6f}
- Training epochs: {len(train_losses)}

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
    
    logger.info("Experiment summary saved")
    print(summary)
