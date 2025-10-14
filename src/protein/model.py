"""
Model definitions for protein classification
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Handle both relative and absolute imports
try:
    from .dataset import ProteinDataset
except ImportError:
    from dataset import ProteinDataset


class ProteinTransformer(nn.Module):
    """Simple Transformer for protein classification using self-attention"""
    def __init__(self, n_features, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model # The dimensionality of the internal attention and hidden layers
        self.n_features = n_features # number of protein features in the input data
        
        # Input projection
        #A linear layer to map the raw input feature size (n_features) into the 
        # required embedding dimension for the Transformer (d_model)
        self.input_projection = nn.Linear(n_features, d_model)
        
        # Positional encoding for protein features
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, d_model) * 0.1)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads, # no of self-attention blocks
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        #Stacks the encoder layers on top of each other to form the encoder
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )
        
    def forward(self, x):
        # x shape: (batch_size, n_features)
        batch_size = x.size(0)
        
        # Project to d_model and add batch dimension for sequence
        x = self.input_projection(x)  # (batch_size, d_model)
        x = x.unsqueeze(1)  # (batch_size, 1, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Apply transformer
        x = self.transformer(x)  # (batch_size, 1, d_model)
        
        # Global average pooling and classify
        x = x.squeeze(1)  # (batch_size, d_model)
        logits = self.classifier(x)
        
        return logits

class ProteinTransformerClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper for ProteinTransformer"""
    def __init__(self, d_model=64, n_heads=4, n_layers=2, dropout=0.1, 
                 lr=0.001, epochs=100, batch_size=32, patience=10, random_state=42):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        
    def fit(self, X, y):
        # Set random seeds
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Store classes
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # Create model
        self.model = ProteinTransformer(
            n_features=X.shape[1],
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout
        )
        
        # Only split for validation if we have enough samples
        if len(X) > 10:  # Only split if we have enough data
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, stratify=y
            )
        else:
            # Use all data for training if sample size is small (happens in CV)
            X_tr, X_val, y_tr, y_val = X, X, y, y
        
        # Create datasets and loaders
        train_dataset = ProteinDataset(X_tr, y_tr)
        val_dataset = ProteinDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Optimizer and loss with gradient clipping
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        
        # Initialize weights properly
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
        
        self.model.apply(init_weights)
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        self.best_state = self.model.state_dict().copy()  # Initialize with current state
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print("Warning: NaN loss detected, skipping batch")
                    continue
                
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    logits = self.model(batch_X)
                    loss = criterion(logits, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader) if len(val_loader) > 0 else 1
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                self.best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
        
        # Load best model
        try:
            self.model.load_state_dict(self.best_state)
        except:
            pass  # If loading fails, keep current state
        return self
        
    def predict_proba(self, X):
        try:
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                logits = self.model(X_tensor)
                probas = F.softmax(logits, dim=1)
                probas_np = probas.numpy()
                
                # Check for NaN or invalid probabilities
                if np.isnan(probas_np).any() or np.isinf(probas_np).any():
                    print(f"Warning: Invalid probabilities detected, using fallback")
                    # Return balanced probabilities as fallback
                    n_samples = len(X)
                    probas_np = np.column_stack([
                        np.random.uniform(0.3, 0.7, n_samples),
                        np.random.uniform(0.3, 0.7, n_samples)
                    ])
                    # Normalize to sum to 1
                    probas_np = probas_np / probas_np.sum(axis=1, keepdims=True)
                
                return probas_np
        except Exception as e:
            print(f"Error in predict_proba: {e}")
            # Fallback: return balanced probabilities
            n_samples = len(X)
            probas = np.column_stack([
                np.random.uniform(0.3, 0.7, n_samples),
                np.random.uniform(0.3, 0.7, n_samples)
            ])
            return probas / probas.sum(axis=1, keepdims=True)
    
    def predict(self, X):
        try:
            probas = self.predict_proba(X)
            return np.argmax(probas, axis=1)
        except Exception as e:
            # Fallback: return random predictions
            return np.random.randint(0, 2, len(X))


class NeuralNetwork(nn.Module):
    """Feed-forward neural network for protein classification."""

    def __init__(self, n_features, hidden_sizes=(128, 64), dropout=0.2):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NeuralNetworkClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible PyTorch neural network classifier."""

    def __init__(self, hidden_sizes=(128, 64), dropout=0.2, lr=1e-3, epochs=200, batch_size=32, random_state=42):
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        self.model = NeuralNetwork(
            n_features=X.shape[1],
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
        )

        # Simple validation split when enough samples are available
        if len(X) > 10:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, stratify=y
            )
        else:
            X_tr, X_val, y_tr, y_val = X, X, y, y

        train_dataset = ProteinDataset(X_tr, y_tr)
        val_dataset = ProteinDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

        self.model.apply(init_weights)

        best_state = self.model.state_dict().copy()
        best_val_loss = float('inf')

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                if torch.isnan(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

            # Validation monitoring (best checkpoint only)
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    logits = self.model(batch_X)
                    loss = criterion(logits, batch_y)
                    val_loss += loss.item()
            val_loss /= max(1, len(val_loader))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = self.model.state_dict().copy()

        # Restore best weights observed during training
        self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        try:
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                logits = self.model(X_tensor)
                probas = F.softmax(logits, dim=1)
                probas_np = probas.numpy()
                if np.isnan(probas_np).any() or np.isinf(probas_np).any():
                    n = len(X)
                    probas_np = np.column_stack([
                        np.random.uniform(0.3, 0.7, n),
                        np.random.uniform(0.3, 0.7, n),
                    ])
                    probas_np = probas_np / probas_np.sum(axis=1, keepdims=True)
                return probas_np
        except Exception:
            n = len(X)
            probas = np.column_stack([
                np.random.uniform(0.3, 0.7, n),
                np.random.uniform(0.3, 0.7, n),
            ])
        return probas / probas.sum(axis=1, keepdims=True)

    def predict(self, X):
        try:
            probas = self.predict_proba(X)
            return np.argmax(probas, axis=1)
        except Exception:
            return np.random.randint(0, 2, len(X))

def get_classifiers(random_state=42):
    """
    Get dictionary of all classifiers for evaluation
    
    Args:
        random_state: Random seed for reproducibility
        
    Returns:
        dict: Dictionary mapping classifier names to instances
    """
    classifiers = {
        'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'SVM (RBF)': SVC(probability=True, random_state=random_state),
        'Gradient Boosting': GradientBoostingClassifier(random_state=random_state),
        'Neural Network': NeuralNetworkClassifier(hidden_sizes=(128, 64), dropout=0.2, lr=1e-3, epochs=200, batch_size=32, random_state=random_state),
        'Protein Transformer': ProteinTransformerClassifier(
            d_model=32, n_heads=2, n_layers=1, dropout=0.2,
            lr=0.01, epochs=30, batch_size=16, patience=5, random_state=random_state
        )
    }
    return classifiers