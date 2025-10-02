"""
Dataset loading and preprocessing for protein classification
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import Dataset


class ProteinDataset(Dataset):
    """PyTorch Dataset for protein features"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ProteinDataLoader:
    """Handles loading and preprocessing of protein data"""
    
    def __init__(self, data_path, label_col='research_group', id_col='RID', random_state=42):
        self.data_path = Path(data_path)
        self.label_col = label_col
        self.id_col = id_col
        self.random_state = random_state
        
        # Initialize preprocessing objects
        self.label_encoder = LabelEncoder()
        self.zero_var_cols = []
        self.feature_cols = []
        
    def load_data(self):
        """Load protein data from CSV"""
        df = pd.read_csv(self.data_path)
        print(f"[DATA] Loaded {len(df)} samples from {self.data_path.name}")
        return df
    
    def prepare_features(self, df, fit=True):
        """
        Extract and preprocess features (without scaling)
        
        Pipeline:
        1. Identify feature columns (exclude metadata)
        2. Impute missing values with median
        3. Remove zero-variance features
        4. Encode labels (AD=1, CN=0)
        
        Scaling is done separately per-fold in CV loop.
        
        Args:
            df: DataFrame with protein data
            fit: If True, identify zero-variance cols and fit label encoder
        
        Returns:
            X: DataFrame with preprocessed features (NOT scaled)
            y_encoded: Encoded labels (AD=1, CN=0)
        """
        # 1. Identify feature columns (exclude metadata)
        if not self.feature_cols:
            exclude_cols = [self.id_col, self.label_col, 'VISCODE', 'subject_age']
            self.feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        # 2. Extract features and impute missing values
        X = df[self.feature_cols].fillna(df[self.feature_cols].median())
        y = df[self.label_col]
        
        # 3. Remove zero-variance features
        if fit:
            self.zero_var_cols = X.columns[X.std() == 0].tolist()
            if self.zero_var_cols:
                print(f"   Removing {len(self.zero_var_cols)} zero-variance features")
        
        if self.zero_var_cols:
            X = X.drop(columns=self.zero_var_cols)
            self.feature_cols = [c for c in self.feature_cols if c not in self.zero_var_cols]
        
        # 4. Encode labels (AD=1, CN=0)
        if fit:
            y_encoded = self.label_encoder.fit_transform(y)
            y_encoded = 1 - y_encoded  # Swap: AD=1, CN=0
        else:
            y_encoded = self.label_encoder.transform(y)
            y_encoded = 1 - y_encoded
        
        return X, y_encoded  # Return DataFrame (not scaled)
    
    def get_train_test_split(self, train_path, test_path=None):
        """
        Load and prepare train and optional test sets
        
        Pipeline applied to both train and test:
        1. Load CSV
        2. Identify features (exclude metadata)
        3. Impute missing values
        4. Remove zero-variance features
        5. Encode labels
        
        NO SCALING - that's done per-fold in CV loop
        
        Args:
            train_path: Path to training CSV
            test_path: Optional path to test CSV
            
        Returns:
            X_train: DataFrame with preprocessed features (NOT scaled)
            y_train: Encoded labels (AD=1, CN=0)
            X_test: DataFrame with preprocessed features (NOT scaled), or None
            y_test: Encoded labels, or None
            train_df: Original training DataFrame with metadata
        """
        # Load and prepare training data
        self.data_path = Path(train_path)
        train_df = self.load_data()
        X_train, y_train = self.prepare_features(train_df, fit=True)
        
        print(f"   Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"   Label encoding: {dict(zip(self.label_encoder.classes_, [1, 0]))}")  # AD=1, CN=0
        
        # Show class distribution
        class_counts = pd.Series(train_df[self.label_col]).value_counts()
        print(f"   Class distribution:")
        for class_name, count in class_counts.items():
            print(f"     {class_name}: {count} ({count/len(train_df)*100:.1f}%)")
        
        # Load test data if provided
        X_test, y_test = None, None
        if test_path:
            try:
                self.data_path = Path(test_path)
                test_df = self.load_data()
                X_test, y_test = self.prepare_features(test_df, fit=False)
                print(f"   Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
            except FileNotFoundError:
                print(f"   Test set not found: {test_path}")
        
        return X_train, y_train, X_test, y_test, train_df