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
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.zero_var_cols = []
        self.feature_cols = []
        
    def load_data(self):
        """Load protein data from CSV"""
        df = pd.read_csv(self.data_path)
        print(f"📂 Loaded {len(df)} samples from {self.data_path.name}")
        return df
    
    def prepare_features(self, df, fit=True):
        """
        Extract and preprocess features
        
        Args:
            df: DataFrame with protein data
            fit: If True, fit scaler and identify zero-variance cols. If False, use existing.
        
        Returns:
            X_scaled: Scaled feature matrix
            y_encoded: Encoded labels (AD=1, CN=0)
            feature_cols: List of feature column names
        """
        # Identify feature columns (exclude metadata)
        if not self.feature_cols:
            exclude_cols = [self.id_col, self.label_col, 'VISCODE', 'original_index', 'fold', 'split_type']
            self.feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        # Extract features and labels
        X = df[self.feature_cols].fillna(df[self.feature_cols].median())
        y = df[self.label_col]
        
        if fit:
            # Remove zero variance features
            self.zero_var_cols = X.columns[X.std() == 0].tolist()
            if self.zero_var_cols:
                print(f"⚠️  Removing {len(self.zero_var_cols)} zero-variance features")
        
        # Drop zero variance columns
        if self.zero_var_cols:
            X = X.drop(columns=self.zero_var_cols)
            self.feature_cols = [c for c in self.feature_cols if c not in self.zero_var_cols]
        
        if fit:
            # Fit label encoder and scaler
            y_encoded = self.label_encoder.fit_transform(y)
            # Swap encoding: AD=1, CN=0
            y_encoded = 1 - y_encoded
            X_scaled = self.scaler.fit_transform(X)
        else:
            # Transform using fitted encoders
            y_encoded = self.label_encoder.transform(y)
            y_encoded = 1 - y_encoded
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, y_encoded
    
    def get_train_test_split(self, train_path, test_path=None):
        """
        Load and prepare train and optional test sets
        
        Args:
            train_path: Path to training CSV
            test_path: Optional path to test CSV
            
        Returns:
            X_train, y_train, X_test, y_test (test can be None)
        """
        # Load and prepare training data
        self.data_path = Path(train_path)
        train_df = self.load_data()
        X_train, y_train = self.prepare_features(train_df, fit=True)
        
        print(f"📊 Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"   Classes: {dict(zip(self.label_encoder.classes_, [1, 0]))}")  # AD=1, CN=0
        
        # Show class distribution
        class_counts = pd.Series(train_df[self.label_col]).value_counts()
        print(f"\n📈 Class distribution:")
        for class_name, count in class_counts.items():
            print(f"   • {class_name}: {count} ({count/len(train_df)*100:.1f}%)")
        
        # Load test data if provided
        X_test, y_test = None, None
        if test_path:
            try:
                self.data_path = Path(test_path)
                test_df = self.load_data()
                X_test, y_test = self.prepare_features(test_df, fit=False)
                print(f"\n📊 Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
            except FileNotFoundError:
                print(f"⚠️  Test set not found: {test_path}")
        
        return X_train, y_train, X_test, y_test, train_df