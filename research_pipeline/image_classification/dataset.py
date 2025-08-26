"""
MRI Dataset and data utilities for 3D brain image classification.

This module provides the MRIDataset class and related utilities for loading
and preprocessing 3D MRI data from DICOM files.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pydicom
from pathlib import Path
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

class MRIDataset(Dataset):
    """
    Dataset class for loading 3D MRI data from DICOM files.
    
    This class handles loading DICOM volumes, preprocessing them to a standard size,
    and applying data augmentation during training.
    """
    
    def __init__(self, 
                 dataframe, 
                 adni_base_path: str,
                 target_size: Tuple[int, int, int] = (128, 128, 128),
                 augment: bool = False,
                 normalization_method: str = 'zscore',
                 percentile_range: Tuple[int, int] = (1, 99),
                 max_files_per_subject: Optional[int] = None):
        """
        Initialize the MRI dataset.
        
        Args:
            dataframe: DataFrame containing subject information and DICOM paths
            adni_base_path: Base path to ADNI data
            target_size: Target 3D volume size (D, H, W)
            augment: Whether to apply data augmentation
            normalization_method: Normalization method ('zscore', 'minmax', 'percentile')
            percentile_range: Percentile range for normalization (if using percentile)
            max_files_per_subject: Maximum DICOM files to load per subject
        """
        self.df = dataframe.dropna(subset=['dicom_folder_path']).reset_index(drop=True)
        self.adni_base_path = Path(adni_base_path)
        self.target_size = target_size
        self.augment = augment
        self.normalization_method = normalization_method
        self.percentile_range = percentile_range
        self.max_files_per_subject = max_files_per_subject
        
        # Label encoding: AD=1, CN=0
        self.df['label'] = (self.df['Group'] == 'AD').astype(int)
        
        print(f"📊 Dataset loaded: {len(self.df)} samples")
        print(f"   • AD: {(self.df['label'] == 1).sum()}")
        print(f"   • CN: {(self.df['label'] == 0).sum()}")
        print(f"   • Target size: {target_size}")
        print(f"   • Augmentation: {'Enabled' if augment else 'Disabled'}")
    
    def __len__(self):
        return len(self.df)
    
    def load_dicom_volume(self, dicom_folder_path: str) -> np.ndarray:
        """
        Load DICOM files from folder and create complete 3D volume.
        
        Args:
            dicom_folder_path: Relative path to DICOM folder
            
        Returns:
            3D numpy array representing the MRI volume
        """
        full_path = self.adni_base_path / dicom_folder_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"DICOM folder not found: {full_path}")
        
        # Get all DICOM files recursively
        dicom_files = []
        for root, dirs, files in os.walk(full_path):
            for file in files:
                if file.lower().endswith('.dcm'):
                    dicom_files.append(os.path.join(root, file))
        
        if not dicom_files:
            raise ValueError(f"No DICOM files found in: {full_path}")
        
        # Limit files if specified
        if self.max_files_per_subject is not None:
            dicom_files = dicom_files[:self.max_files_per_subject]
        
        # Load and sort DICOM files with better sorting logic
        dicom_data = []
        for dcm_file in dicom_files:
            try:
                ds = pydicom.dcmread(dcm_file)
                if hasattr(ds, 'pixel_array'):
                    # Try multiple sorting methods
                    sort_key = None
                    if hasattr(ds, 'SliceLocation'):
                        sort_key = float(ds.SliceLocation)
                    elif hasattr(ds, 'ImagePositionPatient') and len(ds.ImagePositionPatient) >= 3:
                        sort_key = float(ds.ImagePositionPatient[2])  # Z-coordinate
                    elif hasattr(ds, 'InstanceNumber'):
                        sort_key = int(ds.InstanceNumber)
                    else:
                        sort_key = 0  # Fallback
                    
                    dicom_data.append((sort_key, ds.pixel_array))
            except Exception as e:
                print(f"     ⚠️ Failed to read {dcm_file}: {e}")
                continue
        
        if not dicom_data:
            raise ValueError(f"No valid DICOM data found in: {full_path}")
        
        # Sort by the chosen key and stack
        dicom_data.sort(key=lambda x: x[0])
        volume = np.stack([data[1] for data in dicom_data], axis=0)
        
        return volume.astype(np.float32)
    
    def preprocess_volume(self, volume: np.ndarray, target_size: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        """
        Preprocess 3D volume: robust normalization and intelligent resizing.
        
        Args:
            volume: Input 3D volume
            target_size: Target size (uses self.target_size if None)
            
        Returns:
            Preprocessed volume with target size
        """
        if target_size is None:
            target_size = self.target_size
        
        # 1. Robust intensity normalization
        if self.normalization_method == 'percentile':
            # Percentile-based normalization + z-score for stability
            p_lo, p_hi = np.percentile(volume, self.percentile_range)
            volume = np.clip(volume, p_lo, p_hi)
            volume = (volume - volume.mean()) / (volume.std() + 1e-8)
        elif self.normalization_method == 'zscore':
            # Z-score normalization (better than min-max for MRI)
            volume = (volume - volume.mean()) / (volume.std() + 1e-8)
        elif self.normalization_method == 'minmax':
            # Min-max normalization
            volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization_method}")
        
        # 2. Intelligent resizing
        current_shape = volume.shape
        
        # Use scipy for better interpolation if available
        try:
            from scipy.ndimage import zoom
            
            # Calculate zoom factors
            zoom_factors = [
                target_size[i] / current_shape[i] for i in range(3)
            ]
            
            # Apply zoom with better interpolation
            resized = zoom(volume, zoom_factors, order=1, mode='nearest')
            
        except ImportError:
            # Fallback to simple downsampling
            print("     ⚠️ Using simple downsampling (scipy not available)")
            resize_factors = [max(1, current_shape[i] // target_size[i]) for i in range(3)]
            resized = volume[::resize_factors[0], ::resize_factors[1], ::resize_factors[2]]
            
            # Pad or crop to exact target size
            final_volume = np.zeros(target_size)
            min_shape = [min(resized.shape[i], target_size[i]) for i in range(3)]
            final_volume[:min_shape[0], :min_shape[1], :min_shape[2]] = resized[:min_shape[0], :min_shape[1], :min_shape[2]]
            resized = final_volume
        
        return resized.astype(np.float32)
    
    def apply_augmentation(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Apply data augmentation to the volume.
        
        Args:
            volume: Input volume tensor (1, D, H, W)
            
        Returns:
            Augmented volume tensor
        """
        if not self.augment:
            return volume
        
        # Random rotation (small angles)
        if torch.rand(1).item() < 0.5:
            angle = torch.rand(1).item() * 10 - 5  # -5 to +5 degrees
            volume = torch.rot90(volume, k=int(angle/90), dims=[2, 3])
        
        # Random intensity scaling
        if torch.rand(1).item() < 0.5:
            scale = 0.8 + torch.rand(1).item() * 0.4  # 0.8 to 1.2
            volume = volume * scale
        
        # Random noise (small amount)
        if torch.rand(1).item() < 0.3:
            noise = torch.randn_like(volume) * 0.05
            volume = volume + noise
        
        return volume
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (volume_tensor, label_tensor)
        """
        row = self.df.iloc[idx]
        
        try:
            # Load DICOM volume
            volume = self.load_dicom_volume(row['dicom_folder_path'])
            
            # Preprocess to standard size
            volume = self.preprocess_volume(volume)
            
            # Add channel dimension and convert to tensor
            volume = torch.FloatTensor(volume).unsqueeze(0)  # Shape: (1, D, H, W)
            
            # Apply data augmentation if enabled
            volume = self.apply_augmentation(volume)
            
            label = torch.LongTensor([row['label']])[0]
            
            return volume, label
            
        except Exception as e:
            print(f"   ❌ Error loading {row['Subject']}: {e}")
            print(f"   🗗️ Returning dummy sample for {row['Subject']}")
            # Return a dummy sample
            dummy_volume = torch.zeros(1, *self.target_size)
            return dummy_volume, torch.LongTensor([0])[0]

def create_data_loaders(train_dataset: MRIDataset, 
                       val_dataset: MRIDataset, 
                       test_dataset: MRIDataset,
                       batch_size: int = 16,
                       num_workers: int = 0) -> Tuple[torch.utils.data.DataLoader, 
                                                     torch.utils.data.DataLoader, 
                                                     torch.utils.data.DataLoader]:
    """
    Create data loaders for training, validation, and test sets.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size for all loaders
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader

def split_dataframe_for_validation(dataframe, train_val_split: float = 0.8):
    """
    Split dataframe into training and validation subsets.
    
    Args:
        dataframe: Input dataframe
        train_val_split: Fraction of data to use for training
        
    Returns:
        Tuple of (train_subset, val_subset)
    """
    train_size = int(train_val_split * len(dataframe))
    train_subset = dataframe.iloc[:train_size]
    val_subset = dataframe.iloc[train_size:]
    
    print(f"📊 Data split:")
    print(f"   • Training: {len(train_subset)} samples")
    print(f"   • Validation: {len(val_subset)} samples")
    
    return train_subset, val_subset
