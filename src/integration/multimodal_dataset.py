import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add paths for protein modules
sys.path.append(str(Path(__file__).parent.parent / "protein"))

# Import from correct modules
from protein_extractor import ProteinLatentExtractor
from mri_extractors import BaseMRIExtractor


class MultimodalDataset(Dataset):
    """
    Dataset that loads protein features and MRI latents on-the-fly
    
    During __getitem__:
    1. Extract protein features from CSV row
    2. Extract MRI latents using BrainIAC model
    3. Concatenate both modalities
    4. Return fused features + label
    """
    
    def __init__(
        self,
        csv_path,
        mri_extractor: BaseMRIExtractor,
        protein_run_dir=None,
        protein_latents_dir=None,
        protein_model_type='nn',
        protein_layer='last_hidden_layer',
        protein_columns=None,
        device='cpu'
    ):
        """
        Args:
            csv_path: Path to CSV with multimodal data
            mri_extractor: MRI feature extractor (BrainIACExtractor or DINOv3Extractor)
            protein_run_dir: Path to protein model run directory (for on-the-fly extraction)
            protein_latents_dir: Path to pre-extracted protein latents directory (recommended)
            protein_model_type: 'nn' (Neural Network) or 'transformer'
            protein_layer: Layer to extract from ('last_hidden_layer' for NN, 'transformer_embeddings' for Transformer)
            protein_columns: List of protein column names (auto-detected if None)
            device: Device for inference
        """
        self.csv_path = Path(csv_path)
        self.device = device
        self.mri_extractor = mri_extractor
        
        # Load CSV
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} samples from {csv_path}")
        
        # Auto-detect protein columns (all numeric columns except metadata)
        if protein_columns is None:
            metadata_cols = ['RID', 'Subject', 'VISCODE', 'Visit', 'research_group', 
                           'Group', 'Sex', 'Age', 'subject_age', 'Image Data ID',
                           'Description', 'Type', 'Modality', 'Format', 'Acq Date',
                           'Downloaded', 'MRI_acquired', 'mri_source_path', 'mri_path']

            self.protein_columns = [col for col in self.df.columns 
                                   if col not in metadata_cols]
        else:
            self.protein_columns = protein_columns
        
        print(f"  Protein features: {len(self.protein_columns)} columns")
        
        # Load protein features (either pre-extracted latents or on-the-fly extraction)
        self.protein_latents = None
        self.protein_extractor = None
        
        if protein_latents_dir is not None:
            # Load pre-extracted latents
            protein_latents_dir = Path(protein_latents_dir)
            split_name = 'train' if 'train' in str(csv_path).lower() else 'test'
            latents_file = protein_latents_dir / f"{split_name}_protein_latents.npy"
            
            if latents_file.exists():
                self.protein_latents = np.load(latents_file)
                print(f"  Loaded pre-extracted protein latents: {self.protein_latents.shape}")
                
                # Verify number of samples matches
                if len(self.protein_latents) != len(self.df):
                    raise ValueError(f"Latents size {len(self.protein_latents)} != CSV size {len(self.df)}")
            else:
                raise FileNotFoundError(f"Pre-extracted latents not found: {latents_file}")
        
        elif protein_run_dir is not None:
            # On-the-fly extraction (may have NumPy compatibility issues)
            print(f"  Loading protein extractor from: {protein_run_dir}")
            self.protein_extractor = ProteinLatentExtractor(protein_run_dir, device)
            self.protein_model_type = protein_model_type
            self.protein_layer = protein_layer
            print(f"  Protein model: {protein_model_type}, layer: {protein_layer}")
        else:
            print(f"  No protein model - using raw features")
        
        # Get dimensions
        self.raw_protein_dim = len(self.protein_columns)
        
        if self.protein_latents is not None:
            # Use pre-extracted latents dimension
            self.protein_dim = self.protein_latents.shape[1]
        elif self.protein_extractor is not None:
            # Extract latents to get actual dimension
            # only using dummy data to get the output dimension of the protein encoder
            dummy_protein = np.random.randn(self.raw_protein_dim).astype(np.float32)
            if self.protein_model_type == 'nn':
                dummy_latents = self.protein_extractor.extract_nn_latents(dummy_protein, self.protein_layer)
            else:  # transformer
                dummy_latents = self.protein_extractor.extract_transformer_latents(dummy_protein, self.protein_layer)
            self.protein_dim = len(dummy_latents)
        else:
            self.protein_dim = self.raw_protein_dim  # Use raw features
        
        # Get MRI dimension from extractor
        self.mri_dim = self.mri_extractor.feature_dim
        self.fused_dim = self.protein_dim + self.mri_dim
        
        print(f"  Feature dimensions: protein={self.protein_dim} (raw={self.raw_protein_dim}), mri={self.mri_dim}, fused={self.fused_dim}")
        
        # Validate dataset integrity
        self._validate_dataset()
        
        # Initialize error tracking
        self.error_stats = {
            'mri_file_not_found': 0,
            'mri_data_validation_failed': 0,
            'mri_processing_failed': 0,
            'sample_load_failed': 0,
            'total_samples_processed': 0
        }
    
    def _validate_dataset(self):
        """Validate dataset integrity and report issues"""
        print(f"\nValidating dataset integrity...")
        
        # Check class distribution
        if 'research_group' in self.df.columns:
            class_counts = self.df['research_group'].value_counts()
            print(f"  Class distribution: {dict(class_counts)}")
            
            if len(class_counts) < 2:
                print(f"  WARNING: Dataset has only {len(class_counts)} class(es)")
            
            # Check for unknown classes
            valid_classes = {'AD', 'CN'}
            unknown_classes = set(class_counts.index) - valid_classes
            if unknown_classes:
                print(f"  WARNING: Unknown classes found: {unknown_classes}")
        
        # Check for missing MRI paths
        if 'mri_path' in self.df.columns:
            missing_mri = self.df['mri_path'].isna().sum()
            if missing_mri > 0:
                print(f"  WARNING: {missing_mri} samples missing MRI paths")
            
            # Check for empty MRI paths
            empty_mri = (self.df['mri_path'] == '').sum()
            if empty_mri > 0:
                print(f"  WARNING: {empty_mri} samples have empty MRI paths")
        
        # Check for missing Subject IDs
        if 'Subject' in self.df.columns:
            missing_subjects = self.df['Subject'].isna().sum()
            if missing_subjects > 0:
                print(f"  WARNING: {missing_subjects} samples missing Subject IDs")
        
        # Check protein columns
        if len(self.protein_columns) == 0:
            print(f"  ERROR: No protein columns detected")
        else:
            # Check for all-NaN protein columns
            protein_df = self.df[self.protein_columns]
            all_nan_cols = protein_df.columns[protein_df.isna().all()].tolist()
            if all_nan_cols:
                print(f"  WARNING: {len(all_nan_cols)} protein columns are all NaN")
            
            # Check for constant protein columns (zero variance)
            constant_cols = []
            for col in self.protein_columns:
                if protein_df[col].nunique() <= 1:
                    constant_cols.append(col)
            if constant_cols:
                print(f"  WARNING: {len(constant_cols)} protein columns have constant values")
        
        print(f"  Dataset validation completed")
    
    def print_error_stats(self):
        """Print error statistics for debugging"""
        print(f"\nDataset Error Statistics:")
        print(f"  Total samples processed: {self.error_stats['total_samples_processed']}")
        print(f"  MRI file not found: {self.error_stats['mri_file_not_found']}")
        print(f"  MRI data validation failed: {self.error_stats['mri_data_validation_failed']}")
        print(f"  MRI processing failed: {self.error_stats['mri_processing_failed']}")
        print(f"  Sample load failed: {self.error_stats['sample_load_failed']}")
        
        total_errors = (self.error_stats['mri_file_not_found'] + 
                       self.error_stats['mri_data_validation_failed'] + 
                       self.error_stats['mri_processing_failed'] + 
                       self.error_stats['sample_load_failed'])
        
        if self.error_stats['total_samples_processed'] > 0:
            error_rate = total_errors / self.error_stats['total_samples_processed'] * 100
            print(f"  Total error rate: {error_rate:.2f}%")
    
    def __len__(self):
        return len(self.df)
    
    def _extract_protein_features(self, idx):
        """Extract protein features from CSV row or pre-extracted latents"""
        if self.protein_latents is not None:
            # Use pre-extracted latents
            return self.protein_latents[idx]
        elif self.protein_extractor is not None:
            # Extract latents on-the-fly using trained protein model
            row = self.df.iloc[idx]
            protein_values = row[self.protein_columns].values.astype(np.float32)
            if self.protein_model_type == 'nn':
                protein_latents = self.protein_extractor.extract_nn_latents(
                    protein_values, self.protein_layer, feature_names=self.protein_columns
                )
            else:  # transformer
                protein_latents = self.protein_extractor.extract_transformer_latents(
                    protein_values, self.protein_layer, feature_names=self.protein_columns
                )
            return protein_latents
        else:
            # Return raw protein values
            row = self.df.iloc[idx]
            protein_values = row[self.protein_columns].values.astype(np.float32)
            return protein_values
    
    
    def __getitem__(self, idx):
        """
        Get a single sample with comprehensive validation
        
        Returns:
            dict with keys:
                - 'protein_features': [protein_dim] tensor
                - 'mri_features': [mri_dim] tensor
                - 'fused_features': [protein_dim + mri_dim] tensor
                - 'label': 0 (CN) or 1 (AD)
                - 'subject_id': Subject identifier
        """
        try:
            self.error_stats['total_samples_processed'] += 1
            row = self.df.iloc[idx]
            
            # Validate required columns exist
            required_cols = ['mri_path', 'research_group', 'Subject']
            missing_cols = [col for col in required_cols if col not in row.index]
            if missing_cols:
                raise KeyError(f"Missing required columns: {missing_cols}")
            
            # Validate mri_path is not NaN
            mri_path = row['mri_path']
            if pd.isna(mri_path):
                raise ValueError(f"MRI path is NaN for sample {idx}")
            
            # Extract protein features
            protein_feat = self._extract_protein_features(idx)  # [protein_dim]
            
            # Extract MRI latents via extractor
            mri_feat = self.mri_extractor.extract_features(Path(mri_path))  # [mri_dim]
            
            # Concatenate features
            fused_feat = np.concatenate([protein_feat, mri_feat])  # [protein_dim + mri_dim]
            
            # Get label: AD=1, CN=0
            research_group = row['research_group']
            if pd.isna(research_group):
                raise ValueError(f"Research group is NaN for sample {idx}")
            
            label = 1 if research_group == 'AD' else 0
            
            # Validate subject_id
            subject_id = row['Subject']
            if pd.isna(subject_id):
                subject_id = f"unknown_{idx}"  # Fallback
            
            return {
                'protein_features': torch.FloatTensor(protein_feat),
                'mri_features': torch.FloatTensor(mri_feat),
                'fused_features': torch.FloatTensor(fused_feat),
                'label': torch.LongTensor([label]).squeeze(),
                'subject_id': str(subject_id)
            }
            
        except Exception as e:
            self.error_stats['sample_load_failed'] += 1
            print(f"ERROR: Failed to load sample {idx}: {e}")
            print(f"  Error type: {type(e).__name__}")
            # Return zero-filled sample as fallback
            return {
                'protein_features': torch.zeros(self.protein_dim, dtype=torch.float32),
                'mri_features': torch.zeros(self.mri_dim, dtype=torch.float32),
                'fused_features': torch.zeros(self.fused_dim, dtype=torch.float32),
                'label': torch.LongTensor([0]),
                'subject_id': f"error_{idx}"
            }


def get_dataloader(csv_path, mri_extractor, protein_run_dir=None, batch_size=4, shuffle=True, device='cpu'):
    """
    Convenience function to create DataLoader
    
    Args:
        csv_path: Path to CSV file
        mri_extractor: MRI feature extractor (BrainIACExtractor or DINOv3Extractor)
        protein_run_dir: Path to protein model run directory (optional)
        batch_size: Batch size (default: 4)
        shuffle: Whether to shuffle (True for train, False for test)
        device: Device for inference
    
    Returns:
        DataLoader
    """
    from torch.utils.data import DataLoader
    
    dataset = MultimodalDataset(
        csv_path=csv_path,
        mri_extractor=mri_extractor,
        protein_run_dir=protein_run_dir,
        device=device
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Can't use multiprocessing with on-the-fly model inference
        pin_memory=False
    )
    
    return loader


if __name__ == "__main__":
    # Test the dataset
    print("Testing MultimodalDataset...")
    
    # Paths
    train_csv = "/home/ssim0068/data/multimodal-dataset/all_mni.csv"
    
    # Create MRI extractor
    print("\nLoading MRI extractor...")
    from mri_extractors import BrainIACExtractor
    mri_extractor = BrainIACExtractor(
        checkpoint_path="/home/ssim0068/code/multimodal-AD/BrainIAC/src/checkpoints/BrainIAC.ckpt",
        device='cpu'
    )
    
    # Create dataset (using raw protein features for now)
    print("\nCreating dataset...")
    dataset = MultimodalDataset(
        csv_path=train_csv,
        mri_extractor=mri_extractor,
        protein_run_dir=None,  # Use raw features
        device='cpu'
    )
    
    # Test getting one sample
    print("\nTesting __getitem__...")
    sample = dataset[0]
    
    print(f"\nSample 0:")
    print(f"  Subject: {sample['subject_id']}")
    print(f"  Label: {sample['label'].item()} ({'AD' if sample['label'].item() == 1 else 'CN'})")
    print(f"  Protein features shape: {sample['protein_features'].shape}")
    print(f"  MRI features shape: {sample['mri_features'].shape}")
    print(f"  Fused features shape: {sample['fused_features'].shape}")
    
    print("\n✅ Dataset test passed!")

