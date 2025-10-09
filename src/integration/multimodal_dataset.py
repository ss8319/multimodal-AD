import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add paths for protein and MRI modules
sys.path.append(str(Path(__file__).parent.parent / "mri" / "BrainIAC" / "src"))
sys.path.append(str(Path(__file__).parent.parent / "protein"))

# Import from correct modules
from dataset import BrainAgeDataset, get_validation_transform
from load_brainiac import load_brainiac
from protein_extractor import ProteinLatentExtractor


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
        brainiac_model=None,
        brainiac_checkpoint=None,
        protein_run_dir=None,
        protein_latents_dir=None,
        protein_model_type='mlp',
        protein_layer='hidden_layer_2',
        protein_columns=None,
        device='cpu'
    ):
        """
        Args:
            csv_path: Path to train.csv or test.csv
            brainiac_model: Pre-loaded BrainIAC model (recommended for efficiency)
            brainiac_checkpoint: Path to BrainIAC checkpoint (if model not provided)
            protein_run_dir: Path to protein model run directory (for on-the-fly extraction)
            protein_latents_dir: Path to pre-extracted protein latents directory (recommended)
            protein_model_type: 'mlp' or 'transformer'
            protein_layer: Layer to extract from ('hidden_layer_2' for MLP, 'transformer_embeddings' for Transformer)
            protein_columns: List of protein column names (auto-detected if None)
            device: Device for inference
        """
        self.csv_path = Path(csv_path)
        self.device = device
        
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
        
        # Load or store BrainIAC model
        if brainiac_model is not None:
            self.mri_model = brainiac_model
            print(f"  Using provided BrainIAC model")
        elif brainiac_checkpoint is not None:
            print(f"  Loading BrainIAC from: {brainiac_checkpoint}")
            self.mri_model = load_brainiac(brainiac_checkpoint, device)
        else:
            raise ValueError("Must provide either brainiac_model or brainiac_checkpoint")
        
        self.mri_model.eval()
        
        # Get dimensions
        self.raw_protein_dim = len(self.protein_columns)
        
        if self.protein_latents is not None:
            # Use pre-extracted latents dimension
            self.protein_dim = self.protein_latents.shape[1]
        elif self.protein_extractor is not None:
            # Extract latents to get actual dimension
            # only using dummy data to get the output dimension of the protein encoder
            dummy_protein = np.random.randn(self.raw_protein_dim).astype(np.float32)
            if self.protein_model_type == 'mlp':
                dummy_latents = self.protein_extractor.extract_mlp_latents(dummy_protein, self.protein_layer)
            else:  # transformer
                dummy_latents = self.protein_extractor.extract_transformer_latents(dummy_protein, self.protein_layer)
            self.protein_dim = len(dummy_latents)
        else:
            self.protein_dim = self.raw_protein_dim  # Use raw features
        
        self.mri_dim = 768  # BrainIAC output dimension
        self.fused_dim = self.protein_dim + self.mri_dim
        
        print(f"  Feature dimensions: protein={self.protein_dim} (raw={self.raw_protein_dim}), mri={self.mri_dim}, fused={self.fused_dim}")
    
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
            if self.protein_model_type == 'mlp':
                protein_latents = self.protein_extractor.extract_mlp_latents(
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
    
    def _extract_mri_latents(self, mri_path):
        """
        Extract MRI latents using BrainIAC model
        This is done on-the-fly during training
        """
        import nibabel as nib
        from monai.transforms import (
            Compose, LoadImaged, EnsureChannelFirstd, 
            Resized, NormalizeIntensityd, EnsureTyped
        )
        
        # Load and preprocess MRI image
        # Using MONAI transforms similar to BrainIAC dataset
        transform = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Resized(keys=["image"], spatial_size=(96, 96, 96), mode="trilinear"),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image"])
        ])
        
        # Load image
        sample = {"image": str(mri_path)}
        sample = transform(sample)
        image = sample["image"].unsqueeze(0).to(self.device)  # [1, 1, 96, 96, 96]
        
        # Extract features
        with torch.no_grad():
            features = self.mri_model(image)  # [1, 768]
            features = features.cpu().numpy().flatten()  # [768]
        
        return features
    
    def __getitem__(self, idx):
        """
        Get a single sample
        
        Returns:
            dict with keys:
                - 'protein_features': [protein_dim] tensor
                - 'mri_features': [mri_dim] tensor
                - 'fused_features': [protein_dim + mri_dim] tensor
                - 'label': 0 (CN) or 1 (AD)
                - 'subject_id': Subject identifier
        """
        row = self.df.iloc[idx]
        
        # Extract protein features
        protein_feat = self._extract_protein_features(idx)  # [protein_dim]
        
        # Extract MRI latents on-the-fly
        mri_path = row['mri_path']
        mri_feat = self._extract_mri_latents(mri_path)  # [768]
        
        # Concatenate features
        fused_feat = np.concatenate([protein_feat, mri_feat])  # [protein_dim + 768]
        
        # Get label: AD=1, CN=0
        label = 1 if row['research_group'] == 'AD' else 0
        
        return {
            'protein_features': torch.FloatTensor(protein_feat),
            'mri_features': torch.FloatTensor(mri_feat),
            'fused_features': torch.FloatTensor(fused_feat),
            'label': torch.LongTensor([label]).squeeze(),
            'subject_id': row['Subject']
        }


def get_dataloader(csv_path, brainiac_model, protein_run_dir=None, batch_size=4, shuffle=True, device='cpu'):
    """
    Convenience function to create DataLoader
    
    Args:
        csv_path: Path to train.csv or test.csv
        brainiac_model: Pre-loaded BrainIAC model
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
        brainiac_model=brainiac_model,
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
    train_csv = "/home/ssim0068/data/multimodal-dataset/train.csv"
    brainiac_ckpt = "/home/ssim0068/code/multimodal-AD/BrainIAC/src/checkpoints/BrainIAC.ckpt"
    
    # Load BrainIAC model once
    print("\nLoading BrainIAC model...")
    mri_model = load_brainiac(brainiac_ckpt, 'cpu')
    
    # Create dataset (using raw protein features for now)
    print("\nCreating dataset...")
    dataset = MultimodalDataset(
        csv_path=train_csv,
        brainiac_model=mri_model,
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

