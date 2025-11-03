"""
MRI Feature Extractors for Multimodal Fusion

Provides abstract interface and concrete implementations for:
- BrainIAC: 3D CNN-based feature extraction (768-dim)
- DINOv3: 2D ViT with slice aggregation (384-dim for ViT-S, 768-dim for ViT-B)
"""

from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import sys


class BaseMRIExtractor(ABC):
    """Abstract base class for MRI feature extractors"""
    
    @abstractmethod
    def extract_features(self, mri_path: Path) -> np.ndarray:
        """
        Extract features from MRI volume
        
        Args:
            mri_path: Path to NIfTI file
        
        Returns:
            Feature vector as numpy array
        """
        pass
    
    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Return feature dimension"""
        pass


class BrainIACExtractor(BaseMRIExtractor):
    """
    BrainIAC MRI feature extractor
    
    Uses 3D CNN trained for brain age prediction
    Output: 768-dimensional features
    """
    
    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        """
        Args:
            checkpoint_path: Path to BrainIAC checkpoint
            device: Device for inference ('cpu' or 'cuda')
        """
        # Import BrainIAC modules
        sys.path.append(str(Path(__file__).parent.parent / "mri" / "BrainIAC" / "src"))
        from load_brainiac import load_brainiac
        
        self.device = device
        self.model = load_brainiac(checkpoint_path, device)
        self.model.eval()
        self._feature_dim = 768
        
        print(f"  Loaded BrainIAC model from {checkpoint_path}")
        print(f"  Output dimension: {self._feature_dim}")
    
    @property
    def feature_dim(self) -> int:
        return self._feature_dim
    
    def extract_features(self, mri_path: Path) -> np.ndarray:
        """
        Extract features using BrainIAC model
        
        Preprocessing:
        - Resize to (96, 96, 96)
        - Normalize intensity (nonzero, channel-wise)
        
        Args:
            mri_path: Path to NIfTI file
        
        Returns:
            Features [768]
        """
        from monai.transforms import (
            Compose, LoadImaged, EnsureChannelFirstd,
            Resized, NormalizeIntensityd, EnsureTyped
        )
        
        # BrainIAC preprocessing pipeline
        transform = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Resized(keys=["image"], spatial_size=(96, 96, 96), mode="trilinear"),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image"])
        ])
        
        # Load and preprocess
        sample = {"image": str(mri_path)}
        sample = transform(sample)
        image = sample["image"].unsqueeze(0).to(self.device)  # [1, 1, 96, 96, 96]
        
        # Validate input
        if image.shape != (1, 1, 96, 96, 96):
            raise ValueError(f"Unexpected image shape {image.shape}, expected (1, 1, 96, 96, 96)")
        
        if torch.isnan(image).any():
            raise ValueError(f"NaN values detected in MRI image: {mri_path}")
        
        if torch.isinf(image).any():
            raise ValueError(f"Infinite values detected in MRI image: {mri_path}")
        
        # Extract features
        with torch.no_grad():
            features = self.model(image)  # [1, 768]
            features = features.cpu().numpy().flatten()  # [768]
        
        # Validate output
        if len(features) != self._feature_dim:
            raise ValueError(f"Unexpected feature dimension {len(features)}, expected {self._feature_dim}")
        
        if np.isnan(features).any():
            raise ValueError(f"NaN values in extracted features: {mri_path}")
        
        if np.isinf(features).any():
            raise ValueError(f"Infinite values in extracted features: {mri_path}")
        
        return features


class DINOv3Extractor(BaseMRIExtractor):
    """
    DINOv3 MRI feature extractor using DINOv3 dataset infrastructure
    
    Reuses DINOv3 modules:
    - SliceAggregationDataset: for 3D volume loading, slicing, and 2D transforms
    - extract_features_with_slice_aggregation: for feature extraction
    
    Output dimensions:
    - dinov3_vits16: 384-dim
    - dinov3_vitb16: 768-dim
    - dinov3_vitl16: 1024-dim
    - dinov3_vitg14: 1536-dim
    """
    
    def __init__(
        self,
        model_name: str = 'dinov3_vits16',
        weights_path: str = None,
        hub_repo_dir: str = None,
        device: str = 'cuda',
        slice_axis: int = 0,
        stride: int = 2,
        image_size: int = 224
    ):
        """
        Args:
            model_name: DINOv3 model name (dinov3_vits16, dinov3_vitb16, etc.)
            weights_path: Path to pretrained weights
            hub_repo_dir: Path to DINOv3 repository (for model loading)
            device: Device for inference ('cpu' or 'cuda')
            slice_axis: Axis for slice extraction (0=sagittal, 1=coronal, 2=axial)
            stride: Extract every Nth slice
            image_size: 2D image size for DINOv3 (224 for standard models)
        """
        self.device = device
        self.slice_axis = slice_axis
        self.stride = stride
        self.image_size = image_size
        
        # Import DINOv3 modules
        if hub_repo_dir:
            sys.path.insert(0, str(Path(hub_repo_dir).resolve()))
        
        from dinov3.eval.setup import load_model_and_context, ModelConfig
        from dinov3.data.transforms import make_classification_eval_transform
        from dinov3.data.datasets.adni_3d_aggregation import SliceAggregationDataset
        
        print(f"  Loading DINOv3 model: {model_name}")
        
        # Load DINOv3 backbone
        model_config = ModelConfig(
            hub_repo_dir=hub_repo_dir or str(Path(__file__).parent.parent / "mri" / "dinov3"),
            hub_model=model_name,
            pretrained_weights=weights_path
        )
        
        self.model, context = load_model_and_context(model_config, output_dir=None)
        self.model.eval()
        self.model.to(device)
        
        print(f"  DINOv3 backbone loaded")
        
        # Setup 2D transform (from DINOv3)
        self._transform_2d = make_classification_eval_transform(
            resize_size=image_size,
            crop_size=image_size
        )
        
        # Store SliceAggregationDataset class for reuse
        self._SliceAggregationDataset = SliceAggregationDataset
        
        # Auto-detect feature dimension from model
        print(f"  Detecting feature dimension...")
        dummy = torch.randn(1, 3, image_size, image_size).to(device)
        with torch.no_grad():
            dummy_feat = self.model(dummy)
        self._feature_dim = dummy_feat.shape[1]
        
        print(f"  DINOv3 configuration:")
        print(f"    Model: {model_name}")
        print(f"    Weights: {Path(weights_path).name if weights_path else 'default'}")
        print(f"    Slice axis: {slice_axis} ({'sagittal' if slice_axis == 0 else 'coronal' if slice_axis == 1 else 'axial'})")
        print(f"    Stride: {stride}")
        print(f"    2D image size: {image_size}")
        print(f"    Feature dimension: {self._feature_dim}")
        print(f"    Device: {device}")
    
    @property
    def feature_dim(self) -> int:
        return self._feature_dim
    
    def extract_features(self, mri_path: Path) -> np.ndarray:
        """
        Extract features using DINOv3 SliceAggregationDataset infrastructure
        
        This method reuses DINOv3's slice aggregation logic instead of reimplementing it.
        
        Args:
            mri_path: Path to NIfTI file
        
        Returns:
            Aggregated features [384 or 768 depending on model]
        """
        # Create a minimal base dataset for a single image
        class SingleImageDataset:
            """Minimal dataset that provides a single image path and dummy label"""
            def __init__(self, image_path, root=""):
                self.image_path = image_path
                self.root = root
            
            def __len__(self):
                return 1
            
            def get_image_relpath(self, index):
                return str(self.image_path)
            
            def get_target(self, index):
                return 0  # Dummy label (not used for feature extraction)
        
        # Create base dataset with the single image
        base_dataset = SingleImageDataset(mri_path, root="")
        
        # Wrap with SliceAggregationDataset (handles 3D loading, slicing, transforms)
        slice_dataset = self._SliceAggregationDataset(
            base_dataset=base_dataset,
            slice_axis=self.slice_axis,
            stride=self.stride,
            transform=self._transform_2d,
            target_transform=None
        )
        
        # Extract slices using the dataset (handles all preprocessing)
        slices, _ = slice_dataset[0]  # Returns (list of slice tensors, label)
        
        if len(slices) == 0:
            raise ValueError(f"No slices extracted from volume {mri_path}")
        
        # Stack and forward through DINOv3 backbone
        slice_batch = torch.stack(slices).to(self.device)  # [N, 3, 224, 224]
        
        with torch.no_grad():
            slice_features = self.model(slice_batch)  # [N, feature_dim]
        
        # Aggregate with mean pooling
        volume_feature = slice_features.mean(dim=0)  # [feature_dim]
        
        # Validate output
        if volume_feature.shape[0] != self._feature_dim:
            raise ValueError(f"Unexpected feature dimension {volume_feature.shape[0]}, expected {self._feature_dim}")
        
        return volume_feature.cpu().numpy()


def test_extractors():
    """Test extractors with sample data"""
    print("Testing MRI Extractors...")
    
    # Test BrainIAC
    print("\n" + "="*60)
    print("Testing BrainIACExtractor")
    print("="*60)
    try:
        brainiac = BrainIACExtractor(
            checkpoint_path="/home/ssim0068/code/multimodal-AD/BrainIAC/src/checkpoints/BrainIAC.ckpt",
            device='cpu'
        )
        print(f"✓ BrainIAC extractor initialized")
        print(f"  Feature dimension: {brainiac.feature_dim}")
    except Exception as e:
        print(f"✗ BrainIAC initialization failed: {e}")
    
    # Test DINOv3
    print("\n" + "="*60)
    print("Testing DINOv3Extractor")
    print("="*60)
    try:
        dinov3 = DINOv3Extractor(
            model_name='dinov3_vits16',
            weights_path='/home/ssim0068/multimodal-AD/src/mri/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
            hub_repo_dir='/home/ssim0068/multimodal-AD/src/mri/dinov3',
            device='cuda' if torch.cuda.is_available() else 'cpu',
            slice_axis=0,
            stride=2,
            image_size=224
        )
        print(f"✓ DINOv3 extractor initialized")
        print(f"  Feature dimension: {dinov3.feature_dim}")
    except Exception as e:
        print(f"✗ DINOv3 initialization failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✓ Extractor tests completed")


if __name__ == "__main__":
    test_extractors()

