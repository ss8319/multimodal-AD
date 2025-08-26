import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import pydicom
from pathlib import Path
import os
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
import torchvision.models as models
from tqdm import tqdm
import warnings
import argparse
import sys
warnings.filterwarnings('ignore')

# Focal Loss for handling class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# BM-MAE imports (conditional)
try:
    # Add BM-MAE to Python path
    bm_mae_path = str(Path(__file__).parent.parent / "BM-MAE")
    if bm_mae_path not in sys.path:
        sys.path.insert(0, bm_mae_path)
    
    # Import BM-MAE modules
    from bmmae.model import ViTEncoder
    from bmmae.tokenizers import MRITokenizer
    BMMAE_AVAILABLE = True
    print("✅ BM-MAE modules loaded successfully")
except ImportError as e:
    print(f"⚠️ BM-MAE not available: {e}")
    print(f"   Expected path: {Path(__file__).parent.parent / 'BM-MAE'}")
    BMMAE_AVAILABLE = False
except Exception as e:
    print(f"⚠️ Error loading BM-MAE: {e}")
    BMMAE_AVAILABLE = False

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

class MRIDataset(Dataset):
    """Dataset class for loading 3D MRI data"""
    
    def __init__(self, dataframe, adni_base_path, transform=None, target_size=(128, 128, 128), augment=False):
        self.df = dataframe.dropna(subset=['dicom_folder_path']).reset_index(drop=True)
        self.adni_base_path = Path(adni_base_path)
        self.transform = transform
        self.target_size = target_size
        self.augment = augment
        
        # Label encoding: AD=1, CN=0
        self.df['label'] = (self.df['Group'] == 'AD').astype(int)
        
        print(f"📊 Dataset loaded: {len(self.df)} samples")
        print(f"   • AD: {(self.df['label'] == 1).sum()}")
        print(f"   • CN: {(self.df['label'] == 0).sum()}")
    
    def __len__(self):
        return len(self.df)
    
    def load_dicom_volume(self, dicom_folder_path):
        """Load DICOM files from folder and create complete 3D volume"""
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
        
        #print(f"   📁 Found {len(dicom_files)} DICOM files for {dicom_folder_path}")
        
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
        
        #print(f"   ✅ Loaded {len(dicom_data)} valid slices")
        
        # Sort by the chosen key and stack
        dicom_data.sort(key=lambda x: x[0])
        volume = np.stack([data[1] for data in dicom_data], axis=0)
        
        return volume.astype(np.float32)
    
    def preprocess_volume(self, volume, target_size=None):
        """Preprocess 3D volume: robust normalization and intelligent resizing"""
        if target_size is None:
            target_size = self.target_size
            
        #print(f"     🔄 Original volume shape: {volume.shape}")
        
        # 1. Robust intensity normalization (percentile-based)
        # Remove extreme outliers before normalization
        p1, p99 = np.percentile(volume, [1, 99])
        volume = np.clip(volume, p1, p99)
        
        # Z-score normalization (better than min-max for MRI)
        volume = (volume - volume.mean()) / (volume.std() + 1e-8)
        
        # 2. Intelligent resizing
        current_shape = volume.shape
        #print(f"     📆 Target shape: {target_size}")
        
        # Use scipy for better interpolation if available
        try:
            from scipy.ndimage import zoom
            
            # Calculate zoom factors
            zoom_factors = [
                target_size[i] / current_shape[i] for i in range(3)
            ]
            #print(f"     🔍 Zoom factors: {zoom_factors}")
            
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
        
        #print(f"     ✅ Final shape: {resized.shape}")
        return resized.astype(np.float32)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        try:
            #print(f"   📂 Loading {row['Subject']} ({row['Group']})...")
            
            # Load DICOM volume
            volume = self.load_dicom_volume(row['dicom_folder_path'])
            
            # Preprocess to standard size
            volume = self.preprocess_volume(volume)
            
            # Add channel dimension and convert to tensor
            volume = torch.FloatTensor(volume).unsqueeze(0)  # Shape: (1, D, H, W)
            
            # Apply data augmentation if enabled (only for training)
            if self.augment:
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
            
            if self.transform:
                volume = self.transform(volume)
            
            label = torch.LongTensor([row['label']])[0]
            
            #print(f"   ✅ Successfully loaded {row['Subject']} - Shape: {volume.shape}, Label: {label}")
            return volume, label
            
        except Exception as e:
            print(f"   ❌ Error loading {row['Subject']}: {e}")
            print(f"   🗗️ Returning dummy sample for {row['Subject']}")
            # Return a dummy sample
            dummy_volume = torch.zeros(1, *self.target_size)
            return dummy_volume, torch.LongTensor([0])[0]

# Model creation functions using PyTorch's built-in architectures
def create_resnet3d(num_classes=2):
    """Create 3D ResNet using torchvision's implementation adapted for 3D"""
    # Use ResNet18 as base and modify for 3D
    from torchvision.models.video import r3d_18
    
    # Load pretrained video ResNet (3D)
    model = r3d_18(pretrained=False)
    
    # Modify first conv for single channel input (grayscale MRI)
    model.stem[0] = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Modify final classifier
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

def create_alexnet3d(num_classes=2):
    """Create 3D AlexNet-style architecture"""
    class AlexNet3D(nn.Module):
        def __init__(self, num_classes=2):
            super(AlexNet3D, self).__init__()
            
            self.features = nn.Sequential(
                nn.Conv3d(1, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=3, stride=2),
                
                nn.Conv3d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=3, stride=2),
                
                nn.Conv3d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                
                nn.Conv3d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                
                nn.Conv3d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=3, stride=2),
            )
            
            self.avgpool = nn.AdaptiveAvgPool3d((6, 6, 6))
            
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
    
    return AlexNet3D(num_classes)

def create_convnext3d(num_classes=2):
    """Create 3D ConvNeXt using timm's implementation adapted for 3D"""
    try:
        import timm
        # Get 2D ConvNeXt and convert to 3D
        model_2d = timm.create_model('convnext_tiny', pretrained=False, num_classes=num_classes)
        
        # Convert to 3D manually (simplified approach)
        class ConvNeXt3D(nn.Module):
            def __init__(self, num_classes=2):
                super().__init__()
                
                self.stem = nn.Sequential(
                    nn.Conv3d(1, 96, kernel_size=4, stride=4),
                    nn.LayerNorm(96)  # ConvNeXt uses LayerNorm
                )
                
                # Simplified ConvNeXt blocks
                self.stages = nn.Sequential(
                    self._make_stage(96, 96, 3),
                    self._downsample(96, 192),
                    self._make_stage(192, 192, 3),
                    self._downsample(192, 384),
                    self._make_stage(384, 384, 9),
                    self._downsample(384, 768),
                    self._make_stage(768, 768, 3),
                )
                
                self.head = nn.Sequential(
                    nn.AdaptiveAvgPool3d(1),
                    nn.Flatten(),
                    nn.LayerNorm(768),
                    nn.Linear(768, num_classes)
                )
            
            def _make_stage(self, dim, out_dim, depth):
                layers = []
                for _ in range(depth):
                    layers.append(self._convnext_block(dim))
                return nn.Sequential(*layers)
            
            def _downsample(self, in_dim, out_dim):
                return nn.Sequential(
                    nn.LayerNorm(in_dim),
                    nn.Conv3d(in_dim, out_dim, kernel_size=2, stride=2)
                )
            
            def _convnext_block(self, dim):
                return nn.Sequential(
                    nn.Conv3d(dim, dim, 7, padding=3, groups=dim),
                    nn.LayerNorm(dim),
                    nn.Conv3d(dim, 4*dim, 1),
                    nn.GELU(),
                    nn.Conv3d(4*dim, dim, 1),
                )
            
            def forward(self, x):
                x = self.stem(x)
                x = self.stages(x)
                x = self.head(x)
                return x
        
        return ConvNeXt3D(num_classes)
    
    except ImportError:
        print("⚠️ timm not available, using simplified ConvNeXt3D")
        # Fallback to simplified version
        class SimpleConvNeXt3D(nn.Module):
            def __init__(self, num_classes=2):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv3d(1, 64, 7, stride=2, padding=3),
                    nn.BatchNorm3d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool3d(3, stride=2, padding=1),
                    
                    nn.Conv3d(64, 128, 3, padding=1),
                    nn.BatchNorm3d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool3d(2),
                    
                    nn.Conv3d(128, 256, 3, padding=1),
                    nn.BatchNorm3d(256),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool3d(1)
                )
                self.classifier = nn.Linear(256, num_classes)
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        return SimpleConvNeXt3D(num_classes)

def create_vit3d(num_classes=2):
    """Create 3D Vision Transformer using PyTorch's built-in components"""
    try:
        from torchvision.models import vit_b_16
        
        class ViT3D(nn.Module):
            def __init__(self, num_classes=2, patch_size=8, embed_dim=768):
                super().__init__()
                
                self.patch_size = patch_size
                self.embed_dim = embed_dim
                
                # 3D patch embedding
                self.patch_embed = nn.Conv3d(1, embed_dim, 
                                           kernel_size=patch_size, 
                                           stride=patch_size)
                
                # Use PyTorch's transformer
                self.transformer = nn.Transformer(
                    d_model=embed_dim,
                    nhead=12,
                    num_encoder_layers=12,
                    num_decoder_layers=0,  # Encoder only
                    dim_feedforward=embed_dim * 4,
                    batch_first=True
                )
                
                # Classification head
                self.norm = nn.LayerNorm(embed_dim)
                self.head = nn.Linear(embed_dim, num_classes)
                
                # Learnable embeddings
                self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
                self.pos_embed = nn.Parameter(torch.randn(1, 1000, embed_dim))
            
            def forward(self, x):
                B = x.shape[0]
                
                # Patch embedding
                x = self.patch_embed(x)  # (B, embed_dim, D', H', W')
                x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
                
                # Add positional embedding
                num_patches = x.shape[1]
                x = x + self.pos_embed[:, :num_patches]
                
                # Add class token
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat([cls_tokens, x], dim=1)
                
                # Transformer (encoder only)
                x = self.transformer.encoder(x)
                
                # Classification using class token
                x = self.norm(x[:, 0])
                x = self.head(x)
                
                return x
        
        return ViT3D(num_classes)
    
    except Exception as e:
        print(f"⚠️ Error creating ViT3D: {e}, using simplified version")
        
        # Simplified fallback
        class SimpleViT3D(nn.Module):
            def __init__(self, num_classes=2):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv3d(1, 128, 8, stride=8),  # Patch embedding
                    nn.BatchNorm3d(128),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool3d(1)
                )
                self.classifier = nn.Sequential(
                    nn.Linear(128, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                    nn.Linear(256, num_classes)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        return SimpleViT3D(num_classes)

# BM-MAE model classes
def create_bmmae_frozen(num_classes=2, pretrained_path='BM-MAE/pretrained_models/bmmae.pth'):
    """Create BM-MAE with frozen encoder + simple linear classifier"""
    if not BMMAE_AVAILABLE:
        raise ImportError("BM-MAE not available. Please install BM-MAE dependencies.")
    
    class BMMAEFrozen(nn.Module):
        def __init__(self, num_classes=2, pretrained_path=None):
            super().__init__()
            
            # Initialize BM-MAE encoder
            modalities = ['t1']  # Only T1 for our case
            tokenizers = {
                't1': MRITokenizer(
                    patch_size=(16, 16, 16),
                    img_size=(128, 128, 128),
                    hidden_size=768,
                )
            }
            
            self.encoder = ViTEncoder(
                modalities=modalities,
                tokenizers=tokenizers,
                cls_token=True
            )
            
            # Load pretrained weights if available
            if pretrained_path and os.path.exists(pretrained_path):
                try:
                    state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=False)
                    self.encoder.load_state_dict(state_dict, strict=False)
                    print(f"✅ Loaded BM-MAE pretrained weights from {pretrained_path}")
                except Exception as e:
                    print(f"⚠️ Failed to load pretrained weights: {e}")
            else:
                print(f"⚠️ Pretrained weights not found at {pretrained_path}")
            
            # Freeze encoder parameters
            for param in self.encoder.parameters():
                param.requires_grad = False
            
            # Regularized classifier head with dropout and batch norm
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),  # High dropout to prevent overfitting
                nn.Linear(768, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
            
            # Initialize weights properly
            for module in self.classifier.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.constant_(module.bias, 0)
            
            print("🔒 BM-MAE encoder frozen, training regularized classifier")
        
        def forward(self, x):
            # Prepare input for BM-MAE (expects dict with modality keys)
            inputs = {'t1': x}
            
            # Extract features using frozen encoder
            with torch.no_grad():
                features = self.encoder(inputs)  # Shape: [B, 1025, 768]
                # Use CLS token (first token) as global representation
                cls_features = features[:, 0, :]  # Shape: [B, 768]
                
                # Debug feature statistics (only print occasionally)
                if torch.rand(1).item() < 0.1:  # 10% chance to print
                    print(f"   🔍 BM-MAE features debug:")
                    print(f"      Features shape: {features.shape}")
                    print(f"      CLS features mean: {cls_features.mean():.3f}, std: {cls_features.std():.3f}")
                    print(f"      CLS features range: [{cls_features.min():.3f}, {cls_features.max():.3f}]")
            
            # Simple linear classification
            output = self.classifier(cls_features)
            return output
    
    return BMMAEFrozen(num_classes, pretrained_path)

def create_bmmae_finetuned(num_classes=2, pretrained_path='BM-MAE/pretrained_models/bmmae.pth'):
    """Create BM-MAE with fine-tuned encoder + simple linear classifier"""
    if not BMMAE_AVAILABLE:
        raise ImportError("BM-MAE not available. Please install BM-MAE dependencies.")
    
    class BMMAEFineTuned(nn.Module):
        def __init__(self, num_classes=2, pretrained_path=None):
            super().__init__()
            
            # Initialize BM-MAE encoder
            modalities = ['t1']  # Only T1 for our case
            tokenizers = {
                't1': MRITokenizer(
                    patch_size=(16, 16, 16),
                    img_size=(128, 128, 128),
                    hidden_size=768,
                )
            }
            
            self.encoder = ViTEncoder(
                modalities=modalities,
                tokenizers=tokenizers,
                cls_token=True
            )
            
            # Load pretrained weights if available
            if pretrained_path and os.path.exists(pretrained_path):
                try:
                    state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=False)
                    self.encoder.load_state_dict(state_dict, strict=False)
                    print(f"✅ Loaded BM-MAE pretrained weights from {pretrained_path}")
                except Exception as e:
                    print(f"⚠️ Failed to load pretrained weights: {e}")
            else:
                print(f"⚠️ Pretrained weights not found at {pretrained_path}")
            
            # Keep encoder trainable for fine-tuning
            # Use lower learning rate for encoder layers
            
            # Simple linear classifier head
            self.classifier = nn.Linear(768, num_classes)
            
            # Initialize weights properly
            nn.init.xavier_uniform_(self.classifier.weight)
            nn.init.constant_(self.classifier.bias, 0)
            
            print("🔄 BM-MAE encoder will be fine-tuned along with simple linear classifier")
        
        def forward(self, x):
            # Prepare input for BM-MAE (expects dict with modality keys)
            inputs = {'t1': x}
            
            # Extract features using trainable encoder
            features = self.encoder(inputs)  # Shape: [B, 1025, 768]
            # Use CLS token (first token) as global representation
            cls_features = features[:, 0, :]  # Shape: [B, 768]
            
            # Debug feature statistics (only print occasionally)
            if torch.rand(1).item() < 0.1:  # 10% chance to print
                print(f"   🔍 BM-MAE features debug:")
                print(f"      Features shape: {features.shape}")
                print(f"      CLS features mean: {cls_features.mean():.3f}, std: {cls_features.std():.3f}")
                print(f"      CLS features range: [{cls_features.min():.3f}, {cls_features.max():.3f}]")
            
            # Simple linear classification
            output = self.classifier(cls_features)
            return output
        
        def get_param_groups(self, lr_encoder=1e-5, lr_classifier=1e-3):
            """Get parameter groups with different learning rates"""
            return [
                {'params': self.encoder.parameters(), 'lr': lr_encoder},
                {'params': self.classifier.parameters(), 'lr': lr_classifier}
            ]
    
    return BMMAEFineTuned(num_classes, pretrained_path)

# Training and evaluation functions
def visualize_predictions(all_targets, all_preds, all_probs, model_name=''):
    """Visualize validation predictions vs targets"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Validation Results for {model_name}', fontsize=16, fontweight='bold')
    
    # 1. Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
    axes[0,0].set_title('Confusion Matrix')
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('Actual')
    axes[0,0].set_xticklabels(['CN (0)', 'AD (1)'])
    axes[0,0].set_yticklabels(['CN (0)', 'AD (1)'])
    
    # 2. Prediction Distribution by Class
    axes[0,1].hist([all_probs[i] for i, t in enumerate(all_targets) if t == 0], 
                    alpha=0.7, label='CN (0)', bins=20, color='blue')
    axes[0,1].hist([all_probs[i] for i, t in enumerate(all_targets) if t == 1], 
                    alpha=0.7, label='AD (1)', bins=20, color='red')
    axes[0,1].set_title('Prediction Probability Distribution by Class')
    axes[0,1].set_xlabel('Probability of AD (Class 1)')
    axes[0,1].set_ylabel('Count')
    axes[0,1].legend()
    axes[0,1].axvline(x=0.5, color='black', linestyle='--', alpha=0.7)
    
    # 3. ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    auc_score = roc_auc_score(all_targets, all_probs)
    
    axes[1,0].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {auc_score:.3f})')
    axes[1,0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1,0].set_xlim([0.0, 1.0])
    axes[1,0].set_ylim([0.0, 1.05])
    axes[1,0].set_xlabel('False Positive Rate')
    axes[1,0].set_ylabel('True Positive Rate')
    axes[1,0].set_title('ROC Curve')
    axes[1,0].legend(loc="lower right")
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Prediction vs Target Scatter
    axes[1,1].scatter(all_targets, all_probs, alpha=0.6, s=50)
    axes[1,1].set_xlabel('True Label')
    axes[1,1].set_ylabel('Predicted Probability of AD')
    axes[1,1].set_title('Predictions vs True Labels')
    axes[1,1].set_xticks([0, 1])
    axes[1,1].set_xticklabels(['CN (0)', 'AD (1)'])
    axes[1,1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Decision Boundary')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"validation_results_{model_name}_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Validation visualization saved as: {plot_filename}")
    
    plt.show()
    
    # Print detailed statistics
    print(f"\n📈 Detailed Validation Statistics for {model_name}:")
    print(f"   • Total samples: {len(all_targets)}")
    print(f"   • CN samples (0): {(np.array(all_targets) == 0).sum()}")
    print(f"   • AD samples (1): {(np.array(all_targets) == 1).sum()}")
    print(f"   • Correct predictions: {(np.array(all_targets) == np.array(all_preds)).sum()}")
    print(f"   • Incorrect predictions: {(np.array(all_targets) != np.array(all_preds)).sum()}")
    
    # Class-wise accuracy
    for class_label in [0, 1]:
        class_mask = np.array(all_targets) == class_label
        if class_mask.sum() > 0:
            class_acc = (np.array(all_preds)[class_mask] == class_label).sum() / class_mask.sum()
            class_name = "CN" if class_label == 0 else "AD"
            print(f"   • {class_name} (Class {class_label}) accuracy: {class_acc:.3f}")

def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc="Evaluating")):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            
            # Get predictions and probabilities
            probs = torch.softmax(output, dim=1)
            _, predicted = output.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
            
            # Debug first batch
            if batch_idx == 0:
                print(f"   🔍 First validation batch debug:")
                print(f"      Input shape: {data.shape}, range: [{data.min():.3f}, {data.max():.3f}]")
                print(f"      Output logits: {output}")
                print(f"      Probabilities: {probs}")
                print(f"      Targets: {target}")
                print(f"      Predictions: {predicted}")
                print(f"      Loss: {loss.item():.3f}")
    
    accuracy = accuracy_score(all_targets, all_preds)
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except:
        auc = 0.5
    
    # Print class distribution
    unique_targets, counts = np.unique(all_targets, return_counts=True)
    print(f"   📊 Validation class distribution: {dict(zip(unique_targets, counts))}")
    
    return running_loss / len(dataloader), accuracy * 100, auc, all_preds, all_probs, all_targets

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Debug first batch
        if batch_idx == 0:
            print(f"   🔍 First batch debug:")
            print(f"      Input shape: {data.shape}, range: [{data.min():.3f}, {data.max():.3f}]")
            print(f"      Output logits: {output}")
            print(f"      Targets: {target}")
            print(f"      Predictions: {predicted}")
            print(f"      Loss: {loss.item():.3f}")
        
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.3f}',
            'Acc': f'{100.*correct/total:.1f}%'
        })
    
    return running_loss / len(dataloader), 100. * correct / total

def train_model(model, train_loader, val_loader, num_epochs=20, lr=0.001, model_name=''):
    """Train a model with early stopping"""
    
    # Calculate class weights to handle imbalance
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())
    
    # Count class frequencies
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    class_weights = torch.FloatTensor([1.0 / count for count in counts]).to(device)
    
    print(f"📊 Class distribution: {dict(zip(unique_labels, counts))}")
    print(f"⚖️ Class weights: {class_weights.cpu().numpy()}")
    
    # Use focal loss to handle class imbalance (better than weighted CE)
    criterion = FocalLoss(alpha=1, gamma=2)
    print(f"🎯 Using Focal Loss (alpha=1, gamma=2) to handle class imbalance")
    
    # Handle different optimizer configurations for BM-MAE fine-tuned models
    if hasattr(model, 'get_param_groups'):
        # Fine-tuned BM-MAE with different learning rates
        param_groups = model.get_param_groups(lr_encoder=lr/100, lr_classifier=lr)  # Much lower LR for encoder
        optimizer = optim.Adam(param_groups, weight_decay=1e-4)
        print(f"🎯 Using different learning rates: Encoder={lr/100:.2e}, Classifier={lr:.2e}")
    else:
        # Standard optimizer for other models
        if 'frozen' in model_name.lower():
            # Higher weight decay for frozen models to prevent overfitting
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
            print(f"🎯 Using higher weight decay (1e-3) for frozen model to prevent overfitting")
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    best_val_acc = 0
    best_model_state = None
    train_history = {'loss': [], 'acc': []}
    val_history = {'loss': [], 'acc': [], 'auc': []}
    
    # Early stopping parameters
    patience = 8  # Reasonable patience for medical imaging
    patience_counter = 0
    min_epochs = 5  # Train for at least 5 epochs
    
    print(f"🛑 Early stopping: patience={patience}, min_epochs={min_epochs}")
    
    for epoch in range(num_epochs):
        print(f"\n📈 Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_auc, val_preds, val_probs, val_targets = evaluate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"🎯 New best validation accuracy: {best_val_acc:.1f}%")
        else:
            patience_counter += 1
            print(f"⏳ No improvement for {patience_counter} epochs")
        
        # Record history
        train_history['loss'].append(train_loss)
        train_history['acc'].append(train_acc)
        val_history['loss'].append(val_loss)
        val_history['acc'].append(val_acc)
        val_history['auc'].append(val_auc)
        
        print(f"Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.1f}%")
        print(f"Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.1f}%, Val AUC: {val_auc:.3f}")
        
        # Visualize predictions on the last epoch or when we get the best validation accuracy
        if args.visualize and (epoch == num_epochs - 1 or val_acc == best_val_acc):
            print(f"\n🎨 Visualizing validation predictions for {model_name}...")
            visualize_predictions(val_targets, val_preds, val_probs, model_name)
        
        # Early stopping check (but ensure minimum epochs)
        if epoch >= min_epochs and patience_counter >= patience:
            print(f"🛑 Early stopping triggered at epoch {epoch+1}")
            print(f"   Best validation accuracy: {best_val_acc:.1f}%")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"✅ Loaded best model with validation accuracy: {best_val_acc:.1f}%")
    else:
        print("⚠️ No best model state found, using current model")
    
    return model, train_history, val_history, best_val_acc

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='3D MRI Classification with Multiple Baseline Methods')
    
    parser.add_argument('--methods', nargs='+', 
                       choices=['resnet3d', 'alexnet3d', 'convnext3d', 'vit3d', 'bmmae_frozen', 'bmmae_finetuned', 'all'],
                       default=['bmmae_finetuned'],
                       help='Choose which methods to train (default: all)')
    
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size (default: 16)')
    
    parser.add_argument('--target_size', type=int, nargs=3, default=[128, 128, 128],
                       help='Target volume size for all models (default: 128 128 128)')
    
    parser.add_argument('--bmmae_pretrained', type=str, 
                       default='BM-MAE/pretrained_models/bmmae.pth',
                       help='Path to BM-MAE pretrained weights')
    
    parser.add_argument('--output_dir', type=str,
                       default=r"D:\ADNI\AD_CN\proteomics\Biomarkers Consortium Plasma Proteomics MRM\MRI\splits",
                       help='Output directory for results')
    
    parser.add_argument('--visualize', action='store_true',
                       help='Enable validation prediction visualization')
    
    return parser.parse_args()

# Main training script
def main():
    args = parse_arguments()
    
    print("🧠 3D MRI Classification Baseline Training")
    print("=" * 50)
    print(f"📋 Selected methods: {args.methods}")
    print(f"🔧 Training epochs: {args.epochs}")
    print(f"⚡ Learning rate: {args.lr}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"📏 Target size: {args.target_size}")
    print(f"🎨 Visualization: {'Enabled' if args.visualize else 'Disabled'}")
    print("=" * 50)
    
    # Load data splits
    splits_folder = Path(args.output_dir)
    adni_base_path = r"D:\ADNI\AD_CN\proteomics\Biomarkers Consortium Plasma Proteomics MRM\MRI\ADNI"
    
    train_df = pd.read_csv(splits_folder / "train_split.csv")
    test_df = pd.read_csv(splits_folder / "test_split.csv")
    
    print(f"📊 Data loaded:")
    print(f"   • Training samples: {len(train_df)}")
    print(f"   • Test samples: {len(test_df)}")
    
    # Create datasets
    target_size = tuple(args.target_size)
    
    # Split training data for validation (80/20)
    train_size = int(0.8 * len(train_df))
    train_subset = train_df.iloc[:train_size]
    val_subset = train_df.iloc[train_size:]
    
    # Create datasets with augmentation for training
    train_dataset = MRIDataset(train_subset, adni_base_path, target_size=target_size, augment=True)
    val_dataset = MRIDataset(val_subset, adni_base_path, target_size=target_size, augment=False)
    test_dataset = MRIDataset(test_df, adni_base_path, target_size=target_size, augment=False)
    
    # Define available models
    available_models = {
        'resnet3d': lambda: create_resnet3d(num_classes=2),
        'alexnet3d': lambda: create_alexnet3d(num_classes=2),
        'convnext3d': lambda: create_convnext3d(num_classes=2),
        'vit3d': lambda: create_vit3d(num_classes=2),
        'bmmae_frozen': lambda: create_bmmae_frozen(num_classes=2, pretrained_path=args.bmmae_pretrained),
        'bmmae_finetuned': lambda: create_bmmae_finetuned(num_classes=2, pretrained_path=args.bmmae_pretrained)
    }
    
    # Determine which models to train
    if 'all' in args.methods:
        selected_methods = list(available_models.keys())
    else:
        selected_methods = args.methods
    
    # Filter out BM-MAE models if not available
    if not BMMAE_AVAILABLE:
        selected_methods = [m for m in selected_methods if not m.startswith('bmmae')]
        if len([m for m in args.methods if m.startswith('bmmae')]) > 0:
            print("⚠️ BM-MAE methods requested but BM-MAE not available, skipping...")
    
    print(f"🎯 Training methods: {selected_methods}")
    
    results = {}
    
    # Train each selected model
    for method_name in selected_methods:
        print(f"\n🚀 Training {method_name.upper()}")
        print("-" * 50)
        
        try:
            # Create model
            model = available_models[method_name]()
            model = model.to(device)
            
            # All models now use the same target size for fair comparison
            print(f"📦 Creating datasets with standardized size ({target_size})...")
            
            # Create data loaders - all models use same preprocessing now
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
            
            # Train the model
            trained_model, train_hist, val_hist, best_val_acc = train_model(
                model, train_loader, val_loader, 
                num_epochs=args.epochs, lr=args.lr, model_name=method_name
            )
            
            # Test evaluation
            print(f"\n🔍 Testing {method_name.upper()}...")
            test_loss, test_acc, test_auc = evaluate(trained_model, test_loader, nn.CrossEntropyLoss(), device)
            
            results[method_name] = {
                'best_val_acc': best_val_acc,
                'test_acc': test_acc,
                'test_auc': test_auc,
                'train_history': train_hist,
                'val_history': val_hist
            }
            
            print(f"✅ {method_name.upper()} Results:")
            print(f"   • Best Val Acc: {best_val_acc:.1f}%")
            print(f"   • Test Acc: {test_acc:.1f}%")
            print(f"   • Test AUC: {test_auc:.3f}")
            
            # Save model
            torch.save(trained_model.state_dict(), splits_folder / f"{method_name}_best.pth")
            
        except Exception as e:
            print(f"❌ Failed to train {method_name}: {e}")
            print(f"⏭️ Skipping {method_name}...")
            continue
    
    # Summary results
    if results:
        print(f"\n📊 FINAL RESULTS SUMMARY")
        print("=" * 60)
        results_df = pd.DataFrame({
            model: {
                'Val_Acc(%)': f"{res['best_val_acc']:.1f}",
                'Test_Acc(%)': f"{res['test_acc']:.1f}", 
                'Test_AUC': f"{res['test_auc']:.3f}"
            }
            for model, res in results.items()
        }).T
        
        print(results_df)
        
        # Save results
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        results_df.to_csv(splits_folder / f"baseline_results_{timestamp}.csv")
        
        print(f"\n💾 All models and results saved to: {splits_folder}")
        print("🎉 Baseline training complete!")
    else:
        print("\n❌ No models were successfully trained!")
        return 1
    
    return 0

def print_usage_examples():
    """Print usage examples for the command line interface"""
    print("\n🔧 USAGE EXAMPLES:")
    print("=" * 50)
    print("# Train all available methods (default):")
    print("python colab/image_baseline.py")
    print()
    print("# Train only ResNet3D and ViT3D:")
    print("python colab/image_baseline.py --methods resnet3d vit3d")
    print()
    print("# Train only BM-MAE methods:")
    print("python colab/image_baseline.py --methods bmmae_frozen bmmae_finetuned")
    print()
    print("# Custom training parameters:")
    print("python colab/image_baseline.py --methods resnet3d --epochs 20 --lr 0.0005 --batch_size 4")
    print()
    print("# Different target size for all models:")
    print("python colab/image_baseline.py --methods alexnet3d --target_size 96 96 96")
    print()
    print("# Enable validation visualization:")
    print("python colab/image_baseline.py --methods bmmae_frozen --visualize")
    print("=" * 50)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        print("🧠 3D MRI Classification Baseline Training")
        print("Supports multiple baseline methods including traditional CNNs, ViT, and BM-MAE")
        print_usage_examples()
    else:
        sys.exit(main())
