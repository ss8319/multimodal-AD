"""
Model architectures for 3D MRI classification.

This module provides various deep learning architectures including:
- 3D CNNs: ResNet3D, AlexNet3D, ConvNeXt3D
- Vision Transformers: ViT3D, BM-MAE variants
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional
import sys
from pathlib import Path
import os

# BM-MAE imports (conditional)
try:
    # Add BM-MAE to Python path
    bm_mae_path = str(Path(__file__).parent.parent.parent / "BM-MAE")
    if bm_mae_path not in sys.path:
        sys.path.insert(0, bm_mae_path)
    
    # Import BM-MAE modules
    from bmmae.model import ViTEncoder
    from bmmae.tokenizers import MRITokenizer
    BMMAE_AVAILABLE = True
    print("✅ BM-MAE modules loaded successfully")
except ImportError as e:
    print(f"⚠️ BM-MAE not available: {e}")
    print(f"   Expected path: {Path(__file__).parent.parent.parent / 'BM-MAE'}")
    BMMAE_AVAILABLE = False
except Exception as e:
    print(f"⚠️ Error loading BM-MAE: {e}")
    BMMAE_AVAILABLE = False

# ============================================================================
# 3D CNN Models
# ============================================================================

def create_resnet3d(num_classes: int = 2) -> nn.Module:
    """
    Create 3D ResNet using torchvision's implementation adapted for 3D.
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        3D ResNet model
    """
    try:
        from torchvision.models.video import r3d_18
        
        # Load pretrained video ResNet (3D)
        model = r3d_18(pretrained=False)
        
        # Modify first conv for single channel input (grayscale MRI)
        model.stem[0] = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify final classifier
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        print("✅ Created 3D ResNet model")
        return model
        
    except ImportError:
        print("⚠️ torchvision video models not available, using simplified ResNet3D")
        return _create_simple_resnet3d(num_classes)

def _create_simple_resnet3d(num_classes: int = 2) -> nn.Module:
    """Fallback simple 3D ResNet implementation."""
    class SimpleResNet3D(nn.Module):
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
    
    return SimpleResNet3D(num_classes)

def create_alexnet3d(num_classes: int = 2) -> nn.Module:
    """
    Create 3D AlexNet-style architecture.
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        3D AlexNet model
    """
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
    
    print("✅ Created 3D AlexNet model")
    return AlexNet3D(num_classes)

def create_convnext3d(num_classes: int = 2) -> nn.Module:
    """
    Create 3D ConvNeXt architecture.
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        3D ConvNeXt model
    """
    try:
        import timm
        # Get 2D ConvNeXt and convert to 3D (simplified approach)
        print("✅ Created 3D ConvNeXt model (timm-based)")
        
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

# ============================================================================
# Vision Transformer Models
# ============================================================================

def create_vit3d(num_classes: int = 2) -> nn.Module:
    """
    Create 3D Vision Transformer using PyTorch's built-in components.
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        3D Vision Transformer model
    """
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
        
        print("✅ Created 3D Vision Transformer model")
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

# ============================================================================
# BM-MAE Models
# ============================================================================

def create_bmmae_frozen(num_classes: int = 2, pretrained_path: str = 'BM-MAE/pretrained_models/bmmae.pth') -> nn.Module:
    """
    Create BM-MAE with frozen encoder + regularized MLP classifier.
    
    Args:
        num_classes: Number of output classes
        pretrained_path: Path to pretrained BM-MAE weights
        
    Returns:
        BM-MAE frozen model
    """
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
            
            # Regularized classifier head with LayerNorm (more stable for small batches)
            self.classifier = nn.Sequential(
                nn.LayerNorm(768),
                nn.Dropout(0.5),
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
            
            # Initialize weights properly; bias for AD class set later
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
            
            # Simple linear classification
            output = self.classifier(cls_features)
            return output
    
    print("✅ Created BM-MAE frozen model")
    return BMMAEFrozen(num_classes, pretrained_path)

def create_bmmae_finetuned(num_classes: int = 2, pretrained_path: str = 'BM-MAE/pretrained_models/bmmae.pth') -> nn.Module:
    """
    Create BM-MAE with fine-tuned encoder + simple linear classifier.
    
    Args:
        num_classes: Number of output classes
        pretrained_path: Path to pretrained BM-MAE weights
        
    Returns:
        BM-MAE fine-tuned model
    """
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
            
            # Simple linear classification
            output = self.classifier(cls_features)
            return output
        
        def get_param_groups(self, lr_encoder=1e-5, lr_classifier=1e-3):
            """Get parameter groups with different learning rates"""
            return [
                {'params': self.encoder.parameters(), 'lr': lr_encoder},
                {'params': self.classifier.parameters(), 'lr': lr_classifier}
            ]
    
    print("✅ Created BM-MAE fine-tuned model")
    return BMMAEFineTuned(num_classes, pretrained_path)

# ============================================================================
# Model Factory
# ============================================================================

def create_model(model_name: str, num_classes: int = 2, **kwargs) -> nn.Module:
    """
    Factory function to create models by name.
    
    Args:
        model_name: Name of the model to create
        num_classes: Number of output classes
        **kwargs: Additional arguments for specific models
        
    Returns:
        PyTorch model
    """
    model_creators = {
        'resnet3d': create_resnet3d,
        'alexnet3d': create_alexnet3d,
        'convnext3d': create_convnext3d,
        'vit3d': create_vit3d,
        'bmmae_frozen': create_bmmae_frozen,
        'bmmae_finetuned': create_bmmae_finetuned
    }
    
    if model_name not in model_creators:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_creators.keys())}")
    
    creator = model_creators[model_name]
    
    # Handle special cases
    if 'bmmae' in model_name:
        pretrained_path = kwargs.get('pretrained_path', 'BM-MAE/pretrained_models/bmmae.pth')
        return creator(num_classes, pretrained_path)
    else:
        return creator(num_classes)

def get_available_models() -> list:
    """Get list of available model names."""
    return ['resnet3d', 'alexnet3d', 'convnext3d', 'vit3d', 'bmmae_frozen', 'bmmae_finetuned']

def is_bmmae_available() -> bool:
    """Check if BM-MAE is available."""
    return BMMAE_AVAILABLE
