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
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

class MRIDataset(Dataset):
    """Dataset class for loading 3D MRI data"""
    
    def __init__(self, dataframe, adni_base_path, transform=None, target_size=(64, 64, 64)):
        self.df = dataframe.dropna(subset=['dicom_folder_path']).reset_index(drop=True)
        self.adni_base_path = Path(adni_base_path)
        self.transform = transform
        self.target_size = target_size
        
        # Label encoding: AD=1, CN=0
        self.df['label'] = (self.df['Group'] == 'AD').astype(int)
        
        print(f"📊 Dataset loaded: {len(self.df)} samples")
        print(f"   • AD: {(self.df['label'] == 1).sum()}")
        print(f"   • CN: {(self.df['label'] == 0).sum()}")
    
    def __len__(self):
        return len(self.df)
    
    def load_dicom_volume(self, dicom_folder_path):
        """Load DICOM files from folder and create 3D volume"""
        full_path = self.adni_base_path / dicom_folder_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"DICOM folder not found: {full_path}")
        
        # Get all DICOM files
        dicom_files = list(full_path.glob('*.dcm'))
        if not dicom_files:
            raise ValueError(f"No DICOM files found in: {full_path}")
        
        # Load and sort DICOM files
        dicom_data = []
        for dcm_file in dicom_files:
            try:
                ds = pydicom.dcmread(dcm_file)
                if hasattr(ds, 'pixel_array') and hasattr(ds, 'SliceLocation'):
                    dicom_data.append((ds.SliceLocation, ds.pixel_array))
            except Exception as e:
                continue
        
        if not dicom_data:
            raise ValueError(f"No valid DICOM data found in: {full_path}")
        
        # Sort by slice location and stack
        dicom_data.sort(key=lambda x: x[0])
        volume = np.stack([data[1] for data in dicom_data], axis=0)
        
        return volume.astype(np.float32)
    
    def preprocess_volume(self, volume):
        """Preprocess 3D volume: normalize and resize"""
        # Normalize to [0, 1]
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        
        # Simple resize by taking every nth slice/pixel
        current_shape = volume.shape
        resize_factors = [current_shape[i] // self.target_size[i] for i in range(3)]
        resize_factors = [max(1, f) for f in resize_factors]  # Ensure at least 1
        
        # Downsample
        resized = volume[::resize_factors[0], ::resize_factors[1], ::resize_factors[2]]
        
        # Pad or crop to exact target size
        final_volume = np.zeros(self.target_size)
        min_shape = [min(resized.shape[i], self.target_size[i]) for i in range(3)]
        final_volume[:min_shape[0], :min_shape[1], :min_shape[2]] = resized[:min_shape[0], :min_shape[1], :min_shape[2]]
        
        return final_volume
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        try:
            # Load DICOM volume
            volume = self.load_dicom_volume(row['dicom_folder_path'])
            
            # Preprocess
            volume = self.preprocess_volume(volume)
            
            # Add channel dimension and convert to tensor
            volume = torch.FloatTensor(volume).unsqueeze(0)  # Shape: (1, D, H, W)
            
            if self.transform:
                volume = self.transform(volume)
            
            label = torch.LongTensor([row['label']])[0]
            
            return volume, label
            
        except Exception as e:
            print(f"⚠️ Error loading {row['Subject']}: {e}")
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

# Training and evaluation functions
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
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.3f}',
            'Acc': f'{100.*correct/total:.1f}%'
        })
    
    return running_loss / len(dataloader), 100. * correct / total

def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Evaluating"):
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
    
    accuracy = accuracy_score(all_targets, all_preds)
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except:
        auc = 0.5
    
    return running_loss / len(dataloader), accuracy * 100, auc

def train_model(model, train_loader, val_loader, num_epochs=20, lr=0.001):
    """Train a model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    best_val_acc = 0
    best_model_state = None
    train_history = {'loss': [], 'acc': []}
    val_history = {'loss': [], 'acc': [], 'auc': []}
    
    for epoch in range(num_epochs):
        print(f"\n📈 Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # Record history
        train_history['loss'].append(train_loss)
        train_history['acc'].append(train_acc)
        val_history['loss'].append(val_loss)
        val_history['acc'].append(val_acc)
        val_history['auc'].append(val_auc)
        
        print(f"Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.1f}%")
        print(f"Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.1f}%, Val AUC: {val_auc:.3f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, train_history, val_history, best_val_acc

# Main training script
def main():
    print("🧠 3D MRI Classification Baseline Training")
    print("=" * 50)
    
    # Load data splits
    splits_folder = Path(r"D:\ADNI\AD_CN\proteomics\Biomarkers Consortium Plasma Proteomics MRM\MRI\splits")
    adni_base_path = r"D:\ADNI\AD_CN\proteomics\Biomarkers Consortium Plasma Proteomics MRM\MRI\ADNI"
    
    train_df = pd.read_csv(splits_folder / "train_split.csv")
    test_df = pd.read_csv(splits_folder / "test_split.csv")
    
    print(f"📊 Data loaded:")
    print(f"   • Training samples: {len(train_df)}")
    print(f"   • Test samples: {len(test_df)}")
    
    # Create datasets
    target_size = (64, 64, 64)  # Manageable size for training
    
    # Split training data for validation (80/20)
    train_size = int(0.8 * len(train_df))
    train_subset = train_df.iloc[:train_size]
    val_subset = train_df.iloc[train_size:]
    
    train_dataset = MRIDataset(train_subset, adni_base_path, target_size=target_size)
    val_dataset = MRIDataset(val_subset, adni_base_path, target_size=target_size)
    test_dataset = MRIDataset(test_df, adni_base_path, target_size=target_size)
    
    # Create data loaders
    batch_size = 4  # Small batch size for 3D data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Define models using PyTorch's built-in implementations
    models_dict = {
        'ResNet3D': create_resnet3d(num_classes=2),
        'AlexNet3D': create_alexnet3d(num_classes=2),
        'ConvNeXt3D': create_convnext3d(num_classes=2),
        'ViT3D': create_vit3d(num_classes=2)
    }
    
    results = {}
    
    # Train each model
    for model_name, model in models_dict.items():
        print(f"\n🚀 Training {model_name}")
        print("-" * 30)
        
        model = model.to(device)
        
        # Train the model
        trained_model, train_hist, val_hist, best_val_acc = train_model(
            model, train_loader, val_loader, num_epochs=10, lr=0.001
        )
        
        # Test evaluation
        print(f"\n🔍 Testing {model_name}...")
        test_loss, test_acc, test_auc = evaluate(trained_model, test_loader, nn.CrossEntropyLoss(), device)
        
        results[model_name] = {
            'best_val_acc': best_val_acc,
            'test_acc': test_acc,
            'test_auc': test_auc,
            'train_history': train_hist,
            'val_history': val_hist
        }
        
        print(f"✅ {model_name} Results:")
        print(f"   • Best Val Acc: {best_val_acc:.1f}%")
        print(f"   • Test Acc: {test_acc:.1f}%")
        print(f"   • Test AUC: {test_auc:.3f}")
        
        # Save model
        torch.save(trained_model.state_dict(), splits_folder / f"{model_name}_best.pth")
    
    # Summary results
    print(f"\n📊 FINAL RESULTS SUMMARY")
    print("=" * 50)
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
    results_df.to_csv(splits_folder / "baseline_results.csv")
    
    print(f"\n💾 All models and results saved to: {splits_folder}")
    print("🎉 Baseline training complete!")

if __name__ == "__main__":
    main()
