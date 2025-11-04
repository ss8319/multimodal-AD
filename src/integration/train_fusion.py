"""
Training script for multimodal fusion classifier
Simple implementation with train/test evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_recall_fscore_support, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import sys
import json
import wandb
import os
from datetime import datetime

# Add paths
sys.path.append(str(Path(__file__).parent.parent / "mri" / "BrainIAC" / "src"))

from multimodal_dataset import MultimodalDataset
from fusion_model import (
    get_model, get_weighted_fusion_model, get_asymmetric_fusion_model, 
    get_simple_cross_modal_attention_model, get_cross_transformer_fusion_model,
    get_kronecker_product_fusion_model
)
from load_brainiac import load_brainiac
from utils import MetricsCalculator, WandBLogger, VisualizationCreator, aggregate_cv_results, print_results, compute_best_score


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance.
    
    Focal Loss = -alpha * (1-pt)^gamma * log(pt)
    where pt is the predicted probability for the true class.
    
    Args:
        alpha: Weighting factor for rare class (default: 1.0)
        gamma: Focusing parameter (default: 2.0)
        reduction: Specifies the reduction to apply to the output
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Compute pt (predicted probability for true class)
        pt = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def print_feature_statistics(loader, protein_dim, num_batches=1):
    """
    Print statistics about feature distributions to diagnose scaling issues
    
    Args:
        loader: DataLoader with multimodal data
        protein_dim: Dimension of protein features
        num_batches: Number of batches to analyze
    """
    print("\n" + "="*80)
    print("FEATURE DISTRIBUTION DIAGNOSTICS")
    print("="*80)
    
    protein_stats = {'min': float('inf'), 'max': float('-inf'), 'mean': 0, 'std': 0, 'count': 0}
    mri_stats = {'min': float('inf'), 'max': float('-inf'), 'mean': 0, 'std': 0, 'count': 0}
    
    protein_values = []
    mri_values = []
    
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break
        
        fused = batch['fused_features']  # [B, protein_dim + 768]
        protein_feat = fused[:, :protein_dim]
        mri_feat = fused[:, protein_dim:]
        
        protein_values.append(protein_feat.cpu().numpy().flatten())
        mri_values.append(mri_feat.cpu().numpy().flatten())
    
    protein_all = np.concatenate(protein_values)
    mri_all = np.concatenate(mri_values)
    
    print(f"\n📊 PROTEIN FEATURES ({protein_dim}-dim):")
    print(f"   Range: [{protein_all.min():.4f}, {protein_all.max():.4f}]")
    print(f"   Mean:  {protein_all.mean():.4f}")
    print(f"   Std:   {protein_all.std():.4f}")
    print(f"   Median: {np.median(protein_all):.4f}")
    
    print(f"\n📊 MRI FEATURES (768-dim):")
    print(f"   Range: [{mri_all.min():.4f}, {mri_all.max():.4f}]")
    print(f"   Mean:  {mri_all.mean():.4f}")
    print(f"   Std:   {mri_all.std():.4f}")
    print(f"   Median: {np.median(mri_all):.4f}")
    
    print(f"\n⚠️  IMBALANCE ANALYSIS:")
    print(f"   Dimension ratio (MRI/Protein): {768 / protein_dim:.1f}x")
    print(f"   Std ratio (MRI/Protein): {mri_all.std() / protein_all.std():.2f}x")
    print(f"   Mean ratio (|MRI|/|Protein|): {abs(mri_all.mean()) / abs(protein_all.mean()) if protein_all.mean() != 0 else 'inf':.2f}x")
    
    print(f"\n✅ INTERPRETATION:")
    if protein_all.std() > 0 and mri_all.std() > 0:
        std_ratio = mri_all.std() / protein_all.std()
        if std_ratio > 2:
            print(f"   ⚠️  MRI has {std_ratio:.1f}x higher variance - may dominate gradients")
        else:
            print(f"   ✓ Variances are well-balanced")
    
    print("="*80 + "\n")


def create_cv_splits(csv_path, n_splits=5, split_ratio=(0.6, 0.2, 0.2), random_state=42):
    """
    Create stratified K-fold cross-validation splits with specified ratio
    
    Args:
        csv_path: Path to CSV with all data
        n_splits: Number of CV folds
        split_ratio: (train, val, test) ratio - e.g., (0.6, 0.2, 0.2)
        random_state: Random seed for reproducibility
    
    Returns:
        List of (train_indices, val_indices, test_indices) for each fold
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Get labels for stratification
    labels = (df['research_group'] == 'AD').astype(int).values
    
    # Create splits
    cv_splits = []
    train_ratio, val_ratio, test_ratio = split_ratio
    
    print(f"\n{'='*60}")
    print(f"CREATING {n_splits}-FOLD CROSS-VALIDATION SPLITS")
    print(f"{'='*60}")
    print(f"Total samples: {len(df)}")
    print(f"Split ratio (train:val:test): {train_ratio}:{val_ratio}:{test_ratio}")
    print(f"Random state: {random_state}")
    print(f"Class distribution: AD={labels.sum()}, CN={(1-labels).sum()}")
    print()
    
    # Get all indices
    all_indices = np.arange(len(df))
    
    # DIRECT APPROACH:
    # 1. First create n_splits test sets with test_ratio*n_samples samples each
    # 2. Then for each fold, split the remaining samples into train and val
    
    # Calculate target sizes
    n_samples = len(df)
    n_test = int(test_ratio * n_samples)
    n_val = int(val_ratio * n_samples)
    n_train = n_samples - n_test - n_val
    
    # Ensure we have at least 1 sample in each split
    if n_test < 1 or n_val < 1 or n_train < 1:
        raise ValueError(f"Split ratio {split_ratio} results in empty splits. Use a different ratio or more samples.")
    
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # For each fold, create a stratified split
    for fold_idx in range(n_splits):
        # For fold i, we'll:
        # 1. Split data into test and train+val (stratified)
        # 2. Split train+val into train and val (stratified)
        
        # First split: full dataset -> test and train+val
        train_val_idx, test_idx = train_test_split(
            all_indices,
            test_size=test_ratio,
            random_state=random_state + fold_idx,
            stratify=labels
        )
        
        # Second split: train+val -> train and val
        train_val_labels = labels[train_val_idx]
        val_size_relative = val_ratio / (train_ratio + val_ratio)
        
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size_relative,
            random_state=random_state + fold_idx + 100,  # Different seed
            stratify=train_val_labels
        )
        
        # Store the split
        cv_splits.append((train_idx, val_idx, test_idx))
        
        # Print fold info
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        test_labels = labels[test_idx]
        
        print(f"Fold {fold_idx}:")
        print(f"  Train: {len(train_idx):2d} samples (AD={train_labels.sum():2d}, CN={(1-train_labels).sum():2d})")
        print(f"  Val:   {len(val_idx):2d} samples (AD={val_labels.sum():2d}, CN={(1-val_labels).sum():2d})")
        print(f"  Test:  {len(test_idx):2d} samples (AD={test_labels.sum():2d}, CN={(1-test_labels).sum():2d})")
        
        # Calculate and print actual ratios
        total = len(df)
        train_pct = len(train_idx) / total * 100
        val_pct = len(val_idx) / total * 100
        test_pct = len(test_idx) / total * 100
        print(f"  Ratio: {train_pct:.1f}%:{val_pct:.1f}%:{test_pct:.1f}% (target: {train_ratio*100:.1f}%:{val_ratio*100:.1f}%:{test_ratio*100:.1f}%)")
    
    print()
    return cv_splits


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    for batch_idx, batch in enumerate(dataloader):
        # Get features and labels
        features = batch['fused_features'].to(device)  # [batch, input_dim]
        labels = batch['label'].to(device)  # [batch]
        
        # Forward pass
        logits = model(features)  # [batch, 2]
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of AD (class 1)
        preds = torch.argmax(logits, dim=1)
        all_probs.extend(probs.detach().cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(dataloader):
            print(f"  Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    # Calculate all metrics using MetricsCalculator
    avg_loss = total_loss / len(dataloader)
    metrics = MetricsCalculator.calculate_all_metrics(all_preds, all_labels, all_probs)
    
    return avg_loss, metrics


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    all_subjects = []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['fused_features'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits = model(features)
            loss = criterion(logits, labels)
            
            # Track metrics
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of AD
            preds = torch.argmax(logits, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_subjects.extend(batch['subject_id'])
    
    # Calculate all metrics using MetricsCalculator
    avg_loss = total_loss / len(dataloader)
    metrics = MetricsCalculator.calculate_all_metrics(all_preds, all_labels, all_probs)
    
    return avg_loss, metrics, all_subjects, all_preds, all_labels, np.asarray(all_probs)




def main(config_overrides=None, wandb_run=None):
    """
    Main training function for multimodal fusion with cross-validation
    
    Args:
        config_overrides: Dict of config parameters to override defaults
        wandb_run: Optional existing W&B run to log to. If None, no W&B logging.
    
    Returns:
        Dict with complete results structure:
        {
            'config': config,
            'fold_results': [...],
            'aggregated_metrics': {...},
            'aggregated_cm': [...]
        }
    """
    # Configuration
    config = {
        # Data paths
        'data_csv': '/home/ssim0068/data/multimodal-dataset/all_icbm.csv',  # All data for CV
        'brainiac_checkpoint': '/home/ssim0068/code/multimodal-AD/BrainIAC/src/checkpoints/BrainIAC.ckpt',
        'protein_run_dir': '/home/ssim0068/multimodal-AD/src/protein/runs/run_20251016_205054',
        'protein_latents_dir': None,
        
        # MRI Encoder config
        'mri_encoder': 'brainiac',  # Options: 'brainiac', 'dinov3'
        'dinov3_hub_repo': '/home/ssim0068/multimodal-AD/src/mri/dinov3',
        'dinov3_model': 'dinov3_vits16',  # dinov3_vits16 (384-dim) or dinov3_vitb16 (768-dim)
        'dinov3_weights': '/home/ssim0068/multimodal-AD/src/mri/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
        'dinov3_slice_axis': 0,  # 0=sagittal, 1=coronal, 2=axial
        'dinov3_stride': 2,  # Extract every Nth slice
        'dinov3_image_size': 224,  # 2D crop size
        
        # Model config
        'protein_model_type': 'nn',  # 'nn' (Neural Network) or 'transformer'
        'protein_layer': 'last_hidden_layer',  # 'last_hidden_layer' for NN, 'transformer_embeddings' for Transformer
        'fusion_model_type': 'kronecker',  # 'simple', 'weighted_attention', 'asymmetric', 'cross_modal_attention', 'cross_transformer', 'kronecker'
        'hidden_dim': 128, #params for the fusion model
        'fusion_dim': 128, #params for weighted fusion model (shared embedding dimension)
        'dropout': 0.3,
         #params for the fusion model
        
        # Cross-transformer specific params
        'cross_transformer_embed_dim': 256,  # Shared embedding dimension
        'cross_transformer_num_heads': 4,  # Number of attention heads
        'cross_transformer_ff_dim': None,  # Feed-forward dim (None = 4 * embed_dim)
        'cross_transformer_dropout': 0.1,  # Dropout for transformer layers
        
        # Kronecker product specific params
        'kronecker_projected_dim': 512,  # Projection dim (512 = project from 49k to 512, reduces overfitting)
        
        # Cross-validation config
        'n_folds': 5,
        'split_ratio': (0.6, 0.2, 0.2),  # train:val:test
        'cv_seed': 42,
        
        # Training config
        'batch_size': 8 if torch.cuda.is_available() else 4,
        'num_epochs': 15,
        'learning_rate': 0.001,
        'model_seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # Loss function config
        'loss_function': 'cross_entropy',  # Options: 'cross_entropy', 'focal'
        'focal_alpha': 2.0,        # Weight for minority class (AD)
        'focal_gamma': 2.0,        # Focusing parameter; when gamma is 0, the loss is equivalent to the cross entropy loss.
        
        # Model selection metric
        'best_metric': 'composite',  # Options: 'composite', 'val_auc', 'val_balanced_acc', 'val_f1', 'val_acc', val_mcc'

        # Weights & Biases defaults (env overrides supported)
        'wandb_project': os.environ.get('WANDB_PROJECT') or 'multimodal-ad',
        'wandb_entity': os.environ.get('WANDB_ENTITY'),
        'wandb_group': os.environ.get('WANDB_GROUP'),
    }
    
    # Apply config overrides if provided
    if config_overrides:
        config.update(config_overrides)
        print(f"Applied config overrides: {config_overrides}")
    
    # Create meaningful save directory name with model info (only if not provided in overrides)
    if 'save_dir' not in config:
        # Include MRI encoder in save directory name
        encoder_suffix = config['mri_encoder']
        if config['mri_encoder'] == 'dinov3':
            # Extract model variant (e.g., "vits16" from "dinov3_vits16")
            model_variant = config['dinov3_model'].replace('dinov3_', '')
            encoder_suffix = f"dinov3_{model_variant}"
        
        config['save_dir'] = (
            f'/home/ssim0068/multimodal-AD/runs/'
            f'{config["fusion_model_type"]}_{config["protein_model_type"]}_{encoder_suffix}'
        )
    
    print("="*60)
    print("MULTIMODAL FUSION TRAINING - CROSS-VALIDATION")
    print("="*60)
    print(f"Model: Fusion (protein + MRI)")
    print(f"Protein model: {config['protein_model_type']} ({config['protein_layer']})")
    print(f"MRI encoder: {config['mri_encoder']}")
    if config['mri_encoder'] == 'dinov3':
        print(f"  DINOv3 model: {config['dinov3_model']}")
        print(f"  Slice axis: {config['dinov3_slice_axis']}, Stride: {config['dinov3_stride']}")
    print(f"Fusion model: {config['fusion_model_type']}")
    print(f"N-folds: {config['n_folds']}")
    print(f"Split ratio: {config['split_ratio']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Device: {config['device']}")
    print(f"CV seed: {config['cv_seed']}")
    print(f"Model seed: {config['model_seed']}")
    print()
    
    # Setup device
    device = torch.device(config['device'])
    
    # Create save directory
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Determine W&B logging behavior
    # Always enable W&B logging unless explicitly disabled
    use_wandb = not config.get('disable_wandb', False)
    
    if use_wandb and wandb_run is not None:
        print("Weights & Biases logging enabled (using existing run)")
        print(f"W&B Run: {wandb_run.name}")
        print(f"W&B Project: {wandb_run.project}")
        config['use_wandb'] = True
        base_wandb_run = wandb_run
        wandb_run_created = False
    elif use_wandb and wandb_run is None:
        print("Weights & Biases logging enabled (will create new run)")
        # Set up wandb config for standalone run
        wandb_config = {
            'use_wandb': True,
            'wandb_project': config.get('wandb_project') or 'multimodal-ad',
            'wandb_entity': config.get('wandb_entity', None),
            'wandb_group': config.get('wandb_group') or f"cv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        config.update(wandb_config)

        run_name = f"{config['fusion_model_type']}_{config['protein_model_type']}"
        base_wandb_run = wandb.init(
            project=config['wandb_project'],
            entity=config['wandb_entity'],
            group=config['wandb_group'],
            name=run_name,
            config={k: v for k, v in config.items() if k != 'wandb_entity'},
            reinit=True
        )
        wandb_run_created = True
    else:
        print("Weights & Biases logging disabled")
        config['use_wandb'] = False
        base_wandb_run = None
        wandb_run_created = False
    
    # Load MRI feature extractor (shared across all folds)
    print(f"Loading MRI encoder: {config['mri_encoder']}")
    from mri_extractors import BrainIACExtractor, DINOv3Extractor
    
    if config['mri_encoder'] == 'brainiac':
        mri_extractor = BrainIACExtractor(
            checkpoint_path=config['brainiac_checkpoint'],
            device=device
        )
    elif config['mri_encoder'] == 'dinov3':
        mri_extractor = DINOv3Extractor(
            model_name=config['dinov3_model'],
            weights_path=config['dinov3_weights'],
            hub_repo_dir=config['dinov3_hub_repo'],
            device=device,
            slice_axis=config['dinov3_slice_axis'],
            stride=config['dinov3_stride'],
            image_size=config['dinov3_image_size']
        )
    else:
        raise ValueError(f"Unknown mri_encoder: {config['mri_encoder']}. Choose 'brainiac' or 'dinov3'")
    
    print(f"  MRI feature dimension: {mri_extractor.feature_dim}")
    print()
    
    # Create CV splits
    cv_splits = create_cv_splits(
        csv_path=config['data_csv'],
        n_splits=config['n_folds'],
        split_ratio=config['split_ratio'],
        random_state=config['cv_seed']
    )
    
    # Save CV splits (indices) for reuse in unimodal baselines
    splits_serializable = [
        {
            'train': train_idx.tolist(),
            'val': val_idx.tolist(),
            'test': test_idx.tolist()
        }
        for (train_idx, val_idx, test_idx) in cv_splits
    ]
    with open(save_dir / 'cv_splits.json', 'w') as f:
        json.dump(splits_serializable, f, indent=2)
    
    # Save subject lists for each split (once at top level)
    try:
        df_full = pd.read_csv(config['data_csv'])
        split_cols = [c for c in ['RID', 'Subject', 'research_group'] if c in df_full.columns]
        
        for fold_idx, (train_idx, val_idx, test_idx) in enumerate(cv_splits):
            fold_name = f'fold_{fold_idx}'
            pd.DataFrame(df_full.iloc[train_idx][split_cols]).to_csv(save_dir / f'{fold_name}_train_split.csv', index=False)
            pd.DataFrame(df_full.iloc[val_idx][split_cols]).to_csv(save_dir / f'{fold_name}_val_split.csv', index=False)
            pd.DataFrame(df_full.iloc[test_idx][split_cols]).to_csv(save_dir / f'{fold_name}_test_split.csv', index=False)
    except Exception as e:
        print(f"  Warning: could not save split subject CSVs: {e}")
    
    # Track results across folds
    fold_results = []
    
    # Train each fold
    for fold_idx, (train_idx, val_idx, test_idx) in enumerate(cv_splits):
        print("\n" + "="*60)
        print(f"FOLD {fold_idx + 1}/{config['n_folds']}")
        print("="*60)
        
        # Create fold directory
        fold_dir = save_dir / f'fold_{fold_idx}'
        fold_dir.mkdir(exist_ok=True)
        
        # Create full dataset
        full_dataset = MultimodalDataset(
            csv_path=config['data_csv'],
            mri_extractor=mri_extractor,
            protein_run_dir=config['protein_run_dir'],
            protein_latents_dir=config.get('protein_latents_dir'),
            protein_model_type=config['protein_model_type'],
            protein_layer=config['protein_layer'],
            device=device
        )
        
        # Create subset datasets for this fold
        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)
        test_dataset = Subset(full_dataset, test_idx)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
        
        # Create fusion model (fresh for each fold)
        print("\nCreating fusion model...")
        torch.manual_seed(config['model_seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config['model_seed'])
        
        protein_dim = full_dataset.protein_dim
        mri_dim = full_dataset.mri_dim  # Get MRI dimension from extractor (384 or 768)
        
        # Print feature statistics only on fold 0 to avoid spam
        if fold_idx == 0:
            print_feature_statistics(train_loader, protein_dim, num_batches=3)
        
        # Create fusion model based on type
        if config['fusion_model_type'] == 'simple':
            model = get_model(
                protein_dim=protein_dim,
                mri_dim=mri_dim,
                hidden_dim=config['hidden_dim'],
                dropout=config['dropout']
            ).to(device)
        elif config['fusion_model_type'] == 'weighted_attention':
            model = get_weighted_fusion_model(
                protein_dim=protein_dim,
                mri_dim=mri_dim,
                fusion_dim=config['fusion_dim'],
                hidden_dim=config['hidden_dim'],
                dropout=config['dropout']
            ).to(device)
        elif config['fusion_model_type'] == 'asymmetric':
            model = get_asymmetric_fusion_model(
                protein_dim=protein_dim,
                mri_dim=mri_dim,
                protein_proj_dim=96,  # Modest expansion for protein
                mri_proj_dim=160,     # More capacity for MRI
                hidden_dim=config['hidden_dim'],
                dropout=config['dropout']
            ).to(device)
        elif config['fusion_model_type'] == 'cross_modal_attention':
            model = get_simple_cross_modal_attention_model(
                protein_dim=protein_dim,
                mri_dim=mri_dim,
                shared_dim=64,  # Smaller dimension for simpler model
                hidden_dim=config['hidden_dim'],
                dropout=config['dropout'],
                residual_alpha=0.7  # Strong residual to prevent attention overfitting
            ).to(device)
        elif config['fusion_model_type'] == 'cross_transformer':
            model = get_cross_transformer_fusion_model(
                protein_dim=protein_dim,
                mri_dim=mri_dim,
                embed_dim=config.get('cross_transformer_embed_dim', 256),
                num_heads=config.get('cross_transformer_num_heads', 4),
                ff_dim=config.get('cross_transformer_ff_dim', None),  # Default: 4 * embed_dim
                num_classes=2,
                dropout=config.get('cross_transformer_dropout', 0.1)
            ).to(device)
        elif config['fusion_model_type'] == 'kronecker':
            model = get_kronecker_product_fusion_model(
                protein_dim=protein_dim,
                mri_dim=mri_dim,
                projected_dim=config.get('kronecker_projected_dim', None),  # None = no projection
                hidden_dim=config['hidden_dim'],
                num_classes=2,
                dropout=config['dropout']
            ).to(device)
        else:
            raise ValueError(f"Unknown fusion_model_type: {config['fusion_model_type']}. Must be 'simple', 'weighted_attention', 'asymmetric', 'cross_modal_attention', 'cross_transformer', or 'kronecker'")
        
        # Setup training with Focal Loss to handle class imbalance
        # Calculate class distribution for reporting
        df_full = pd.read_csv(config['data_csv'])
        train_labels = (df_full.iloc[train_idx]['research_group'] == 'AD').astype(int).values
        class_counts = np.bincount(train_labels)
        
        print(f"Class distribution in training: CN={class_counts[0]}, AD={class_counts[1]}")
        
        # Select loss function based on config
        if config['loss_function'] == 'focal':
            # Use Focal Loss for handling class imbalance
            criterion = FocalLoss(
                alpha=config['focal_alpha'], 
                gamma=config['focal_gamma'], 
                reduction='mean'
            )
            print(f"Using Focal Loss: alpha={config['focal_alpha']}, gamma={config['focal_gamma']}")
        else:
            # Use standard Cross Entropy Loss
            criterion = nn.CrossEntropyLoss()
            print("Using Cross Entropy Loss")
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        # Learning rate scheduler - reduce LR when validation metric plateaus
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',           # Maximize the validation metric
            factor=0.5,          # Reduce LR by half
            patience=5,          # Wait 5 epochs before reducing
            min_lr=1e-6         # Don't reduce below this
        )
        # Track best model for this fold
        best_val_score = float('-inf')  # Initialize to negative infinity for MCC
        best_epoch = 0
        fold_history = {'train': [], 'val': [], 'test': []}
        
        # Initialize W&B logger for this fold
        wandb_logger = WandBLogger(run=base_wandb_run, enabled=use_wandb)
        
        # Training loop
        print("\nTraining...")
        for epoch in range(config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
            print("-" * 40)
            
            # Train
            train_loss, train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            
            # Validate
            val_loss, val_metrics, _, val_preds, val_labels, val_probs = evaluate(
                model, val_loader, criterion, device
            )
            
            # Test (for monitoring only, not for model selection)
            test_loss, test_metrics, _, test_preds, test_labels, test_probs = evaluate(
                model, test_loader, criterion, device
            )
            
            # Store history
            fold_history['train'].append({'loss': train_loss, **train_metrics})
            fold_history['val'].append({'loss': val_loss, **val_metrics})
            fold_history['test'].append({'loss': test_loss, **test_metrics})
            
            # Print results using utility function
            print_results("Train", train_loss, train_metrics)
            print_results("Val", val_loss, val_metrics)
            print_results("Test", test_loss, test_metrics)
            
            # Log metrics using WandBLogger
            wandb_logger.log_metrics(train_metrics, prefix=f"fold_{fold_idx}/train", step=epoch)
            wandb_logger.log_metrics(val_metrics, prefix=f"fold_{fold_idx}/val", step=epoch)
            wandb_logger.log_metrics(test_metrics, prefix=f"fold_{fold_idx}/test", step=epoch)
            
            # Log confusion matrix as image (every 5 epochs)
            if config.get('log_confusion_matrices', True) and epoch % 5 == 0:
                fig = VisualizationCreator.create_confusion_matrix(
                    val_metrics['confusion_matrix'], 
                    f'Validation Confusion Matrix (Fold {fold_idx}, Epoch {epoch+1})'
                )
                if fig is not None:
                    wandb_logger.log_image(wandb.Image(fig), "confusion_matrix", prefix=f"fold_{fold_idx}/val")
            
            # Calculate best metric score based on config
            current_score, metric_desc = compute_best_score(val_metrics, config['best_metric'])
            
            # Step the learning rate scheduler based on validation performance
            scheduler.step(current_score)
            
            # Save best model based on selected metric
            if current_score > best_val_score:
                best_val_score = current_score
                best_epoch = epoch + 1
                
                torch.save({
                    'fold': fold_idx,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_metric': config['best_metric'],
                    'best_val_score': best_val_score,
                    'config': config
                }, fold_dir / 'best_model.pth')
                
                print(f"  ✅ New best model! (Best Metric [{config['best_metric']}]: {current_score:.4f} [{metric_desc}])")
        
        # Load best model and evaluate on test set
        print(f"\n{'='*40}")
        print(f"Best epoch: {best_epoch} (Best Metric [{config['best_metric']}]: {best_val_score:.4f})")
        print(f"{'='*40}")
        
        checkpoint = torch.load(fold_dir / 'best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Final test evaluation
        test_loss, test_metrics, test_subjects, test_preds, test_labels, test_probs = evaluate(
            model, test_loader, criterion, device
        )
        
        # Store fold results
        fold_result = {
            'fold': fold_idx,
            'best_epoch': best_epoch,
            'val_composite_score': best_val_score,
            'test_loss': test_loss,
            'test_acc': test_metrics['accuracy'],
            'test_balanced_acc': test_metrics['balanced_accuracy'],
            'test_auc': test_metrics['auc'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_f1': test_metrics['f1'],
            'test_sensitivity': test_metrics['sensitivity'],
            'test_specificity': test_metrics['specificity'],
            'test_mcc': test_metrics['mcc'],
            'test_cm': test_metrics['confusion_matrix'].tolist(),
            'n_train': len(train_idx),
            'n_val': len(val_idx),
            'n_test': len(test_idx)
        }
        fold_results.append(fold_result)
        
        # Save fold results
        with open(fold_dir / 'results.json', 'w') as f:
            json.dump(fold_result, f, indent=2)
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'subject_id': test_subjects,
            'true_label': test_labels,
            'pred_label': test_preds
        })
        predictions_df.to_csv(fold_dir / 'predictions.csv', index=False)
        
        print(f"\nFold {fold_idx} Final Test Results:")
        print_results("Test", test_loss, test_metrics)
        
        # Log fold results to wandb
        wandb_logger.log_metrics({
            'best_epoch': best_epoch,
            'val_composite_score': best_val_score,
            'test/loss': test_loss,
            'test/accuracy': test_metrics['accuracy'],
            'test/balanced_accuracy': test_metrics['balanced_accuracy'],
            'test/precision': test_metrics['precision'],
            'test/recall': test_metrics['recall'],
            'test/f1': test_metrics['f1'],
            'test/sensitivity': test_metrics['sensitivity'],
            'test/specificity': test_metrics['specificity'],
            'test/mcc': test_metrics['mcc'],
        }, prefix=f"fold_{fold_idx}")
        
        # Log AUC if valid
        if test_metrics['auc'] is not None and not np.isnan(test_metrics['auc']):
            wandb_logger.log_metrics({'test/auc': test_metrics['auc']}, prefix=f"fold_{fold_idx}")
        
        # Log confusion matrix
        fig = VisualizationCreator.create_confusion_matrix(
            test_metrics['confusion_matrix'], 
            f'Test Confusion Matrix (Fold {fold_idx})'
        )
        if fig is not None:
            wandb_logger.log_image(wandb.Image(fig), "confusion_matrix", prefix=f"fold_{fold_idx}/test")
        
        # Log ROC curve if AUC is valid
        if test_metrics['auc'] is not None and not np.isnan(test_metrics['auc']) and len(np.unique(test_labels)) > 1:
            fig = VisualizationCreator.create_roc_curve(
                test_labels, test_probs, test_metrics['auc'], 
                f'ROC Curve (Fold {fold_idx})'
            )
            if fig is not None:
                wandb_logger.log_image(wandb.Image(fig), "roc_curve", prefix=f"fold_{fold_idx}/test")
        
        # Log predictions as a table
        wandb_logger.log_table(wandb.Table(dataframe=predictions_df), "predictions", prefix=f"fold_{fold_idx}/test")
    
    # Aggregate results across folds using utility function
    metrics_to_aggregate = ['test_acc', 'test_balanced_acc', 'test_auc', 'test_precision', 'test_recall', 'test_f1', 'test_sensitivity', 'test_specificity', 'test_mcc']
    aggregated = aggregate_cv_results(fold_results, metrics_to_aggregate)
    
    # Print aggregated results
    for metric in metrics_to_aggregate:
        values = aggregated[metric]
        metric_name = metric.replace('test_', '').replace('_', ' ').upper()
        if np.isnan(values['mean']):
            print(f"{metric_name}: undefined")
        else:
            print(f"{metric_name}: {values['mean']:.4f} ± {values['std']:.4f}")
    
    # Aggregate confusion matrix
    total_cm = np.sum([np.array(f['test_cm']) for f in fold_results], axis=0)
    print(f"\nAggregated Confusion Matrix:")
    print(f"  TN={total_cm[0,0]}, FP={total_cm[0,1]}")
    print(f"  FN={total_cm[1,0]}, TP={total_cm[1,1]}")
    
    # Save aggregated results
    aggregated_results = {
        'config': config,
        'fold_results': fold_results,
        'aggregated_metrics': {k: {'mean': v['mean'], 'std': v['std']} for k, v in aggregated.items()},
        'aggregated_cm': total_cm.tolist()
    }
    
    with open(save_dir / 'aggregated_results.json', 'w') as f:
        json.dump(aggregated_results, f, indent=2)
    
    # Log aggregated results to wandb
    if use_wandb:
        try:
            # Use existing run if provided, otherwise create new run for standalone training
            if base_wandb_run is not None:
                summary_logger = WandBLogger(run=base_wandb_run, enabled=True)
            else:
                summary_logger = WandBLogger(run=None, enabled=False)
            
            # Log aggregated metrics
            summary_metrics = {}
            for metric, values in aggregated.items():
                if not np.isnan(values['mean']):
                    metric_name = metric.replace('test_', '').replace('_', ' ')
                    summary_metrics[f"{metric_name}/mean"] = values['mean']
                    summary_metrics[f"{metric_name}/std"] = values['std']
            
            summary_logger.log_metrics(summary_metrics, prefix="summary")
            
            # Log aggregated confusion matrix
            fig = VisualizationCreator.create_confusion_matrix(
                total_cm, 
                f'Aggregated Confusion Matrix ({config["n_folds"]}-fold CV)'
            )
            if fig is not None:
                summary_logger.log_image(wandb.Image(fig), "confusion_matrix", prefix="summary")
            
            # Create a summary table with all fold results
            fold_summary = []
            for fold_idx, fold in enumerate(fold_results):
                fold_summary.append({
                    'Fold': fold_idx,
                    'Best Epoch': fold['best_epoch'],
                    'Accuracy': f"{fold['test_acc']:.4f}",
                    'Balanced Acc': f"{fold['test_balanced_acc']:.4f}",
                    'AUC': f"{fold['test_auc']:.4f}" if not np.isnan(fold['test_auc']) else "undefined",
                    'F1': f"{fold['test_f1']:.4f}",
                    'Precision': f"{fold['test_precision']:.4f}",
                    'Recall': f"{fold['test_recall']:.4f}",
                    'Sensitivity': f"{fold['test_sensitivity']:.4f}",
                    'Specificity': f"{fold['test_specificity']:.4f}",
                    'MCC': f"{fold['test_mcc']:.4f}"
                })
            
            summary_logger.log_table(wandb.Table(dataframe=pd.DataFrame(fold_summary)), "fold_results", prefix="summary")
            
            # Finish the run if we created it
            if wandb_run_created and base_wandb_run is not None:
                wandb.finish()
        except Exception as e:
            print(f"  Warning: Could not log aggregated results to W&B: {e}")
    
    print(f"\n✅ Cross-validation complete!")
    print(f"Results saved to: {save_dir}")
    if use_wandb:
        print(f"Experiment logged to W&B: {config['wandb_project']}/{config['wandb_group']}")
    
    # Return structured results
    return {
        'config': config,
        'fold_results': fold_results,
        'aggregated_metrics': {k: {'mean': v['mean'], 'std': v['std']} for k, v in aggregated.items()},
        'aggregated_cm': total_cm.tolist(),
        'save_dir': str(save_dir)
    }


if __name__ == "__main__":
    # For standalone execution, create a new W&B run
    results = main(config_overrides=None, wandb_run=None)
    print(f"\nTraining completed successfully!")
    print(f"Results: {results['aggregated_metrics']}")

