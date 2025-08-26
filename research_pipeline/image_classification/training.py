"""
Training functions for 3D MRI classification models.

This module provides training loops, loss functions, and optimization utilities
for training deep learning models on 3D MRI data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Focal Loss for handling class imbalance
class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance.
    
    This loss function down-weights easy examples and focuses on hard examples,
    which is particularly useful for imbalanced medical datasets.
    """
    
    def __init__(self, alpha: float = 1, gamma: float = 2, reduction: str = 'mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for class balancing
            gamma: Focusing parameter (higher values focus more on hard examples)
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Focal Loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Focal loss value
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train_epoch(model: nn.Module, 
                dataloader: DataLoader, 
                criterion: nn.Module, 
                optimizer: optim.Optimizer, 
                device: torch.device,
                gradient_clip_norm: float = 1.0) -> Tuple[float, float]:
    """
    Train model for one epoch.
    
    Args:
        model: PyTorch model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        gradient_clip_norm: Maximum gradient norm for clipping
        
    Returns:
        Tuple of (average_loss, accuracy_percentage)
    """
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
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
        
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

def train_model(model: nn.Module, 
                train_loader: DataLoader, 
                val_loader: DataLoader, 
                config: dict,
                model_name: str = '') -> Tuple[nn.Module, Dict, Dict, float]:
    """
    Train a model with early stopping and learning rate scheduling.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Experiment configuration
        model_name: Name of the model for logging
        
    Returns:
        Tuple of (trained_model, train_history, val_history, best_val_acc)
    """
    
    # Calculate class weights to handle imbalance
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())
    
    # Count class frequencies
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    class_weights = torch.FloatTensor([1.0 / count for count in counts]).to(model.device if hasattr(model, 'device') else next(model.parameters()).device)
    
    print(f"📊 Class distribution: {dict(zip(unique_labels, counts))}")
    print(f"⚖️ Class weights: {class_weights.cpu().numpy()}")
    
    # Use focal loss with alpha from class prior (weight minority more)
    pos_prior = 0.5
    if 1 in unique_labels:
        total = counts.sum()
        pos_count = counts[list(unique_labels).index(1)]
        pos_prior = float(pos_count) / float(total)
    alpha = 1.0 - pos_prior
    criterion = FocalLoss(alpha=alpha, gamma=2)
    print(f"🎯 Using Focal Loss (alpha={alpha:.3f}, gamma=2) to handle class imbalance")
    
    # Handle different optimizer configurations for BM-MAE fine-tuned models
    if hasattr(model, 'get_param_groups'):
        # Fine-tuned BM-MAE with different learning rates
        lr_encoder = config['training']['learning_rate'] / 100
        lr_classifier = config['training']['learning_rate']
        wd_head = config['training'].get('weight_decay_classifier_head', 0.0)
        wd_enc = config['training'].get('weight_decay_encoder', 0.0)
        param_groups = model.get_param_groups(lr_encoder=lr_encoder, lr_classifier=lr_classifier)
        # Assign per-group weight decay: assume [encoder, classifier] order
        if len(param_groups) >= 1:
            param_groups[0]['weight_decay'] = wd_enc
        if len(param_groups) >= 2:
            param_groups[1]['weight_decay'] = wd_head
        optimizer = optim.Adam(param_groups)
        print(f"🎯 Using LRs enc/cls: {lr_encoder:.2e}/{lr_classifier:.2e}; WDs enc/cls: {wd_enc:.2e}/{wd_head:.2e}")
    else:
        # Standard optimizer for other models
        if 'frozen' in model_name.lower():
            # Apply classifier head weight decay (only head params are trainable in frozen model)
            weight_decay = config['training'].get('weight_decay_classifier_head', 0.0)
            optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=weight_decay)
            print(f"🎯 Using classifier head weight decay ({weight_decay:.2e}) for frozen model")
        else:
            # Non-frozen model without param groups (rare). Fallback to head-style decay.
            weight_decay = config['training'].get('weight_decay_classifier_head', 0.0)
            optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        patience=3, 
        factor=0.5,
        verbose=True
    )
    
    # Training state
    best_val_acc = 0
    best_model_state = None
    train_history = {'loss': [], 'acc': []}
    val_history = {'loss': [], 'acc': [], 'auc': []}
    
    # Early stopping parameters
    patience = config['training']['early_stopping_patience']
    patience_counter = 0
    min_epochs = config['training']['min_epochs']
    
    print(f"🛑 Early stopping: patience={patience}, min_epochs={min_epochs}")

    # Prepare per-epoch logging
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"training_log_{model_name}.csv"
    if not log_file.exists():
        # Initialize CSV with header
        header_df = pd.DataFrame([{
            'epoch': 0,
            'train_loss': None,
            'train_acc': None,
            'val_loss': None,
            'val_acc': None,
            'val_auc': None,
            'lr': optimizer.param_groups[0]['lr'] if len(optimizer.param_groups) > 0 else None
        }]).iloc[0:0]
        header_df.to_csv(log_file, index=False)
    
    # Training loop
    for epoch in range(config['training']['epochs']):
        print(f"\n📈 Epoch {epoch+1}/{config['training']['epochs']}")
        
        # Preview a random train image (non-blocking) for sanity check
        if config.get('enable_visualization', False):
            try:
                batch = next(iter(train_loader))
                imgs, labels = batch
                idx = random.randrange(0, imgs.shape[0])
                vol = imgs[idx].cpu().numpy()  # (1, D, H, W)
                lbl = int(labels[idx].cpu().item())
                d, h, w = vol.shape[1], vol.shape[2], vol.shape[3]
                mid_slices = [
                    vol[0, d//2, :, :],
                    vol[0, :, h//2, :],
                    vol[0, :, :, w//2]
                ]
                fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                titles = ['Axial', 'Coronal', 'Sagittal']
                vmin, vmax = float(np.min(vol)), float(np.max(vol))
                for ax, img2d, t in zip(axes, mid_slices, titles):
                    ax.imshow(img2d, cmap='gray', vmin=vmin, vmax=vmax)
                    ax.set_title(t)
                    ax.axis('off')
                fig.suptitle(f'{model_name} Train preview (epoch {epoch+1}) - label {lbl}')
                preview_path = Path(config['output_dir']) / f'preview_train_{model_name}_epoch{epoch+1}.png'
                plt.tight_layout()
                plt.savefig(preview_path, dpi=120, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                print(f"⚠️ Failed to save train preview: {e}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, 
            next(model.parameters()).device,
            config['training']['gradient_clip_norm']
        )
        
        # Validate
        try:
            from .evaluation import evaluate_model
        except ImportError:
            from evaluation import evaluate_model
        
        # Preview a random val image before evaluation
        if config.get('enable_visualization', False):
            try:
                batch = next(iter(val_loader))
                imgs, labels = batch
                idx = random.randrange(0, imgs.shape[0])
                vol = imgs[idx].cpu().numpy()
                lbl = int(labels[idx].cpu().item())
                d, h, w = vol.shape[1], vol.shape[2], vol.shape[3]
                mid_slices = [
                    vol[0, d//2, :, :],
                    vol[0, :, h//2, :],
                    vol[0, :, :, w//2]
                ]
                fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                titles = ['Axial', 'Coronal', 'Sagittal']
                vmin, vmax = float(np.min(vol)), float(np.max(vol))
                for ax, img2d, t in zip(axes, mid_slices, titles):
                    ax.imshow(img2d, cmap='gray', vmin=vmin, vmax=vmax)
                    ax.set_title(t)
                    ax.axis('off')
                fig.suptitle(f'{model_name} Val preview (epoch {epoch+1}) - label {lbl}')
                preview_path = Path(config['output_dir']) / f'preview_val_{model_name}_epoch{epoch+1}.png'
                plt.tight_layout()
                plt.savefig(preview_path, dpi=120, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                print(f"⚠️ Failed to save val preview: {e}")
        val_loss, val_acc, val_auc, val_preds, val_probs, val_targets = evaluate_model(
            model, val_loader, criterion, next(model.parameters()).device
        )

        # Initialize classifier bias to class prior once (first epoch) for heads with 2 outputs
        try:
            if epoch == 0 and hasattr(model, 'classifier'):
                # estimate prior from train loader labels collected earlier
                p = float(pos_prior)
                p = min(max(p, 1e-3), 1-1e-3)
                bias_val = np.log(p/(1-p))
                last_layer = None
                for m in reversed(list(model.classifier.modules())):
                    if isinstance(m, nn.Linear) and m.out_features == 2:
                        last_layer = m
                        break
                if last_layer is not None:
                    with torch.no_grad():
                        # bias index 1 corresponds to AD class
                        b = last_layer.bias.data
                        b[1] = torch.tensor(bias_val, device=b.device, dtype=b.dtype)
                        last_layer.bias.copy_(b)
                    print(f"⚙️ Initialized classifier bias for AD to prior logit {bias_val:.3f}")
        except Exception as e:
            print(f"⚠️ Bias init skipped: {e}")
        
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

        # Persist epoch metrics to CSV
        try:
            current_lr = optimizer.param_groups[0]['lr'] if len(optimizer.param_groups) > 0 else None
            row_df = pd.DataFrame([{
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_auc': val_auc,
                'lr': current_lr
            }])
            row_df.to_csv(log_file, mode='a', header=False, index=False)
        except Exception as e:
            print(f"⚠️ Failed to write epoch log: {e}")
        
        print(f"Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.1f}%")
        print(f"Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.1f}%, Val AUC: {val_auc:.3f}")
        
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

def create_optimizer(model: nn.Module, 
                    learning_rate: float, 
                    weight_decay: float,
                    model_name: str = '') -> optim.Optimizer:
    """
    Create optimizer with appropriate configuration for the model.
    
    Args:
        model: PyTorch model
        learning_rate: Learning rate
        weight_decay: Weight decay
        model_name: Name of the model for special handling
        
    Returns:
        Configured optimizer
    """
    if hasattr(model, 'get_param_groups'):
        # Special handling for models with parameter groups (e.g., BM-MAE fine-tuned)
        param_groups = model.get_param_groups(
            lr_encoder=learning_rate/100, 
            lr_classifier=learning_rate
        )
        return optim.Adam(param_groups, weight_decay=weight_decay)
    else:
        # Standard optimizer
        if 'frozen' in model_name.lower():
            # Use configured weight decay for frozen models
            return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def create_scheduler(optimizer: optim.Optimizer, 
                    patience: int = 3, 
                    factor: float = 0.5) -> optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        patience: Number of epochs without improvement before reducing LR
        factor: Factor by which to reduce learning rate
        
    Returns:
        Configured learning rate scheduler
    """
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        patience=patience, 
        factor=factor,
        verbose=True
    )
