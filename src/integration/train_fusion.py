"""
Training script for multimodal fusion classifier
Simple implementation with train/test evaluation
"""

import torch
import torch.nn as nn
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
from fusion_model import get_model
from load_brainiac import load_brainiac


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
    all_true_labels = []  # Store for balanced accuracy
    
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
        all_true_labels.extend(labels.cpu().numpy())
        
        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(dataloader):
            print(f"  Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    
    # Check if model is predicting only one class (pathological case)
    n_unique_preds = len(np.unique(all_preds))
    n_unique_labels = len(np.unique(all_labels))
    
    # Calculate AUC
    if n_unique_labels < 2:
        # Single class in labels - AUC undefined (data split issue)
        print(f"  ⚠️  Warning: Only one class in labels - AUC undefined (data issue)")
        auc = float('nan')  # Use NaN to indicate undefined
    elif n_unique_preds < 2:
        # Model predicting only one class - AUC is 0.5 
        print(f"  ⚠️  Warning: Model predicting only ONE class - Using AUC = 0.5 (random chance)")
        auc = 0.5
    else:
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError as e:
            print(f"  ⚠️  Warning: AUC calculation failed - {e}")
            auc = float('nan')  # Use NaN to indicate calculation error

    
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    balanced_acc = balanced_accuracy_score(all_true_labels, all_preds)
    
    return avg_loss, acc, auc, cm, balanced_acc


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
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    
    # Check if model is predicting only one class (pathological case)
    n_unique_preds = len(np.unique(all_preds))
    n_unique_labels = len(np.unique(all_labels))
    
    # Calculate AUC
    if n_unique_labels < 2:
        # Single class in labels - AUC undefined (data split issue)
        print(f"  ⚠️  Warning: Only one class in labels - AUC undefined (data issue)")
        auc = float('nan')  # Use NaN to indicate undefined
    elif n_unique_preds < 2:
        # Model predicting only one class - AUC is 0.5 
        print(f"  ⚠️  Warning: Model predicting only ONE class - Using AUC = 0.5 (random chance)")
        auc = 0.5
    else:
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError as e:
            print(f"  ⚠️  Warning: AUC calculation failed - {e}")
            auc = float('nan')  # Use NaN to indicate calculation error
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    
    return avg_loss, acc, auc, cm, all_subjects, all_preds, all_labels


def print_results(split_name, loss, acc, auc=None, cm=None):
    """Print evaluation results"""
    print(f"\n{split_name} Results:")
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    if auc is not None:
        if np.isnan(auc):
            print(f"  AUC: undefined")
        else:
            print(f"  AUC: {auc:.4f}")
    if cm is not None:
        print(f"  Confusion Matrix:")
        print(f"    TN={cm[0,0]}, FP={cm[0,1]}")
        print(f"    FN={cm[1,0]}, TP={cm[1,1]}")
        
        # Calculate sensitivity and specificity
        if cm[1,1] + cm[1,0] > 0:
            sensitivity = cm[1,1] / (cm[1,1] + cm[1,0])
            print(f"  Sensitivity: {sensitivity:.4f}")
        if cm[0,0] + cm[0,1] > 0:
            specificity = cm[0,0] / (cm[0,0] + cm[0,1])
            print(f"  Specificity: {specificity:.4f}")


def main():
    # Configuration
    config = {
        # Data paths
        'data_csv': '/home/ssim0068/data/multimodal-dataset/all.csv',  # All data for CV
        'brainiac_checkpoint': '/home/ssim0068/code/multimodal-AD/BrainIAC/src/checkpoints/BrainIAC.ckpt',
        'protein_run_dir': '/home/ssim0068/multimodal-AD/src/protein/runs/run_20251003_133215',
        'protein_latents_dir': None,
        
        # Model config
        'protein_model_type': 'transformer',  # 'mlp' or 'transformer'
        'protein_layer': 'transformer_embeddings',  # 'hidden_layer_2' for MLP, 'transformer_embeddings' for Transformer
        'hidden_dim': 128, #params for the fusion model
        'dropout': 0.3, #params for the fusion model
        
        # Cross-validation config
        'n_folds': 2,
        'split_ratio': (0.6, 0.2, 0.2),  # train:val:test
        'cv_seed': 42,
        
        # Training config
        'batch_size': 8 if torch.cuda.is_available() else 4,
        'num_epochs': 15,
        'learning_rate': 0.001,
        'model_seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # Save config
        'save_dir': f'/home/ssim0068/multimodal-AD/runs/cv_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    }
    
    print("="*60)
    print("MULTIMODAL FUSION TRAINING - CROSS-VALIDATION")
    print("="*60)
    print(f"Model: Fusion (protein + MRI)")
    print(f"Protein model: {config['protein_model_type']} ({config['protein_layer']})")
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
    
    # Initialize wandb
    wandb_config = {
        # Add wandb configuration
        'use_wandb': config.get('use_wandb', True),
        'wandb_project': config.get('wandb_project', 'multimodal-ad'),
        'wandb_entity': config.get('wandb_entity', None),
        'wandb_group': config.get('wandb_group', f"cv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
    }
    
    # Add wandb config to main config
    config.update(wandb_config)
    
    # Initialize wandb config but don't start a run yet
    if config['use_wandb']:
        print("Weights & Biases logging enabled")
        print(f"W&B Project: {config['wandb_project']}")
        print(f"W&B Group: {config['wandb_group']}")
        # We'll initialize runs for each fold and for the final summary
    else:
        print("Weights & Biases logging disabled")
    
    # Load BrainIAC model once (shared across all folds)
    print("Loading BrainIAC model...")
    brainiac_model = load_brainiac(config['brainiac_checkpoint'], device)
    print()
    
    # Create CV splits
    cv_splits = create_cv_splits(
        csv_path=config['data_csv'],
        n_splits=config['n_folds'],
        split_ratio=config['split_ratio'],
        random_state=config['cv_seed']
    )
    
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
            brainiac_model=brainiac_model,
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
        model = get_model(
            protein_dim=protein_dim,
            mri_dim=768,
            hidden_dim=config['hidden_dim'],
            dropout=config['dropout']
        ).to(device)
        
        # Setup training with class weights to handle imbalance
        # Calculate class weights from training data
        df_full = pd.read_csv(config['data_csv'])
        train_labels = (df_full.iloc[train_idx]['research_group'] == 'AD').astype(int).values
        class_counts = np.bincount(train_labels)
        class_weights = torch.FloatTensor(len(train_labels) / (len(class_counts) * class_counts)).to(device)
        
        print(f"Class distribution in training: CN={class_counts[0]}, AD={class_counts[1]}")
        print(f"Class weights: CN={class_weights[0]:.3f}, AD={class_weights[1]:.3f}")
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        # Track best model for this fold
        # Use balanced accuracy for model selection (better for imbalanced data)
        best_val_score = 0
        best_epoch = 0
        fold_history = {'train': [], 'val': [], 'test': []}
        
        # Training loop
        print("\nTraining...")
        for epoch in range(config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
            print("-" * 40)
            
            # Train
            train_loss, train_acc, train_auc, train_cm, train_balanced_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            
            # Validate
            val_loss, val_acc, val_auc, val_cm, _, val_preds, val_labels = evaluate(
                model, val_loader, criterion, device
            )
            
            # Calculate balanced accuracy for validation (better for imbalanced data)
            val_balanced_acc = balanced_accuracy_score(val_labels, val_preds)
            
            # Test (for monitoring only, not for model selection)
            test_loss, test_acc, test_auc, test_cm, _, test_preds, test_labels = evaluate(
                model, test_loader, criterion, device
            )
            test_balanced_acc = balanced_accuracy_score(test_labels, test_preds)
            
            # Store history
            train_metrics = {'loss': train_loss, 'acc': train_acc, 'auc': train_auc, 'balanced_acc': train_balanced_acc}
            val_metrics = {'loss': val_loss, 'acc': val_acc, 'auc': val_auc, 'balanced_acc': val_balanced_acc}
            test_metrics = {'loss': test_loss, 'acc': test_acc, 'auc': test_auc, 'balanced_acc': test_balanced_acc}
            
            fold_history['train'].append(train_metrics)
            fold_history['val'].append(val_metrics)
            fold_history['test'].append(test_metrics)
            
            # Print results
            print_results("Train", train_loss, train_acc, train_auc, train_cm)
            print(f"  Balanced Accuracy: {train_balanced_acc:.4f}")
            print_results("Val", val_loss, val_acc, val_auc, val_cm)
            print(f"  Balanced Accuracy: {val_balanced_acc:.4f}")
            print_results("Test", test_loss, test_acc, test_auc, test_cm)
            print(f"  Balanced Accuracy: {test_balanced_acc:.4f}")
            
            # Log metrics to wandb
            if config['use_wandb']:
                # Initialize wandb for this fold's training if not already initialized
                if wandb.run is None:
                    fold_run_name = f"fusion_{config['protein_model_type']}_fold{fold_idx}_training"
                    wandb.init(
                        project=config['wandb_project'],
                        entity=config['wandb_entity'],
                        group=config['wandb_group'],
                        name=fold_run_name,
                        config={k: v for k, v in config.items() if k != 'wandb_entity'},
                        reinit=True
                    )
                
                # Prepare metrics for logging
                wandb_metrics = {
                    f"train/loss": train_loss,
                    f"train/acc": train_acc,
                    f"train/balanced_acc": train_balanced_acc,
                    f"val/loss": val_loss,
                    f"val/acc": val_acc,
                    f"val/balanced_acc": val_balanced_acc,
                    f"test/loss": test_loss,
                    f"test/acc": test_acc,
                    f"test/balanced_acc": test_balanced_acc,
                    "epoch": epoch
                }
                
                # Only log AUC if it's not NaN
                if not np.isnan(train_auc):
                    wandb_metrics[f"train/auc"] = train_auc
                if not np.isnan(val_auc):
                    wandb_metrics[f"val/auc"] = val_auc
                if not np.isnan(test_auc):
                    wandb_metrics[f"test/auc"] = test_auc
                
                # Log confusion matrix as image
                if config.get('log_confusion_matrices', True) and epoch % 5 == 0:  # Log every 5 epochs to reduce clutter
                    try:
                        import matplotlib.pyplot as plt
                        import seaborn as sns
                        
                        # Create confusion matrix plots
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('True')
                        ax.set_title(f'Validation Confusion Matrix (Epoch {epoch+1})')
                        wandb_metrics[f"val/confusion_matrix"] = wandb.Image(fig)
                        plt.close(fig)
                    except ImportError:
                        print("  Warning: matplotlib/seaborn not available for confusion matrix visualization")
                
                # Log metrics
                wandb.log(wandb_metrics)
            
            # Composite score: Use balanced accuracy + AUC for model selection
            # This prevents models that predict only one class from being selected
            # Handle NaN AUC values (undefined or error)
            if np.isnan(val_auc):
                val_composite_score = val_balanced_acc  # Use only balanced accuracy if AUC is undefined
                print(f"  ⚠️  Using only balanced accuracy for model selection (AUC is undefined)")
            else:
                val_composite_score = (val_balanced_acc + val_auc) / 2
            
            # Save best model based on composite score
            if val_composite_score > best_val_score:
                best_val_score = val_composite_score
                best_epoch = epoch + 1
                
                torch.save({
                    'fold': fold_idx,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auc': val_auc,
                    'val_acc': val_acc,
                    'val_balanced_acc': val_balanced_acc,
                    'val_composite_score': val_composite_score,
                    'test_auc': test_auc,
                    'test_acc': test_acc,
                    'config': config
                }, fold_dir / 'best_model.pth')
                
                print(f"  ✅ New best model! (Val Composite Score: {val_composite_score:.4f}, Bal Acc: {val_balanced_acc:.4f}, AUC: {val_auc:.4f})")
        
        # Load best model and evaluate on test set
        print(f"\n{'='*40}")
        print(f"Best epoch: {best_epoch} (Val Composite Score: {best_val_score:.4f})")
        print(f"{'='*40}")
        
        checkpoint = torch.load(fold_dir / 'best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Final test evaluation
        test_loss, test_acc, test_auc, test_cm, test_subjects, test_preds, test_labels = evaluate(
            model, test_loader, criterion, device
        )
        
        # Calculate precision, recall, F1, balanced accuracy
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, test_preds, average='binary', zero_division=0
        )
        test_final_balanced_acc = balanced_accuracy_score(test_labels, test_preds)
        
        # Store fold results
        fold_result = {
            'fold': fold_idx,
            'best_epoch': best_epoch,
            'val_composite_score': best_val_score,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_balanced_acc': test_final_balanced_acc,
            'test_auc': test_auc,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1,
            'test_cm': test_cm.tolist(),
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
        print_results("Test", test_loss, test_acc, test_auc, test_cm)
        print(f"  Balanced Accuracy: {test_final_balanced_acc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1: {f1:.4f}")
        
        # Log fold results to wandb
        if config['use_wandb']:
            # Create a new run for this fold's final results
            if wandb.run is not None:
                wandb.finish()
            
            fold_run_name = f"fusion_{config['protein_model_type']}_fold{fold_idx}"
            wandb.init(
                project=config['wandb_project'],
                entity=config['wandb_entity'],
                group=config['wandb_group'],
                name=fold_run_name,
                config={k: v for k, v in config.items() if k != 'wandb_entity'},
                reinit=True
            )
            
            # Log fold metrics
            fold_metrics = {
                "fold": fold_idx,
                "best_epoch": best_epoch,
                "val_composite_score": best_val_score,
                "test/loss": test_loss,
                "test/accuracy": test_acc,
                "test/balanced_accuracy": test_final_balanced_acc,
                "test/precision": precision,
                "test/recall": recall,
                "test/f1": f1,
            }
            
            # Only log AUC if it's not NaN
            if not np.isnan(test_auc):
                fold_metrics["test/auc"] = test_auc
            
            wandb.log(fold_metrics)
            
            # Log confusion matrix
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                # Create confusion matrix plot
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_title(f'Test Confusion Matrix (Fold {fold_idx})')
                wandb.log({"test/confusion_matrix": wandb.Image(fig)})
                plt.close(fig)
                
                # Create ROC curve if AUC is valid
                if not np.isnan(test_auc) and len(np.unique(test_labels)) > 1:
                    from sklearn.metrics import roc_curve
                    # Get probabilities for positive class
                    test_probs = torch.softmax(model(torch.tensor(test_dataset.dataset.get_fused_features(test_dataset.indices)).to(device)), dim=1)[:, 1].detach().cpu().numpy()
                    fpr, tpr, _ = roc_curve(test_labels, test_probs)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {test_auc:.3f})')
                    ax.plot([0, 1], [0, 1], 'k--')
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title(f'ROC Curve (Fold {fold_idx})')
                    ax.legend(loc="lower right")
                    wandb.log({"test/roc_curve": wandb.Image(fig)})
                    plt.close(fig)
            except Exception as e:
                print(f"  Warning: Could not create visualizations for wandb: {e}")
            
            # Log predictions as a table
            wandb.log({"test/predictions": wandb.Table(dataframe=predictions_df)})
            
            # Finish the fold run
            wandb.finish()
    
    # Aggregate results across folds
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print("="*60)
    
    metrics_to_aggregate = ['test_acc', 'test_balanced_acc', 'test_auc', 'test_precision', 'test_recall', 'test_f1']
    aggregated = {}
    
    for metric in metrics_to_aggregate:
        values = [f[metric] for f in fold_results]
        
        # Handle NaN values (e.g., undefined AUC)
        if metric == 'test_auc' and any(np.isnan(v) for v in values):
            # If any fold has NaN AUC, report as undefined
            aggregated[metric] = {
                'mean': float('nan'),
                'std': float('nan'),
                'values': values
            }
            metric_name = metric.replace('test_', '').replace('_', ' ').upper()
            print(f"{metric_name}: undefined (some folds had undefined AUC)")
        else:
            # Filter out NaN values for calculation
            valid_values = [v for v in values if not np.isnan(v)]
            if valid_values:
                aggregated[metric] = {
                    'mean': np.mean(valid_values),
                    'std': np.std(valid_values),
                    'values': values
                }
                metric_name = metric.replace('test_', '').replace('_', ' ').upper()
                print(f"{metric_name}: {aggregated[metric]['mean']:.4f} ± {aggregated[metric]['std']:.4f}")
            else:
                aggregated[metric] = {
                    'mean': float('nan'),
                    'std': float('nan'),
                    'values': values
                }
                metric_name = metric.replace('test_', '').replace('_', ' ').upper()
                print(f"{metric_name}: undefined (all folds had undefined values)")
    
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
    if config['use_wandb']:
        # Create a new run for aggregated results
        if wandb.run is not None:
            wandb.finish()
        
        # Initialize the final summary run
        wandb.init(
            project=config['wandb_project'],
            entity=config['wandb_entity'],
            group=config['wandb_group'],
            name=f"fusion_{config['protein_model_type']}_{config['n_folds']}fold_cv_summary",
            config={k: v for k, v in config.items() if k != 'wandb_entity'},
            reinit=True
        )
        
        # Log aggregated metrics
        summary_metrics = {}
        for metric, values in aggregated.items():
            if not np.isnan(values['mean']):
                metric_name = metric.replace('test_', '').replace('_', ' ')
                summary_metrics[f"summary/{metric_name}/mean"] = values['mean']
                summary_metrics[f"summary/{metric_name}/std"] = values['std']
        
        # Log confusion matrix
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create aggregated confusion matrix plot
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(f'Aggregated Confusion Matrix ({config["n_folds"]}-fold CV)')
            summary_metrics["summary/confusion_matrix"] = wandb.Image(fig)
            plt.close(fig)
            
            # Create bar chart of metrics across folds
            metrics_to_plot = ['test_acc', 'test_balanced_acc', 'test_f1']
            metrics_to_plot = [m for m in metrics_to_plot if m in aggregated]
            
            if metrics_to_plot:
                fig, ax = plt.subplots(figsize=(10, 6))
                x = np.arange(len(metrics_to_plot))
                means = [aggregated[m]['mean'] for m in metrics_to_plot]
                stds = [aggregated[m]['std'] for m in metrics_to_plot]
                
                ax.bar(x, means, yerr=stds, alpha=0.7, capsize=10)
                ax.set_xticks(x)
                ax.set_xticklabels([m.replace('test_', '').replace('_', ' ').title() for m in metrics_to_plot])
                ax.set_ylabel('Score')
                ax.set_title(f'Performance Metrics ({config["n_folds"]}-fold CV)')
                ax.set_ylim(0, 1)
                
                for i, v in enumerate(means):
                    ax.text(i, v + 0.05, f'{v:.3f}±{stds[i]:.3f}', ha='center')
                
                summary_metrics["summary/metrics_comparison"] = wandb.Image(fig)
                plt.close(fig)
        except Exception as e:
            print(f"  Warning: Could not create visualizations for wandb: {e}")
        
        # Log all metrics
        wandb.log(summary_metrics)
        
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
                'Recall': f"{fold['test_recall']:.4f}"
            })
        
        wandb.log({"summary/fold_results": wandb.Table(data=fold_summary)})
        
        # Finish the run
        wandb.finish()
    
    print(f"\n✅ Cross-validation complete!")
    print(f"Results saved to: {save_dir}")
    if config['use_wandb']:
        print(f"Experiment logged to W&B: {config['wandb_project']}/{config['wandb_group']}")


if __name__ == "__main__":
    main()

