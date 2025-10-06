"""
Training script for multimodal fusion classifier
Simple implementation with train/test evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import sys

# Add paths
sys.path.append(str(Path(__file__).parent.parent / "mri" / "BrainIAC" / "src"))

from multimodal_dataset import MultimodalDataset
from fusion_model import get_model
from load_brainiac import load_brainiac


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
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
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(dataloader):
            print(f"  Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, acc


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
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0  # In case of single class in batch
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return avg_loss, acc, auc, cm, all_subjects, all_preds, all_labels


def print_results(split_name, loss, acc, auc=None, cm=None):
    """Print evaluation results"""
    print(f"\n{split_name} Results:")
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    if auc is not None:
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
        'train_csv': '/home/ssim0068/data/multimodal-dataset/train.csv',
        'test_csv': '/home/ssim0068/data/multimodal-dataset/test.csv',
        'brainiac_checkpoint': '/home/ssim0068/code/multimodal-AD/BrainIAC/src/checkpoints/BrainIAC.ckpt',
        'protein_run_dir': None,  # On-the-fly extraction (may have compatibility issues)
        'protein_latents_dir': '/home/ssim0068/data/multimodal-dataset/protein_latents',  # Pre-extracted latents
        'protein_model_type': 'mlp',  # 'mlp' or 'transformer'
        'protein_layer': 'hidden_layer_2',  # 'hidden_layer_2' for MLP, 'transformer_embeddings' for Transformer
        'batch_size': 8 if torch.cuda.is_available() else 4,  # Larger batch on GPU
        'num_epochs': 50,
        'learning_rate': 0.001,
        'hidden_dim': 128,
        'dropout': 0.3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # Auto-detect
        'save_dir': '/home/ssim0068/multimodal-AD/runs/fusion'
    }
    
    print("="*60)
    print("MULTIMODAL FUSION TRAINING")
    print("="*60)
    print(f"Model: Fusion (protein + MRI)")
    print(f"Protein model: {config['protein_model_type']} ({config['protein_layer']})")
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Device: {config['device']}")
    print()
    
    # Setup device
    device = torch.device(config['device'])
    
    # Load BrainIAC model once (shared across all dataloaders)
    print("Loading BrainIAC model...")
    brainiac_model = load_brainiac(config['brainiac_checkpoint'], device)
    print()
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = MultimodalDataset(
        csv_path=config['train_csv'],
        brainiac_model=brainiac_model,
        protein_run_dir=config['protein_run_dir'],
        protein_latents_dir=config.get('protein_latents_dir'),
        protein_model_type=config['protein_model_type'],
        protein_layer=config['protein_layer'],
        device=device
    )
    
    test_dataset = MultimodalDataset(
        csv_path=config['test_csv'],
        brainiac_model=brainiac_model,
        protein_run_dir=config['protein_run_dir'],
        protein_latents_dir=config.get('protein_latents_dir'),
        protein_model_type=config['protein_model_type'],
        protein_layer=config['protein_layer'],
        device=device
    )
    print()
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Create fusion model
    print("Creating fusion model...")
    protein_dim = train_dataset.protein_dim
    model = get_model(
        protein_dim=protein_dim,
        mri_dim=768,
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout']
    ).to(device)
    print()
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    print("="*60)
    print("TRAINING")
    print("="*60)
    
    best_test_auc = 0
    best_epoch = 0
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Evaluate on test set
        test_loss, test_acc, test_auc, test_cm, _, _, _ = evaluate(
            model, test_loader, criterion, device
        )
        
        # Print results
        print_results("Train", train_loss, train_acc)
        print_results("Test", test_loss, test_acc, test_auc, test_cm)
        
        # Save best model
        if test_auc > best_test_auc:
            best_test_auc = test_auc
            best_epoch = epoch + 1
            
            # Save model
            save_dir = Path(config['save_dir'])
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_auc': test_auc,
                'test_acc': test_acc,
                'config': config
            }, save_dir / 'best_model.pth')
            
            print(f"  ✅ New best model saved! (AUC: {test_auc:.4f})")
    
    # Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best Test AUC: {best_test_auc:.4f} (Epoch {best_epoch})")
    print(f"Model saved to: {config['save_dir']}/best_model.pth")


if __name__ == "__main__":
    main()

