"""
Evaluation and visualization functions for 3D MRI classification models.

This module provides functions for evaluating model performance, computing metrics,
and visualizing results including confusion matrices, ROC curves, and predictions.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from typing import Tuple, List, Optional
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def evaluate_model(model: nn.Module, 
                  dataloader: DataLoader, 
                  criterion: nn.Module, 
                  device: torch.device) -> Tuple[float, float, float, List, List, List]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: PyTorch model to evaluate
        dataloader: Data loader for evaluation
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        Tuple of (average_loss, accuracy_percentage, auc_score, predictions, probabilities, targets)
    """
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
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except:
        auc = 0.5
    
    # Print class distribution
    unique_targets, counts = np.unique(all_targets, return_counts=True)
    print(f"   📊 Validation class distribution: {dict(zip(unique_targets, counts))}")
    
    return running_loss / len(dataloader), accuracy * 100, auc, all_preds, all_probs, all_targets

def visualize_predictions(all_targets: List, 
                         all_preds: List, 
                         all_probs: List, 
                         model_name: str = '',
                         save_path: Optional[str] = None) -> None:
    """
    Visualize validation predictions vs targets.
    
    Args:
        all_targets: List of true labels
        all_preds: List of predicted labels
        all_probs: List of prediction probabilities
        model_name: Name of the model for the plot title
        save_path: Optional path to save the plot
    """
    
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
    
    # Save the plot if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Validation visualization saved as: {save_path}")
    
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

def compute_metrics(targets: List, predictions: List, probabilities: List) -> dict:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        targets: List of true labels
        predictions: List of predicted labels
        probabilities: List of prediction probabilities
        
    Returns:
        Dictionary containing all metrics
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
    
    # Basic metrics
    accuracy = accuracy_score(targets, predictions)
    auc = roc_auc_score(targets, probabilities)
    
    # Precision, recall, F1
    precision = precision_score(targets, predictions, average='binary')
    recall = recall_score(targets, predictions, average='binary')
    f1 = f1_score(targets, predictions, average='binary')
    
    # Confusion matrix
    cm = confusion_matrix(targets, predictions)
    
    # Classification report
    report = classification_report(targets, predictions, target_names=['CN', 'AD'], output_dict=True)
    
    metrics = {
        'accuracy': accuracy,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report': report,
        'total_samples': len(targets),
        'cn_samples': sum(1 for t in targets if t == 0),
        'ad_samples': sum(1 for t in targets if t == 1)
    }
    
    return metrics

def print_metrics_summary(metrics: dict, model_name: str = '') -> None:
    """
    Print a summary of evaluation metrics.
    
    Args:
        metrics: Dictionary of metrics from compute_metrics
        model_name: Name of the model for the summary
    """
    print(f"\n📊 Metrics Summary for {model_name}")
    print("=" * 50)
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"AUC:       {metrics['auc']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1-Score:  {metrics['f1_score']:.3f}")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"CN samples: {metrics['cn_samples']}")
    print(f"AD samples: {metrics['ad_samples']}")
    
    # Confusion matrix
    print(f"\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"           Predicted")
    print(f"           CN    AD")
    print(f"Actual CN  {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"      AD  {cm[1,0]:4d}  {cm[1,1]:4d}")

def save_results_to_csv(targets: List, 
                       predictions: List, 
                       probabilities: List, 
                       save_path: str,
                       model_name: str = '') -> None:
    """
    Save evaluation results to CSV file.
    
    Args:
        targets: List of true labels
        predictions: List of predicted labels
        probabilities: List of prediction probabilities
        save_path: Path to save the CSV file
        model_name: Name of the model
    """
    results_df = pd.DataFrame({
        'true_label': targets,
        'predicted_label': predictions,
        'probability_ad': probabilities,
        'correct': [t == p for t, p in zip(targets, predictions)]
    })
    
    # Add sample index
    results_df.index.name = 'sample_index'
    
    # Save to CSV
    results_df.to_csv(save_path)
    print(f"💾 Results saved to: {save_path}")
    
    # Print summary
    accuracy = (results_df['correct'].sum() / len(results_df)) * 100
    print(f"   • Overall accuracy: {accuracy:.1f}%")
    print(f"   • Total samples: {len(results_df)}")
    print(f"   • Correct predictions: {results_df['correct'].sum()}")
    print(f"   • Incorrect predictions: {len(results_df) - results_df['correct'].sum()}")

def plot_training_history(train_history: dict, 
                         val_history: dict, 
                         model_name: str = '',
                         save_path: Optional[str] = None) -> None:
    """
    Plot training history (loss and accuracy over epochs).
    
    Args:
        train_history: Training history dictionary
        val_history: Validation history dictionary
        model_name: Name of the model for the plot title
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'Training History for {model_name}', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(train_history['loss']) + 1)
    
    # Plot loss
    ax1.plot(epochs, train_history['loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, val_history['loss'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, train_history['acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_history['acc'], 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training history plot saved as: {save_path}")
    
    plt.show()
