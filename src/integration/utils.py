"""
Utility classes and functions for multimodal fusion training
"""

import numpy as np
import torch
import wandb
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, confusion_matrix,
    matthews_corrcoef
)


class MetricsCalculator:
    """
    Centralized metrics calculation for multimodal fusion training
    """
    
    @staticmethod
    def calculate_all_metrics(predictions, labels, probabilities=None):
        """
        Calculate all metrics in one place
        
        Args:
            predictions: Array of predicted labels
            labels: Array of true labels
            probabilities: Array of prediction probabilities (optional, for AUC)
        
        Returns:
            Dict with all calculated metrics
        """
        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        balanced_acc = balanced_accuracy_score(labels, predictions)
        
        # Confusion matrix for sensitivity/specificity
        cm = confusion_matrix(labels, predictions, labels=[0, 1])
        
        # Sensitivity and specificity
        sensitivity = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
        specificity = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
        
        # Precision, recall, F1
        precision = precision_score(labels, predictions, average='binary', zero_division=0)
        recall = recall_score(labels, predictions, average='binary', zero_division=0)
        f1 = f1_score(labels, predictions, average='binary', zero_division=0)
        
        # Matthews Correlation Coefficient (MCC)
        # MCC is a balanced measure that considers all four confusion matrix categories
        # Range: -1 (total disagreement) to +1 (perfect agreement)
        # Reference: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html
        mcc = matthews_corrcoef(labels, predictions)
        
        # AUC calculation with safety checks
        auc = MetricsCalculator._safe_auc_score(labels, probabilities) if probabilities is not None else None
        
        return {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'mcc': mcc,
            'auc': auc,
            'confusion_matrix': cm
        }
    
    @staticmethod
    def _safe_auc_score(labels, probabilities):
        """
        Calculate AUC with safety checks for edge cases
        """
        if probabilities is None:
            return None
        
        # Check if we have both classes
        n_unique_labels = len(np.unique(labels))
        n_unique_probs = len(np.unique(probabilities))
        
        if n_unique_labels < 2:
            print("  Warning: Only one class in labels - AUC undefined")
            return float('nan')
        elif n_unique_probs < 2:
            print("  Warning: Model predicting only ONE class - Using AUC = 0.5 (random chance)")
            return 0.5
        else:
            try:
                return roc_auc_score(labels, probabilities)
            except ValueError as e:
                print(f"  Warning: AUC calculation failed - {e}")
                return float('nan')


class WandBLogger:
    """
    Simplified W&B logging wrapper
    """
    
    def __init__(self, run=None, enabled=True):
        self.run = run
        self.enabled = enabled and run is not None
    
    def log_metrics(self, metrics_dict, prefix="", step=None, exclude_nan=True):
        """
        Log metrics with automatic prefixing and error handling
        
        Args:
            metrics_dict: Dict of metrics to log
            prefix: Prefix for metric names (e.g., "fold_0/train")
            step: Step/epoch number
            exclude_nan: Whether to exclude NaN values from logging
        """
        if not self.enabled:
            return
        
        # Filter out NaN values if requested
        if exclude_nan:
            metrics_dict = {k: v for k, v in metrics_dict.items() 
                           if not (isinstance(v, float) and np.isnan(v))}
        
        # Add prefix to metric names
        if prefix:
            prefixed_metrics = {f"{prefix}/{k}": v for k, v in metrics_dict.items()}
        else:
            prefixed_metrics = metrics_dict.copy()
        
        # Add step if provided
        if step is not None:
            prefixed_metrics['step'] = step
        
        # Log with error handling
        try:
            wandb.log(prefixed_metrics)
        except Exception as e:
            print(f"Warning: W&B logging failed: {e}")
    
    def log_image(self, image, name, prefix=""):
        """
        Log image with error handling
        """
        if not self.enabled:
            return
        
        try:
            if prefix:
                wandb.log({f"{prefix}/{name}": image})
            else:
                wandb.log({name: image})
        except Exception as e:
            print(f"Warning: W&B image logging failed: {e}")
    
    def log_table(self, table, name, prefix=""):
        """
        Log table with error handling
        """
        if not self.enabled:
            return
        
        try:
            if prefix:
                wandb.log({f"{prefix}/{name}": table})
            else:
                wandb.log({name: table})
        except Exception as e:
            print(f"Warning: W&B table logging failed: {e}")


class VisualizationCreator:
    """
    Create visualizations for W&B logging
    """
    
    @staticmethod
    def create_confusion_matrix(cm, title="Confusion Matrix"):
        """
        Create confusion matrix visualization
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(title)
            
            return fig
        except ImportError:
            print("Warning: matplotlib/seaborn not available for confusion matrix visualization")
            return None
    
    @staticmethod
    def create_roc_curve(labels, probabilities, auc_score, title="ROC Curve"):
        """
        Create ROC curve visualization
        """
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import roc_curve
            
            fpr, tpr, _ = roc_curve(labels, probabilities)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.3f})')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(title)
            ax.legend(loc="lower right")
            
            return fig
        except ImportError:
            print("Warning: matplotlib not available for ROC curve visualization")
            return None


def aggregate_cv_results(fold_results, metrics_to_aggregate):
    """
    Aggregate cross-validation results across folds
    
    Args:
        fold_results: List of fold result dictionaries
        metrics_to_aggregate: List of metric names to aggregate
    
    Returns:
        Dict with aggregated metrics (mean, std, values)
    """
    aggregated = {}
    
    for metric in metrics_to_aggregate:
        values = [f[metric] for f in fold_results]

        try:
            numeric_values = np.array(values, dtype=float)
        except (TypeError, ValueError) as exc:
            msg = f"Non-numeric value encountered for metric '{metric}': {values}"
            print(f"ERROR: {msg}")
            raise ValueError(msg) from exc

        if np.isnan(numeric_values).any():
            msg = f"NaN detected for metric '{metric}' across folds: {values}"
            print(f"ERROR: {msg}")
            raise ValueError(msg)

        aggregated[metric] = {
            'mean': float(np.mean(numeric_values)),
            'std': float(np.std(numeric_values)),
            'values': values
        }
    
    return aggregated


def print_results(split_name, loss, metrics_dict):
    """
    Print evaluation results in a clean format
    
    Args:
        split_name: Name of the split (e.g., "Train", "Val", "Test")
        loss: Loss value
        metrics_dict: Dict of metrics from MetricsCalculator
    """
    print(f"\n{split_name} Results:")
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {metrics_dict['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics_dict['balanced_accuracy']:.4f}")
    
    if metrics_dict['auc'] is not None:
        if np.isnan(metrics_dict['auc']):
            print(f"  AUC: undefined")
        else:
            print(f"  AUC: {metrics_dict['auc']:.4f}")
    
    cm = metrics_dict['confusion_matrix']
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}, TP={cm[1,1]}")
    print(f"  Sensitivity: {metrics_dict['sensitivity']:.4f}")
    print(f"  Specificity: {metrics_dict['specificity']:.4f}")
    print(f"  Precision: {metrics_dict['precision']:.4f}")
    print(f"  Recall: {metrics_dict['recall']:.4f}")
    print(f"  F1: {metrics_dict['f1']:.4f}")
    print(f"  MCC: {metrics_dict['mcc']:.4f}")


def compute_best_score(val_metrics, best_metric):
    """
    Compute the score based on the best_metric configuration.
    Calculation of this val metric helps us determine the best checkpoint.
    
    Args:
        val_metrics: Dict of validation metrics from MetricsCalculator
        best_metric: Configuration for the best metric to optimize.
                     Can be 'composite', 'val_auc', 'val_balanced_acc', 'val_f1', 'val_acc', 'val_mcc'
    
    Returns:
        tuple: (score, metric_name)
    """
    if best_metric == 'composite':
        # Composite score: balanced accuracy + AUC
        if np.isnan(val_metrics['auc']):
            return val_metrics['balanced_accuracy'], "Balanced Acc (AUC undefined)"
        else:
            return (val_metrics['balanced_accuracy'] + val_metrics['auc']) / 2, "(Balanced Acc + AUC) / 2"
    elif best_metric == 'val_auc':
        return val_metrics['auc'], "AUC"
    elif best_metric == 'val_balanced_acc':
        return val_metrics['balanced_accuracy'], "Balanced Acc"
    elif best_metric == 'val_f1':
        return val_metrics['f1'], "F1"
    elif best_metric == 'val_acc':
        return val_metrics['accuracy'], "Accuracy"
    elif best_metric == 'val_mcc':
        return val_metrics['mcc'], "MCC"
    else:
        raise ValueError(f"Unknown best_metric: {best_metric}. Use 'composite', 'val_auc', 'val_balanced_acc', 'val_f1', 'val_acc', or 'val_mcc'")
