"""
W&B Hyperparameter Sweep Configuration
Much simpler than Optuna - just define the sweep config and W&B handles the optimization
"""

import wandb
import json
from pathlib import Path
from datetime import datetime

# Import the main training function
from train_fusion import main


def create_sweep_config():
    """
    Create W&B sweep configuration
    """
    sweep_config = {
        'method': 'bayes',  # Bayesian optimization (best choice)
        'metric': {
            'name': 'test/f1',
            'goal': 'maximize'
        },
        'parameters': {
            'focal_alpha': {
                'distribution': 'uniform',
                'min': 1.0,
                'max': 3.5
            },
            'focal_gamma': {
                'distribution': 'uniform', 
                'min': 0.0,
                'max': 3.5
            },
            'learning_rate': {
                'values': [0.0005, 0.001, 0.0015]
            }
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 3,
            'eta': 2
        }
    }
    
    return sweep_config


def create_quick_sweep_config():
    """
    Create quick W&B sweep configuration for testing
    """
    sweep_config = {
        'method': 'random',  # Random search for quick testing
        'metric': {
            'name': 'test/f1',
            'goal': 'maximize'
        },
        'parameters': {
            'focal_alpha': {
                'values': [1.0, 2.0, 3.0]
            },
            'focal_gamma': {
                'values': [0.0, 1.5, 3.0]
            },
            'learning_rate': {
                'values': [0.001]
            }
        }
    }
    
    return sweep_config


def train_with_sweep():
    """
    Training function that W&B will call for each trial
    """
    # Initialize W&B run (this will be called by W&B sweep)
    run = wandb.init()
    
    # Get hyperparameters from W&B
    config = wandb.config
    
    # Create config overrides
    config_overrides = {
        'focal_alpha': config.focal_alpha,
        'focal_gamma': config.focal_gamma,
        'learning_rate': config.learning_rate,
        'save_dir': f'/home/ssim0068/multimodal-AD/runs/wandb_sweep_{run.id}',
        'num_epochs': 15,
        'n_folds': 5,
        'use_wandb': True,  # Enable W&B logging
        'wandb_project': run.project,  # Use the project from the sweep run
        'wandb_group': f'sweep_{run.sweep_id}',
        'wandb_entity': run.entity
    }
    
    try:
        # Run training using the new clean interface
        results = main(config_overrides=config_overrides, wandb_run=run)
        
        # Log final aggregated metrics to W&B
        aggregated_metrics = results['aggregated_metrics']
        
        # Log all aggregated metrics with clear naming
        final_metrics = {
            'test/f1': aggregated_metrics.get('test_f1', {}).get('mean', 0.0),
            'test/accuracy': aggregated_metrics.get('test_acc', {}).get('mean', 0.0),
            'test/balanced_accuracy': aggregated_metrics.get('test_balanced_acc', {}).get('mean', 0.0),
            'test/auc': aggregated_metrics.get('test_auc', {}).get('mean', 0.0),
            'test/precision': aggregated_metrics.get('test_precision', {}).get('mean', 0.0),
            'test/recall': aggregated_metrics.get('test_recall', {}).get('mean', 0.0),
            'test/sensitivity': aggregated_metrics.get('test_sensitivity', {}).get('mean', 0.0),
            'test/specificity': aggregated_metrics.get('test_specificity', {}).get('mean', 0.0),
        }
        
        # Log standard deviations
        final_metrics.update({
            'test/f1_std': aggregated_metrics.get('test_f1', {}).get('std', 0.0),
            'test/accuracy_std': aggregated_metrics.get('test_acc', {}).get('std', 0.0),
            'test/balanced_accuracy_std': aggregated_metrics.get('test_balanced_acc', {}).get('std', 0.0),
            'test/auc_std': aggregated_metrics.get('test_auc', {}).get('std', 0.0),
            'test/precision_std': aggregated_metrics.get('test_precision', {}).get('std', 0.0),
            'test/recall_std': aggregated_metrics.get('test_recall', {}).get('std', 0.0),
            'test/sensitivity_std': aggregated_metrics.get('test_sensitivity', {}).get('std', 0.0),
            'test/specificity_std': aggregated_metrics.get('test_specificity', {}).get('std', 0.0),
        })
        
        # Log hyperparameters for easy tracking
        final_metrics.update({
            'hyperparams/focal_alpha': config.focal_alpha,
            'hyperparams/focal_gamma': config.focal_gamma,
            'hyperparams/learning_rate': config.learning_rate,
        })
        
        wandb.log(final_metrics)
        
        print(f"Trial completed successfully!")
        print(f"Final F1: {final_metrics['test/f1']:.4f} ± {final_metrics['test/f1_std']:.4f}")
        print(f"Final Accuracy: {final_metrics['test/accuracy']:.4f} ± {final_metrics['test/accuracy_std']:.4f}")
        print(f"Final AUC: {final_metrics['test/auc']:.4f} ± {final_metrics['test/auc_std']:.4f}")
        
    except Exception as e:
        print(f"Trial failed: {e}")
        # Log failure metrics
        if wandb.run is not None:
            wandb.log({
                'test/f1': 0.0,
                'test/accuracy': 0.0,
                'test/balanced_accuracy': 0.0,
                'test/auc': 0.0,
                'test/precision': 0.0,
                'test/recall': 0.0,
                'test/sensitivity': 0.0,
                'test/specificity': 0.0,
                'error': str(e)
            })
    
    finally:
        wandb.finish()


def create_and_run_sweep(n_trials=30, quick=False):
    """
    Create and run W&B sweep
    """
    import os
    
    print("="*80)
    print("W&B HYPERPARAMETER SWEEP")
    print("="*80)
    
    # Get project and entity from environment or use defaults
    project = os.environ.get('WANDB_PROJECT', 'multimodal-ad')
    entity = os.environ.get('WANDB_ENTITY', 'shamussim')
    
    print(f"W&B Project: {project}")
    print(f"W&B Entity: {entity}")
    print()
    
    # Create sweep config
    if quick:
        sweep_config = create_quick_sweep_config()
        print("Using quick sweep configuration")
    else:
        sweep_config = create_sweep_config()
        print("Using full sweep configuration")
    
    print(f"Number of trials: {n_trials}")
    print(f"Optimization method: {sweep_config['method']}")
    print(f"Target metric: {sweep_config['metric']['name']}")
    print()
    
    # Create sweep
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=project,
        entity=entity
    )
    
    print(f"Sweep created with ID: {sweep_id}")
    print(f"Sweep URL: https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}")
    print()
    
    # Run sweep
    print("Starting sweep...")
    wandb.agent(
        sweep_id=sweep_id,
        function=train_with_sweep,
        count=n_trials,
        project=project,
        entity=entity
    )
    
    print("Sweep completed!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        create_and_run_sweep(n_trials=2, quick=True)
    else:
        create_and_run_sweep(n_trials=30, quick=False)
