"""
Hyperparameter sweep for fusion training
Config-driven approach with minimal changes to train_fusion.py
"""

import itertools
import json
import time
from pathlib import Path
from datetime import datetime
import pandas as pd

# Import the main training function
from train_fusion import main


def run_focal_loss_sweep():
    """
    Run hyperparameter sweep for Focal Loss parameters
    """
    print("="*80)
    print("FOCAL LOSS HYPERPARAMETER SWEEP")
    print("="*80)
    
    # Define parameter ranges - optimized for your specific issues
    sweep_config = {
        'focal_alpha': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],      # AD class weight (1.0=no weighting, 3.5=strong)
        'focal_gamma': [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],      # Focusing parameter (0.0=no focusing, 3.5=strong)
        'learning_rate': [0.0005, 0.001, 0.0015], # Learning rate (more conservative range)
    }
    
    # Create all combinations
    param_combinations = list(itertools.product(
        sweep_config['focal_alpha'],
        sweep_config['focal_gamma'], 
        sweep_config['learning_rate']
    ))
    
    print(f"Total combinations: {len(param_combinations)}")
    print(f"Parameters: {sweep_config}")
    print()
    
    # Track results
    results = []
    start_time = time.time()
    
    for i, (alpha, gamma, lr) in enumerate(param_combinations):
        print(f"\n{'='*60}")
        print(f"COMBINATION {i+1}/{len(param_combinations)}")
        print(f"Alpha: {alpha}, Gamma: {gamma}, LR: {lr}")
        print(f"{'='*60}")
        
        # Create config overrides
        config_overrides = {
            'focal_alpha': alpha,
            'focal_gamma': gamma,
            'learning_rate': lr,
            'save_dir': f'/home/ssim0068/multimodal-AD/runs/fusion_sweep_alpha{alpha}_gamma{gamma}_lr{lr}'
        }
        
        # Run training
        try:
            combination_start = time.time()
            main(config_overrides)
            combination_time = time.time() - combination_start
            
            # Load results from the saved file
            results_dir = Path(config_overrides['save_dir'])
            results_file = results_dir / 'aggregated_results.json'
            
            if results_file.exists():
                with open(results_file, 'r') as f:
                    result_data = json.load(f)
                
                # Extract key metrics
                metrics = result_data['aggregated_metrics']
                result_summary = {
                    'combination': i + 1,
                    'alpha': alpha,
                    'gamma': gamma,
                    'lr': lr,
                    'time_minutes': round(combination_time / 60, 2),
                    'acc_mean': metrics.get('test_acc', {}).get('mean', float('nan')),
                    'acc_std': metrics.get('test_acc', {}).get('std', float('nan')),
                    'balanced_acc_mean': metrics.get('test_balanced_acc', {}).get('mean', float('nan')),
                    'balanced_acc_std': metrics.get('test_balanced_acc', {}).get('std', float('nan')),
                    'auc_mean': metrics.get('test_auc', {}).get('mean', float('nan')),
                    'auc_std': metrics.get('test_auc', {}).get('std', float('nan')),
                    'f1_mean': metrics.get('test_f1', {}).get('mean', float('nan')),
                    'f1_std': metrics.get('test_f1', {}).get('std', float('nan')),
                    'precision_mean': metrics.get('test_precision', {}).get('mean', float('nan')),
                    'precision_std': metrics.get('test_precision', {}).get('std', float('nan')),
                    'recall_mean': metrics.get('test_recall', {}).get('mean', float('nan')),
                    'recall_std': metrics.get('test_recall', {}).get('std', float('nan')),
                    'status': 'success'
                }
            else:
                result_summary = {
                    'combination': i + 1,
                    'alpha': alpha,
                    'gamma': gamma,
                    'lr': lr,
                    'time_minutes': round(combination_time / 60, 2),
                    'status': 'no_results_file'
                }
            
            results.append(result_summary)
            print(f"✅ Completed in {combination_time/60:.1f} minutes")
            
        except Exception as e:
            print(f"❌ Error in combination {i+1}: {e}")
            results.append({
                'combination': i + 1,
                'alpha': alpha,
                'gamma': gamma,
                'lr': lr,
                'status': 'error',
                'error': str(e)
            })
    
    # Save results
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETED")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"{'='*60}")
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'/home/ssim0068/multimodal-AD/runs/sweep_results_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump({
            'sweep_config': sweep_config,
            'total_combinations': len(param_combinations),
            'total_time_minutes': round(total_time / 60, 2),
            'results': results
        }, f, indent=2)
    
    # Create summary DataFrame
    df_results = pd.DataFrame(results)
    csv_file = f'/home/ssim0068/multimodal-AD/runs/sweep_results_{timestamp}.csv'
    df_results.to_csv(csv_file, index=False)
    
    print(f"Results saved to: {results_file}")
    print(f"CSV saved to: {csv_file}")
    
    # Print best results
    print_best_results(df_results)
    
    return df_results


def print_best_results(df_results):
    """
    Print the best hyperparameter combinations
    """
    print(f"\n{'='*60}")
    print("BEST RESULTS")
    print(f"{'='*60}")
    
    # Filter successful results
    successful = df_results[df_results['status'] == 'success'].copy()
    
    if len(successful) == 0:
        print("No successful runs found!")
        return
    
    # Sort by different metrics
    metrics_to_rank = ['f1_mean', 'auc_mean', 'balanced_acc_mean', 'acc_mean']
    
    for metric in metrics_to_rank:
        if metric in successful.columns:
            # Sort by metric (descending)
            sorted_results = successful.sort_values(metric, ascending=False)
            
            print(f"\n🏆 TOP 3 BY {metric.upper().replace('_', ' ')}:")
            for i, (_, row) in enumerate(sorted_results.head(3).iterrows()):
                print(f"  {i+1}. Alpha={row['alpha']}, Gamma={row['gamma']}, LR={row['lr']}")
                print(f"     {metric}: {row[metric]:.4f} ± {row[f'{metric.replace("_mean", "_std")}']:.4f}")
                print(f"     Time: {row['time_minutes']:.1f} min")


def run_quick_sweep():
    """
    Run a quick sweep with fewer combinations for testing
    """
    print("="*80)
    print("QUICK FOCAL LOSS SWEEP (TESTING)")
    print("="*80)
    
    # Smaller parameter ranges for quick testing - most promising combinations
    sweep_config = {
        'focal_alpha': [2.5, 3.0],      # Higher values for FP>FN issue
        'focal_gamma': [2.5, 3.0],      # Higher values for precision
        'learning_rate': [0.001],        # Standard learning rate
    }
    
    # Create combinations
    param_combinations = list(itertools.product(
        sweep_config['focal_alpha'],
        sweep_config['focal_gamma'], 
        sweep_config['learning_rate']
    ))
    
    print(f"Quick test combinations: {len(param_combinations)}")
    print(f"Parameters: {sweep_config}")
    
    # Run the sweep (same logic as main sweep)
    results = []
    
    for i, (alpha, gamma, lr) in enumerate(param_combinations):
        print(f"\nQuick test {i+1}/{len(param_combinations)}: Alpha={alpha}, Gamma={gamma}, LR={lr}")
        
        config_overrides = {
            'focal_alpha': alpha,
            'focal_gamma': gamma,
            'learning_rate': lr,
            'save_dir': f'/home/ssim0068/multimodal-AD/runs/quick_sweep_alpha{alpha}_gamma{gamma}_lr{lr}'
        }
        
        try:
            main(config_overrides)
            results.append({'alpha': alpha, 'gamma': gamma, 'lr': lr, 'status': 'success'})
        except Exception as e:
            print(f"Error: {e}")
            results.append({'alpha': alpha, 'gamma': gamma, 'lr': lr, 'status': 'error', 'error': str(e)})
    
    print(f"\nQuick sweep completed!")
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        run_quick_sweep()
    else:
        run_focal_loss_sweep()
