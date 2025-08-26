"""
Main training script for 3D MRI classification.

This script orchestrates the entire training pipeline using the modular structure:
- Loads configuration from YAML files
- Creates datasets and data loaders
- Trains models using the specified architectures
- Evaluates performance and saves results
"""

import pandas as pd
import numpy as np
import yaml
import torch
import torch.nn as nn
from pathlib import Path
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import our modules
try:
    from .config import get_config, create_example_config
    from .dataset import MRIDataset, create_data_loaders, split_dataframe_for_validation
    from .models import create_model, get_available_models, is_bmmae_available
    from .training import train_model
    from .evaluation import evaluate_model, visualize_predictions, save_results_to_csv, plot_training_history
    from .quality_control import DataQualityController, plot_3d_volume_slices
except ImportError:
    # Fallback for direct script execution
    from config import get_config, create_example_config
    from dataset import MRIDataset, create_data_loaders, split_dataframe_for_validation
    from models import create_model, get_available_models, is_bmmae_available
    from training import train_model
    from evaluation import evaluate_model, visualize_predictions, save_results_to_csv, plot_training_history
    from quality_control import DataQualityController, plot_3d_volume_slices

def run_quality_control_only(config):
    """Run comprehensive QC checks without training."""
    print("\n🔍 Running Quality Control Analysis")
    print("=" * 60)
    
    # Load data splits
    print("📊 Loading data splits...")
    splits_folder = Path(config['data']['splits_folder'])
    train_df = pd.read_csv(splits_folder / "train_split.csv")
    test_df = pd.read_csv(splits_folder / "test_split.csv")
    
    adni_base_path = config['data']['adni_base_path']
    target_size = tuple(config['training']['target_size'])
    
    print(f"✅ Data loaded: {len(train_df)} train, {len(test_df)} test samples")
    
    # Create QC output directory
    qc_output_dir = Path(config['output_dir']) / "quality_control_analysis"
    qc_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize QC controller
    qc_controller = DataQualityController(save_plots=True, output_dir=qc_output_dir)
    
    # Create datasets for QC
    print("\n🔧 Creating datasets for QC...")
    train_dataset = MRIDataset(
        train_df, adni_base_path, target_size=target_size,
        augment=False,  # No augmentation for QC
        normalization_method=config['data']['normalization_method'],
        percentile_range=tuple(config['data']['percentile_range']),
        max_files_per_subject=config['data']['max_files_per_subject']
    )
    
    test_dataset = MRIDataset(
        test_df, adni_base_path, target_size=target_size,
        augment=False,
        normalization_method=config['data']['normalization_method'],
        percentile_range=tuple(config['data']['percentile_range']),
        max_files_per_subject=config['data']['max_files_per_subject']
    )
    
    # QC on training data
    print(f"\n📊 Running QC on training data...")
    qc_sample_count = config.get('qc_sample_count', 5)  # More samples for QC-only mode
    train_qc_results = qc_controller.run_comprehensive_qc(
        train_dataset, 
        sample_indices=None,
        max_samples=qc_sample_count
    )
    
    # QC on test data (smaller sample)
    print(f"\n📊 Running QC on test data...")
    test_qc_results = qc_controller.run_comprehensive_qc(
        test_dataset, 
        sample_indices=None,
        max_samples=min(2, len(test_dataset))
    )
    
    # Save comprehensive QC results
    import json
    
    comprehensive_qc = {
        'config_used': config,
        'train_qc_results': train_qc_results,
        'test_qc_results': test_qc_results,
        'analysis_timestamp': datetime.now().isoformat(),
        'datasets_info': {
            'train_size': len(train_dataset),
            'test_size': len(test_dataset),
            'target_size': target_size,
            'normalization_method': config['data']['normalization_method']
        }
    }
    
    qc_results_path = qc_output_dir / "comprehensive_qc_results.json"
    with open(qc_results_path, 'w') as f:
        json.dump(comprehensive_qc, f, indent=2, default=str)
    
    print(f"\n✅ Quality Control Analysis Complete!")
    print(f"📁 Results saved to: {qc_output_dir}")
    print(f"📄 Detailed results: {qc_results_path}")
    
    return comprehensive_qc

def main():
    """Main training function."""
    
    print("🧠 3D MRI Classification Training Pipeline")
    print("=" * 60)
    
    # Load configuration and parse CLI arguments
    config = None
    config_path_arg = None
    qc_only_mode = False
    
    argv = sys.argv[1:]
    if "--config" in argv:
        idx = argv.index("--config")
        if idx + 1 < len(argv):
            config_path_arg = argv[idx + 1]
    
    if "--qc-only" in argv:
        qc_only_mode = True
        print("🔍 QC-only mode enabled. Will run quality control checks without training.")
    
    if config_path_arg is not None:
        # Resolve path relative to this script if not absolute
        cfg_path = Path(config_path_arg)
        if not cfg_path.is_absolute():
            cfg_path = Path(__file__).parent / cfg_path
        if cfg_path.exists():
            config = get_config(str(cfg_path))
            print(f"✅ Configuration loaded from {cfg_path}")
        else:
            print(f"⚠️ Config file not found at {cfg_path}. Using defaults.")
            config = get_config(None)
    else:
        print("📋 No --config provided. Using defaults.")
        config = get_config(None)
    
    # Ensure output directory exists
    try:
        out_dir = Path(config['output_dir'])
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"❌ Failed to create output directory {config.get('output_dir')}: {e}")
        return 1

    # Save a snapshot of the effective config to the output directory
    try:
        config_snapshot_path = out_dir / 'config_used.yaml'
        with open(config_snapshot_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        print(f"📄 Saved config snapshot to: {config_snapshot_path}")
    except Exception as e:
        print(f"⚠️ Failed to save config snapshot: {e}")

    # Print configuration summary
    print_config_summary(config)
    
    # Handle QC-only mode
    if qc_only_mode:
        config['enable_quality_control'] = True  # Force enable QC
        config['qc_sample_count'] = config.get('qc_sample_count', 5)  # More samples for QC-only
        return run_quality_control_only(config)
    
    # Check BM-MAE availability
    if not is_bmmae_available():
        print("⚠️ BM-MAE not available - BM-MAE models will be skipped")
        # Filter out BM-MAE methods
        config['training']['methods'] = [m for m in config['training']['methods'] if not m.startswith('bmmae')]
        if not config['training']['methods']:
            print("❌ No valid methods available after filtering")
            return 1
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    # Load data splits
    print(f"\n📊 Loading data splits...")
    try:
        splits_folder = Path(config['data']['splits_folder'])
        adni_base_path = config['data']['adni_base_path']
        
        train_df = pd.read_csv(splits_folder / "train_split.csv")
        test_df = pd.read_csv(splits_folder / "test_split.csv")
        
        print(f"✅ Data loaded successfully:")
        print(f"   • Training samples: {len(train_df)}")
        print(f"   • Test samples: {len(test_df)}")
        
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return 1
    
    # Create datasets/test loader (test set is fixed)
    print(f"\n🔧 Preparing datasets...")
    target_size = tuple(config['training']['target_size'])
    # Show test class distribution
    try:
        test_counts = test_df['Group'].value_counts().to_dict()
        print(f"📊 Test class distribution: {test_counts}")
    except Exception:
        pass
    test_dataset = MRIDataset(
        test_df,
        adni_base_path,
        target_size=target_size,
        augment=False,
        normalization_method=config['data']['normalization_method'],
        percentile_range=tuple(config['data']['percentile_range']),
        max_files_per_subject=config['data']['max_files_per_subject']
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )

    # Determine CV folds to run
    cv_indices_path = Path(config['data']['splits_folder']) / "cv_fold_indices.csv"
    folds_to_run = [1]
    cv_df = None
    use_all_folds = False
    if cv_indices_path.exists():
        import ast
        cv_df = pd.read_csv(cv_indices_path)
        # derive folds
        if 'fold' in cv_df.columns:
            folds_to_run = sorted(cv_df['fold'].unique().tolist())
        else:
            folds_to_run = list(range(1, len(cv_df) + 1))
        use_all_folds = True
        print(f"📚 Detected {len(folds_to_run)} CV folds from cv_fold_indices.csv. Running all folds.")
    else:
        print("⚠️ cv_fold_indices.csv not found. Falling back to single split via train_val_split.")

    # Training results storage
    results = {}

    # Train each selected model
    for method_name in config['training']['methods']:
        print(f"\n🚀 Training {method_name.upper()}")
        print("-" * 60)
        fold_metrics = []
        for fold_id in folds_to_run:
            try:
                # Build fold-specific train/val subsets
                if cv_df is not None:
                    import ast
                    if 'fold' in cv_df.columns:
                        row = cv_df[cv_df['fold'] == fold_id].iloc[0]
                    else:
                        row = cv_df.iloc[fold_id - 1]
                    train_indices = ast.literal_eval(str(row['train_indices']))
                    val_indices = ast.literal_eval(str(row['val_indices']))
                    train_subset = train_df.iloc[train_indices].reset_index(drop=True)
                    val_subset = train_df.iloc[val_indices].reset_index(drop=True)
                    print(f"📊 Using CV fold {fold_id}: train={len(train_subset)}, val={len(val_subset)}")
                else:
                    train_subset, val_subset = split_dataframe_for_validation(
                        train_df, config['training']['train_val_split']
                    )

                # Create datasets and loaders per fold
                try:
                    train_counts = train_subset['Group'].value_counts().to_dict()
                    val_counts = val_subset['Group'].value_counts().to_dict()
                    print(f"   • Train class distribution (fold {fold_id}): {train_counts}")
                    print(f"   • Val class distribution (fold {fold_id}): {val_counts}")
                except Exception:
                    pass
                train_dataset = MRIDataset(
                    train_subset, adni_base_path, target_size=target_size,
                    augment=config['training']['enable_augmentation'],
                    normalization_method=config['data']['normalization_method'],
                    percentile_range=tuple(config['data']['percentile_range']),
                    max_files_per_subject=config['data']['max_files_per_subject']
                )
                val_dataset = MRIDataset(
                    val_subset, adni_base_path, target_size=target_size,
                    augment=False,
                    normalization_method=config['data']['normalization_method'],
                    percentile_range=tuple(config['data']['percentile_range']),
                    max_files_per_subject=config['data']['max_files_per_subject']
                )
                
                # 🔍 Quality Control Checks
                if config.get('enable_quality_control', False):
                    print(f"🔍 Running Quality Control checks for fold {fold_id}...")
                    
                    # Initialize QC controller
                    qc_output_dir = Path(config['output_dir']) / f"qc_fold_{fold_id}"
                    qc_controller = DataQualityController(
                        save_plots=True, 
                        output_dir=qc_output_dir
                    )
                    
                    # Run QC on training data (representative samples)
                    qc_sample_count = config.get('qc_sample_count', 3)
                    print(f"   📊 Checking {qc_sample_count} training samples...")
                    
                    qc_results = qc_controller.run_comprehensive_qc(
                        train_dataset, 
                        sample_indices=None,  # Random selection
                        max_samples=qc_sample_count
                    )
                    
                    # Save QC results
                    import json
                    qc_results_path = qc_output_dir / "qc_results.json"
                    qc_output_dir.mkdir(parents=True, exist_ok=True)
                    with open(qc_results_path, 'w') as f:
                        json.dump(qc_results, f, indent=2)
                    
                    print(f"   💾 QC results saved to: {qc_results_path}")
                    
                    # Only run QC for first fold to avoid too many plots
                    if fold_id == folds_to_run[0]:
                        print("   ✅ QC completed for first fold. Skipping QC for remaining folds to save time.")
                        config['enable_quality_control'] = False
                
                train_loader, val_loader, _ = create_data_loaders(
                    train_dataset, val_dataset, test_dataset,
                    batch_size=config['training']['batch_size']
                )

                # Create model per fold
                print(f"🔧 Creating {method_name} model (fold {fold_id})...")
                model = create_model(
                    method_name,
                    num_classes=2,
                    pretrained_path=config['model']['bmmae_pretrained_path']
                )
                model = model.to(device)

                # Train
                print(f"🎯 Starting training (fold {fold_id})...")
                method_name_with_fold = f"{method_name}_fold{fold_id}"
                trained_model, train_hist, val_hist, best_val_acc = train_model(
                    model, train_loader, val_loader, config, method_name_with_fold
                )

                # Test evaluation
                print(f"\n🔍 Testing {method_name.upper()} (fold {fold_id})...")
                test_loss, test_acc, test_auc, test_preds, test_probs, test_targets = evaluate_model(
                    trained_model, test_loader, nn.CrossEntropyLoss(), device
                )

                fold_metrics.append({'val_acc': best_val_acc, 'test_acc': test_acc, 'test_auc': test_auc})

                # Save model if enabled
                if config['save_model']:
                    model_save_path = Path(config['output_dir']) / f"{method_name_with_fold}_best.pth"
                    torch.save(trained_model.state_dict(), model_save_path)
                    print(f"💾 Model saved to: {model_save_path}")

                # Visualization if enabled
                if config['enable_visualization']:
                    print(f"\n🎨 Creating visualizations for {method_name} (fold {fold_id})...")
                    val_targets, val_preds, val_probs = [], [], []
                    trained_model.eval()
                    with torch.no_grad():
                        for data, target in val_loader:
                            data = data.to(device)
                            logits = trained_model(data)
                            probs = torch.softmax(logits, dim=1)[:, 1]
                            preds = torch.argmax(logits, dim=1)
                            val_targets.extend(target.cpu().tolist())
                            val_preds.extend(preds.cpu().tolist())
                            val_probs.extend(probs.cpu().tolist())
                    viz_save_path = Path(config['output_dir']) / f"validation_results_{method_name_with_fold}.png"
                    visualize_predictions(val_targets, val_preds, val_probs, method_name_with_fold, str(viz_save_path))
                    history_save_path = Path(config['output_dir']) / f"training_history_{method_name_with_fold}.png"
                    plot_training_history(train_hist, val_hist, method_name_with_fold, str(history_save_path))

                # Save detailed results if enabled
                if config['save_results']:
                    results_save_path = Path(config['output_dir']) / f"results_{method_name_with_fold}.csv"
                    save_results_to_csv(
                        test_targets, test_preds, test_probs,
                        str(results_save_path), method_name_with_fold
                    )

            except Exception as e:
                print(f"❌ Failed fold {fold_id} for {method_name}: {e}")
                continue

        # Aggregate across folds
        if fold_metrics:
            val_accs = [m['val_acc'] for m in fold_metrics]
            test_accs = [m['test_acc'] for m in fold_metrics]
            test_aucs = [m['test_auc'] for m in fold_metrics]
            results[method_name] = {
                'best_val_acc': float(np.mean(val_accs)),
                'test_acc': float(np.mean(test_accs)),
                'test_auc': float(np.mean(test_aucs)),
                'val_acc_std': float(np.std(val_accs)),
                'test_acc_std': float(np.std(test_accs)),
                'test_auc_std': float(np.std(test_aucs)),
            }
            print(f"✅ {method_name.upper()} Cross-validated Results (mean±std):")
            print(f"   • Val Acc: {np.mean(val_accs):.1f} ± {np.std(val_accs):.1f}%")
            print(f"   • Test Acc: {np.mean(test_accs):.1f} ± {np.std(test_accs):.1f}%")
            print(f"   • Test AUC: {np.mean(test_aucs):.3f} ± {np.std(test_aucs):.3f}")
    
    # Summary results
    if results:
        print(f"\n📊 FINAL RESULTS SUMMARY")
        print("=" * 80)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            model: {
                'Val_Acc(%)': f"{res['best_val_acc']:.1f}",
                'Test_Acc(%)': f"{res['test_acc']:.1f}", 
                'Test_AUC': f"{res['test_auc']:.3f}"
            }
            for model, res in results.items()
        }).T
        
        print(results_df)
        
        # Save summary results
        if config['save_results']:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_save_path = Path(config['output_dir']) / f"summary_results_{timestamp}.csv"
            results_df.to_csv(summary_save_path)
            print(f"\n💾 Summary results saved to: {summary_save_path}")
        
        print(f"\n💾 All models and results saved to: {config['output_dir']}")
        print("🎉 Training pipeline complete!")
        
    else:
        print("\n❌ No models were successfully trained!")
        return 1
    
    return 0

def print_config_summary(config):
    """Print a summary of the configuration."""
    print(f"\n📋 Configuration Summary:")
    print(f"   • Experiment: {config['experiment_name']}")
    print(f"   • Methods: {config['training']['methods']}")
    print(f"   • Epochs: {config['training']['epochs']}")
    print(f"   • Learning Rate: {config['training']['learning_rate']}")
    print(f"   • Batch Size: {config['training']['batch_size']}")
    print(f"   • Target Size: {tuple(config['training']['target_size'])}")
    print(f"   • Early Stopping: {config['training']['early_stopping_patience']} epochs")
    print(f"   • Augmentation: {'Enabled' if config['training']['enable_augmentation'] else 'Disabled'}")
    print(f"   • Visualization: {'Enabled' if config['enable_visualization'] else 'Disabled'}")
    print(f"   • Output Directory: {config['output_dir']}")

def create_example_config_file():
    """Create an example configuration file."""
    print("📋 Creating example configuration file...")
    config_path = create_example_config("example_config.yaml")
    print(f"✅ Example configuration created: {config_path}")
    print("📝 Edit this file to customize your experiment settings")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--create-config":
            create_example_config_file()
        elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("🧠 3D MRI Classification Training Pipeline")
            print("\nUsage:")
            print("  python main.py                                    # Run with default config")
            print("  python main.py --config path/to/config.yaml       # Run with YAML config")
            print("  python main.py --config path/to/config.yaml --qc-only  # Run QC checks only (no training)")
            print("  python main.py --create-config                    # Create example config file")
            print("  python main.py --help                             # Show this help message")
            print("\nConfiguration:")
            print("  - Provide --config to load a YAML config")
            print("  - Use --create-config to generate an example config file next to this script")
            print("  - Use --qc-only to run comprehensive quality control without training")
        else:
            # Allow other args (e.g., --config) to be handled inside main()
            sys.exit(main())
    else:
        sys.exit(main())
