import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score
import pickle
import os
from typing import Dict, Any, Tuple

def load_model_and_data(experiment_dir: str) -> Tuple[np.ndarray, list, object]:
    """Load the trained model and original data"""
    # Load the trained model
    model_path = os.path.join(experiment_dir, 'best_autoencoder.pth')
    scaler_path = os.path.join(experiment_dir, 'scaler.pkl')
    
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load original data to get protein names
    config_path = os.path.join(experiment_dir, 'config_snapshot.yml')
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load original data
    df = pd.read_csv(config['dataset']['metadata_path'])
    exclude_cols = config['dataset']['exclude_columns']
    protein_cols = [col for col in df.columns if col not in exclude_cols]
    
    return protein_cols, scaler, config

def analyze_feature_contributions(experiment_dir: str, save_path: str = None):
    """Analyze how each of the 8 features relates to original proteins"""
    
    # Load data
    protein_cols, scaler, config = load_model_and_data(experiment_dir)
    
    # Load original data
    df = pd.read_csv(config['dataset']['metadata_path'])
    exclude_cols = config['dataset']['exclude_columns']
    
    # Prepare original data
    X_original = df[protein_cols].values
    X_original = np.nan_to_num(X_original, nan=0.0)
    X_scaled = scaler.transform(X_original)
    
    # Load the trained model
    import torch
    import torch.nn as nn
    from protein_model import ProteinAutoencoder
    
    model = ProteinAutoencoder(input_size=len(protein_cols), hidden_size=8)
    model.load_state_dict(torch.load(os.path.join(experiment_dir, 'best_autoencoder.pth')))
    model.eval()
    
    # Extract encoder weights from both layers
    first_layer_weights = model.encoder[0].weight.data.numpy()  # 320 → 16
    second_layer_weights = model.encoder[4].weight.data.numpy()  # 16 → 8 (bottleneck)
    
    # Calculate the effective weights from input to bottleneck
    # This combines both layers: input → 16 → 8
    effective_weights = np.dot(second_layer_weights, first_layer_weights)  # Shape: (8, 320)
    
    # Analyze each of the 8 features
    feature_analysis = {}
    
    for feature_idx in range(8):
        # Get weights for this feature (from effective weights combining both layers)
        feature_weights = effective_weights[feature_idx, :]
        
        # Create feature importance dataframe
        feature_df = pd.DataFrame({
            'protein': protein_cols,
            'weight': feature_weights,
            'abs_weight': np.abs(feature_weights)
        })
        
        # Sort by absolute weight
        feature_df = feature_df.sort_values('abs_weight', ascending=False)
        
        # Get top contributors (positive and negative)
        top_positive = feature_df[feature_df['weight'] > 0].head(10)
        top_negative = feature_df[feature_df['weight'] < 0].head(10)
        
        feature_analysis[f'feature_{feature_idx+1}'] = {
            'all_weights': feature_df,
            'top_positive': top_positive,
            'top_negative': top_negative,
            'total_contribution': np.sum(np.abs(feature_weights))
        }
    
    return feature_analysis, protein_cols

def plot_feature_interpretation(feature_analysis: Dict, save_path: str):
    """Create comprehensive plots for feature interpretation"""
    
    n_features = len(feature_analysis)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, (feature_name, analysis) in enumerate(feature_analysis.items()):
        ax = axes[i]
        
        # Get top 10 contributors (positive and negative)
        top_positive = analysis['top_positive']
        top_negative = analysis['top_negative']
        
        # Combine for plotting
        plot_data = pd.concat([
            top_positive[['protein', 'weight']].assign(type='Positive'),
            top_negative[['protein', 'weight']].assign(type='Negative')
        ])
        
        # Create horizontal bar plot
        colors = ['green' if x == 'Positive' else 'red' for x in plot_data['type']]
        bars = ax.barh(range(len(plot_data)), plot_data['weight'], color=colors, alpha=0.7)
        
        # Add protein names
        ax.set_yticks(range(len(plot_data)))
        ax.set_yticklabels(plot_data['protein'], fontsize=8)
        
        ax.set_title(f'{feature_name}\nTotal Contribution: {analysis["total_contribution"]:.3f}')
        ax.set_xlabel('Weight')
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Positive'),
            Patch(facecolor='red', alpha=0.7, label='Negative')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_feature_summary_report(feature_analysis: Dict, experiment_dir: str, save_path: str = None):
    """Create a detailed text report of feature interpretations"""
    
    report = []
    report.append("=" * 80)
    report.append("AUTOENCODER FEATURE INTERPRETATION REPORT")
    report.append("=" * 80)
    report.append("")
    
    for feature_name, analysis in feature_analysis.items():
        report.append(f"FEATURE {feature_name.upper()}")
        report.append("-" * 40)
        report.append(f"Total contribution: {analysis['total_contribution']:.3f}")
        report.append("")
        
        # Top positive contributors
        report.append("TOP POSITIVE CONTRIBUTORS (Proteins that increase this feature):")
        for _, row in analysis['top_positive'].head(5).iterrows():
            report.append(f"  {row['protein']}: {row['weight']:.4f}")
        report.append("")
        
        # Top negative contributors
        report.append("TOP NEGATIVE CONTRIBUTORS (Proteins that decrease this feature):")
        for _, row in analysis['top_negative'].head(5).iterrows():
            report.append(f"  {row['protein']}: {row['weight']:.4f}")
        report.append("")
        
        # Biological interpretation hints
        report.append("BIOLOGICAL INTERPRETATION HINTS:")
        positive_proteins = analysis['top_positive']['protein'].tolist()
        negative_proteins = analysis['top_negative']['protein'].tolist()
        
        # Look for common patterns in protein names
        positive_patterns = [p.split('_')[0] if '_' in p else p for p in positive_proteins[:3]]
        negative_patterns = [p.split('_')[0] if '_' in p else p for p in negative_proteins[:3]]
        
        report.append(f"  This feature may represent a balance between:")
        report.append(f"    Positive: {', '.join(positive_patterns)}")
        report.append(f"    Negative: {', '.join(negative_patterns)}")
        report.append("")
        report.append("=" * 80)
        report.append("")
    
    # Save report
    report_text = "\n".join(report)
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
    
    print(report_text)
    return report_text

def analyze_feature_correlations_with_diagnosis(experiment_dir: str):
    """Analyze how each feature correlates with AD diagnosis"""
    
    # Load features and labels
    train_features = np.load(os.path.join(experiment_dir, 'train_features_autoencoder.npy'))
    test_features = np.load(os.path.join(experiment_dir, 'test_features_autoencoder.npy'))
    all_features = np.vstack([train_features, test_features])
    
    # Load labels
    config_path = os.path.join(experiment_dir, 'config_snapshot.yml')
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    df = pd.read_csv(config['dataset']['metadata_path'])
    y = (df['research_group'] == 'AD').astype(int)
    
    # Calculate correlations
    correlations = []
    for i in range(all_features.shape[1]):
        corr = np.corrcoef(all_features[:, i], y)[0, 1]
        correlations.append(corr)
    
    # Create correlation plot
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    feature_names = [f'Feature {i+1}' for i in range(len(correlations))]
    bars = plt.bar(feature_names, correlations, color=['red' if c < 0 else 'blue' for c in correlations])
    plt.title('Feature Correlation with AD Diagnosis')
    plt.ylabel('Correlation Coefficient')
    plt.xticks(rotation=45)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add correlation values on bars
    for bar, corr in zip(bars, correlations):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height > 0 else -0.01),
                f'{corr:.3f}', ha='center', va='bottom' if height > 0 else 'top')
    
    # Feature importance ranking
    plt.subplot(1, 2, 2)
    abs_correlations = np.abs(correlations)
    sorted_indices = np.argsort(abs_correlations)[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_correlations = [correlations[i] for i in sorted_indices]
    
    colors = ['red' if c < 0 else 'blue' for c in sorted_correlations]
    plt.bar(range(len(sorted_features)), sorted_correlations, color=colors)
    plt.title('Feature Importance (by absolute correlation)')
    plt.ylabel('Correlation Coefficient')
    plt.xticks(range(len(sorted_features)), sorted_features, rotation=45)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, 'feature_diagnosis_correlations.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return correlations

def main():
    """Main function to run feature interpretation"""
    # You'll need to provide the experiment directory
    experiment_dir = input("Enter the path to your experiment directory: ").strip()
    
    if not os.path.exists(experiment_dir):
        print(f"Experiment directory not found: {experiment_dir}")
        return
    
    print("Analyzing feature contributions...")
    feature_analysis, protein_cols = analyze_feature_contributions(experiment_dir)
    
    print("Creating feature interpretation plots...")
    plot_feature_interpretation(feature_analysis, 
                               os.path.join(experiment_dir, 'feature_interpretation.png'))
    
    print("Creating feature summary report...")
    create_feature_summary_report(feature_analysis, experiment_dir,
                                 os.path.join(experiment_dir, 'feature_interpretation_report.txt'))
    
    print("Analyzing feature correlations with diagnosis...")
    correlations = analyze_feature_correlations_with_diagnosis(experiment_dir)
    
    print("\nFeature interpretation completed!")
    print(f"Results saved to: {experiment_dir}")

if __name__ == "__main__":
    main()