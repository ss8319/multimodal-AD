"""
Visualize attention weights from the Protein Attention Pooling model.

This script loads a trained model and visualizes which proteins receive
the highest attention for AD vs CN classification.
"""
import argparse
import pickle
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from model import ProteinAttentionPoolingClassifier
from dataset import prepare_features


def load_model_and_data(run_dir, test_data_path):
    """Load trained model, scaler, and test data"""
    run_path = Path(run_dir)
    
    # Load model
    model_path = run_path / 'models' / 'protein_attention_pooling.pth'
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load scaler
    scaler_path = run_path / 'scaler.pkl'
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load feature names
    features_path = run_path / 'scaler_features.json'
    with open(features_path, 'r') as f:
        feature_cols = json.load(f)
    
    # Load test data
    df = pd.read_csv(test_data_path)
    X, y, feature_cols_actual, label_map = prepare_features(
        df, 
        label_col='research_group',
        id_col='RID'
    )
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Load model
    model = ProteinAttentionPoolingClassifier()
    model.model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.model.eval()
    
    return model, X_scaled, y, feature_cols, label_map


def get_attention_weights(model, X):
    """Get attention weights for all samples"""
    X_tensor = torch.FloatTensor(X)
    
    with torch.no_grad():
        # Forward pass
        _ = model.model(X_tensor, return_attention=False)
        # Get stored attention weights
        attention_weights = model.model.last_attention_weights
    
    return attention_weights.numpy()


def plot_mean_attention_by_class(attention_weights, y, feature_names, output_path=None, top_k=20):
    """Plot mean attention weights for AD vs CN"""
    # Separate by class
    ad_mask = (y == 1)
    cn_mask = (y == 0)
    
    ad_attention = attention_weights[ad_mask].mean(axis=0)
    cn_attention = attention_weights[cn_mask].mean(axis=0)
    
    # Get top proteins by difference
    attention_diff = ad_attention - cn_attention
    top_indices = np.argsort(np.abs(attention_diff))[-top_k:][::-1]
    
    # Create dataframe for plotting
    plot_data = []
    for idx in top_indices:
        plot_data.append({
            'Protein': feature_names[idx],
            'AD': ad_attention[idx],
            'CN': cn_attention[idx],
            'Difference': attention_diff[idx]
        })
    
    df_plot = pd.DataFrame(plot_data)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Side-by-side comparison
    x = np.arange(len(df_plot))
    width = 0.35
    
    axes[0].barh(x - width/2, df_plot['AD'], width, label='AD', alpha=0.8, color='#d62728')
    axes[0].barh(x + width/2, df_plot['CN'], width, label='CN', alpha=0.8, color='#1f77b4')
    axes[0].set_yticks(x)
    axes[0].set_yticklabels(df_plot['Protein'], fontsize=9)
    axes[0].set_xlabel('Mean Attention Weight', fontsize=12)
    axes[0].set_title(f'Top {top_k} Proteins by Attention (AD vs CN)', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='x', alpha=0.3)
    axes[0].invert_yaxis()
    
    # Plot 2: Difference plot
    colors = ['#d62728' if d > 0 else '#1f77b4' for d in df_plot['Difference']]
    axes[1].barh(x, df_plot['Difference'], color=colors, alpha=0.8)
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(df_plot['Protein'], fontsize=9)
    axes[1].set_xlabel('Attention Difference (AD - CN)', fontsize=12)
    axes[1].set_title(f'Attention Difference: AD-enriched (red) vs CN-enriched (blue)', fontsize=14, fontweight='bold')
    axes[1].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[1].grid(axis='x', alpha=0.3)
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return df_plot


def plot_attention_heatmap(attention_weights, y, feature_names, output_path=None, top_k=30):
    """Plot heatmap of attention weights for all samples"""
    # Get top proteins by variance
    attention_var = attention_weights.var(axis=0)
    top_indices = np.argsort(attention_var)[-top_k:][::-1]
    
    # Create heatmap data
    heatmap_data = attention_weights[:, top_indices]
    
    # Sort samples by class
    sort_idx = np.argsort(y)
    heatmap_data = heatmap_data[sort_idx]
    y_sorted = y[sort_idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot heatmap
    sns.heatmap(
        heatmap_data.T,
        cmap='YlOrRd',
        cbar_kws={'label': 'Attention Weight'},
        yticklabels=[feature_names[i] for i in top_indices],
        xticklabels=False,
        ax=ax
    )
    
    # Add class labels on top
    class_colors = ['#1f77b4' if label == 0 else '#d62728' for label in y_sorted]
    for i, color in enumerate(class_colors):
        ax.add_patch(plt.Rectangle((i, -0.5), 1, 0.5, color=color, clip_on=False))
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='CN'),
        Patch(facecolor='#d62728', label='AD')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1.0))
    
    ax.set_xlabel('Samples (sorted by class)', fontsize=12)
    ax.set_ylabel('Proteins', fontsize=12)
    ax.set_title(f'Attention Weights Heatmap: Top {top_k} Most Variable Proteins', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def save_attention_statistics(attention_weights, y, feature_names, output_path):
    """Save detailed attention statistics to CSV"""
    # Compute statistics
    ad_mask = (y == 1)
    cn_mask = (y == 0)
    
    stats = []
    for i, name in enumerate(feature_names):
        ad_attention = attention_weights[ad_mask, i]
        cn_attention = attention_weights[cn_mask, i]
        
        stats.append({
            'protein': name,
            'mean_attention_AD': ad_attention.mean(),
            'std_attention_AD': ad_attention.std(),
            'mean_attention_CN': cn_attention.mean(),
            'std_attention_CN': cn_attention.std(),
            'difference_AD_CN': ad_attention.mean() - cn_attention.mean(),
            'abs_difference': abs(ad_attention.mean() - cn_attention.mean()),
            'overall_mean': attention_weights[:, i].mean(),
            'overall_std': attention_weights[:, i].std(),
        })
    
    df_stats = pd.DataFrame(stats)
    df_stats = df_stats.sort_values('abs_difference', ascending=False)
    df_stats.to_csv(output_path, index=False)
    print(f"Saved attention statistics to: {output_path}")
    
    return df_stats


def main():
    parser = argparse.ArgumentParser(description='Visualize attention weights from Protein Attention Pooling model')
    parser.add_argument('--run-dir', type=str, required=True,
                        help='Path to run directory containing trained model')
    parser.add_argument('--test-data', type=str, required=True,
                        help='Path to test data CSV')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save plots (default: run_dir/attention_analysis)')
    parser.add_argument('--top-k', type=int, default=20,
                        help='Number of top proteins to show in plots')
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir is None:
        output_dir = Path(args.run_dir) / 'attention_analysis'
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("ATTENTION WEIGHT VISUALIZATION")
    print("=" * 70)
    
    # Load model and data
    print("\n[1/4] Loading model and data...")
    model, X_scaled, y, feature_names, label_map = load_model_and_data(
        args.run_dir, args.test_data
    )
    print(f"   Loaded {len(X_scaled)} samples with {len(feature_names)} features")
    print(f"   Class distribution: AD={sum(y==1)}, CN={sum(y==0)}")
    
    # Get attention weights
    print("\n[2/4] Computing attention weights...")
    attention_weights = get_attention_weights(model, X_scaled)
    print(f"   Attention weights shape: {attention_weights.shape}")
    print(f"   Mean attention: {attention_weights.mean():.6f}")
    print(f"   Std attention: {attention_weights.std():.6f}")
    
    # Plot mean attention by class
    print(f"\n[3/4] Creating visualizations (top {args.top_k} proteins)...")
    plot_path = output_dir / f'attention_by_class_top{args.top_k}.png'
    df_top = plot_mean_attention_by_class(
        attention_weights, y, feature_names, 
        output_path=plot_path, top_k=args.top_k
    )
    
    # Plot heatmap
    heatmap_path = output_dir / 'attention_heatmap.png'
    plot_attention_heatmap(
        attention_weights, y, feature_names,
        output_path=heatmap_path, top_k=30
    )
    
    # Save statistics
    print("\n[4/4] Saving attention statistics...")
    stats_path = output_dir / 'attention_statistics.csv'
    df_stats = save_attention_statistics(
        attention_weights, y, feature_names, stats_path
    )
    
    # Print top proteins
    print("\n" + "=" * 70)
    print("TOP 10 PROTEINS BY ATTENTION DIFFERENCE (AD vs CN)")
    print("=" * 70)
    print(df_top.head(10).to_string(index=False))
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - {plot_path.name}")
    print(f"  - {heatmap_path.name}")
    print(f"  - {stats_path.name}")


if __name__ == '__main__':
    main()

