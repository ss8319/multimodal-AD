"""
Simple script to load Neural Network latents and reduce to 4 dimensions using PCA
Includes pairwise scatterplots of PCs colored by AD/CN class
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

# Configuration
RUN_DIR = "src/protein/runs/run_20251002_170038"
LATENTS_DIR = f"{RUN_DIR}/latents/neural_network"
SHOW_DIAGONALS = False  # Set to True to show histograms on diagonal, False to hide them

# 1. Load latents from files
print("Loading latents...")
layer1 = np.load(f"{LATENTS_DIR}/hidden_layer_1.npy")
layer2 = np.load(f"{LATENTS_DIR}/hidden_layer_2.npy")

print(f"Layer 1 shape: {layer1.shape}")
print(f"Layer 2 shape: {layer2.shape}")

# 2. Load class labels directly from saved files if available
print("\nLoading class labels...")
try:
    # First try to load from labels.npy (new approach)
    labels_path = Path(f"{LATENTS_DIR}/labels.npy")
    if labels_path.exists():
        labels_raw = np.load(labels_path, allow_pickle=True)
        class_labels = np.array([1 if label == 'AD' else 0 for label in labels_raw])
        print(f"Loaded labels from {labels_path}")
    else:
        # Fall back to loading from test CSV
        print("Labels file not found, loading from test CSV...")
        test_csv = "src/data/protein/proteomic_encoder_test.csv"
        test_df = pd.read_csv(test_csv)
        class_labels = np.array([1 if label == 'AD' else 0 for label in test_df["research_group"]])
        print(f"Loaded labels from {test_csv}")
except Exception as e:
    print(f"Error loading labels: {e}")
    print("Using dummy labels for demonstration...")
    class_labels = np.zeros(layer1.shape[0], dtype=int)
    class_labels[::2] = 1  # Alternate labels for demonstration

print(f"Class labels shape: {class_labels.shape}")
print(f"Class distribution: AD={np.sum(class_labels==1)}, CN={np.sum(class_labels==0)}")

# 3. Apply PCA to reduce to 4 dimensions
print("\nReducing to 4 dimensions with PCA...")
pca1 = PCA(n_components=4)
pca2 = PCA(n_components=4)

# Fit and transform in one step
layer1_reduced = pca1.fit_transform(layer1)
layer2_reduced = pca2.fit_transform(layer2)

print(f"Layer 1 reduced shape: {layer1_reduced.shape}")
print(f"Layer 2 reduced shape: {layer2_reduced.shape}")

# Show explained variance
print("\nExplained variance by component:")
print("Layer 1:")
for i, var in enumerate(pca1.explained_variance_ratio_):
    print(f"  PC{i+1}: {var:.4f} ({np.sum(pca1.explained_variance_ratio_[:i+1]):.4f} cumulative)")

print("\nLayer 2:")
for i, var in enumerate(pca2.explained_variance_ratio_):
    print(f"  PC{i+1}: {var:.4f} ({np.sum(pca2.explained_variance_ratio_[:i+1]):.4f} cumulative)")

print(f"\nTotal variance explained:")
print(f"  Layer 1: {np.sum(pca1.explained_variance_ratio_):.4f}")
print(f"  Layer 2: {np.sum(pca2.explained_variance_ratio_):.4f}")

# 4. Create pairwise scatterplots of the 4 PCs
def plot_pairwise_pca(reduced_data, class_labels, layer_name, output_file, show_diagonals=True):
    """
    Create pairwise scatterplots of the 4 principal components
    
    Args:
        reduced_data: PCA-reduced data (n_samples, 4)
        class_labels: Class labels (AD=1, CN=0)
        layer_name: Name of the layer for the title
        output_file: Output file path
        show_diagonals: Whether to show histograms on diagonal (default: True)
    """
    # Create figure with subplots for all pairwise combinations
    fig, axes = plt.subplots(4, 4, figsize=(14, 12))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    # Define colors and markers for classes
    colors = {0: 'blue', 1: 'red'}  # CN=0 (blue), AD=1 (red)
    markers = {0: 'o', 1: 's'}      # CN=circle, AD=square
    class_names = {0: 'CN', 1: 'AD'}
    
    # Plot all pairwise combinations
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            
            # On diagonal, show histogram or empty plot
            if i == j:
                if show_diagonals:
                    for class_val in np.unique(class_labels):
                        mask = class_labels == class_val
                        ax.hist(
                            reduced_data[mask, i], 
                            alpha=0.5, 
                            color=colors[class_val], 
                            label=class_names[class_val]
                        )
                    ax.set_title(f"PC{i+1} Distribution")
                    
                    # Only show legend on first diagonal
                    if i == 0:
                        ax.legend()
                else:
                    # Empty diagonal plot
                    ax.set_title(f"PC{i+1}")
                    ax.text(0.5, 0.5, f"PC{i+1}", 
                           transform=ax.transAxes, ha='center', va='center',
                           fontsize=12, alpha=0.5)
            
            # Off diagonal, show scatter
            else:
                for class_val in np.unique(class_labels):
                    mask = class_labels == class_val
                    ax.scatter(
                        reduced_data[mask, j],
                        reduced_data[mask, i],
                        c=colors[class_val],
                        marker=markers[class_val],
                        s=50,
                        alpha=0.7,
                        label=class_names[class_val]
                    )
                
                # Add labels
                ax.set_xlabel(f"PC{j+1}")
                ax.set_ylabel(f"PC{i+1}")
                
                # Add grid
                ax.grid(alpha=0.3)
                
                # Only show legend on first plot
                if i == 1 and j == 0:
                    ax.legend()
    
    # Add overall title
    plt.suptitle(f"Pairwise Scatterplots of PCA Components - {layer_name}", fontsize=16)
    
    # Save figure
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved pairwise PCA plot to: {output_file}")
    
    return fig

# Plot pairwise PCA for both layers
print(f"\nCreating pairwise PCA plots (diagonals: {'ON' if SHOW_DIAGONALS else 'OFF'})...")
plot_pairwise_pca(layer1_reduced, class_labels, "Hidden Layer 1", "layer1_pca_pairwise.png", show_diagonals=SHOW_DIAGONALS)
plot_pairwise_pca(layer2_reduced, class_labels, "Hidden Layer 2", "layer2_pca_pairwise.png", show_diagonals=SHOW_DIAGONALS)

# 5. Create a single plot showing PC1 vs PC2 for both layers
plt.figure(figsize=(12, 6))

# Plot layer 1
plt.subplot(1, 2, 1)
for class_val in np.unique(class_labels):
    mask = class_labels == class_val
    class_name = "AD" if class_val == 1 else "CN"
    color = 'red' if class_val == 1 else 'blue'
    plt.scatter(
        layer1_reduced[mask, 0],
        layer1_reduced[mask, 1],
        c=color,
        label=f"{class_name} (n={np.sum(mask)})",
        alpha=0.7,
        s=50
    )

plt.title("Hidden Layer 1: PC1 vs PC2")
plt.xlabel(f"PC1 ({pca1.explained_variance_ratio_[0]:.2%})")
plt.ylabel(f"PC2 ({pca1.explained_variance_ratio_[1]:.2%})")
plt.grid(alpha=0.3)
plt.legend()

# Plot layer 2
plt.subplot(1, 2, 2)
for class_val in np.unique(class_labels):
    mask = class_labels == class_val
    class_name = "AD" if class_val == 1 else "CN"
    color = 'red' if class_val == 1 else 'blue'
    plt.scatter(
        layer2_reduced[mask, 0],
        layer2_reduced[mask, 1],
        c=color,
        label=f"{class_name} (n={np.sum(mask)})",
        alpha=0.7,
        s=50
    )

plt.title("Hidden Layer 2: PC1 vs PC2")
plt.xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]:.2%})")
plt.ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]:.2%})")
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("nn_layers_pc1_pc2_comparison.png", dpi=150, bbox_inches='tight')
print("Saved PC1 vs PC2 comparison plot to: nn_layers_pc1_pc2_comparison.png")

