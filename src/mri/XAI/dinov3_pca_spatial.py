import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import nibabel as nib
from PIL import Image

def main():
    # --- Configuration ---
    base_path_features = '/home/ssim0068/multimodal-AD/src/mri/XAI/features'
    base_path_images = '/home/ssim0068/data/multimodal-dataset/all_icbm/images'
    subject_id = '002_S_0413'  # Using the first default subject
    
    # Files
    patch_embedding_file = os.path.join(base_path_features, f'{subject_id}_dinov3_patch_embeddings.npy')
    image_file = os.path.join(base_path_images, f'{subject_id}.nii.gz')
    output_plot = os.path.join(base_path_features, f'{subject_id}_dinov3_spatial_pca.png')

    print("=" * 70)
    print(f"DINOv3 2D Spatial PCA Analysis for Subject: {subject_id}")
    print("=" * 70)

    # --- 1. Check if feature file exists ---
    if not os.path.exists(patch_embedding_file):
        print(f"Error: Feature file not found: {patch_embedding_file}")
        print("Please run 'python3 extract_dinov3_features.py' first to generate embeddings.")
        return

    # --- 2. Load Original MRI Image & Extract Slice ---
    print(f"Loading MRI image: {image_file}")
    try:
        img_nii = nib.load(image_file)
        img_data = img_nii.get_fdata()
        print(f"Image shape: {img_data.shape}")
        
        # Extract middle slice (dim 2, same as extraction script default)
        slice_idx = img_data.shape[2] // 2
        slice_2d = img_data[:, :, slice_idx]
        print(f"Extracted middle slice (index {slice_idx}), shape: {slice_2d.shape}")
        
    except Exception as e:
        print(f"Error loading MRI image: {e}")
        return

    # --- 3. Load Patch Embeddings ---
    print(f"Loading patch embeddings: {patch_embedding_file}")
    try:
        patch_embeddings = np.load(patch_embedding_file)
        n_patches, feature_dim = patch_embeddings.shape
        print(f"Loaded: {n_patches} patches, {feature_dim} features per patch")
    except Exception as e:
        print(f"Error loading patch embeddings: {e}")
        return

    # --- 4. Infer Spatial Dimensions ---
    # DINOv3 (ViT-B/16) on 224x224 image produces 14x14 patches = 196 patches
    # DINOv3 often includes 4 register tokens, resulting in 200 tokens if CLS is removed.
    
    target_grid_size = 14
    target_patches = target_grid_size * target_grid_size  # 196
    
    if n_patches == target_patches:
        print(f"Perfect match: {n_patches} patches map to {target_grid_size}x{target_grid_size} grid.")
        spatial_dims = (target_grid_size, target_grid_size)
        patches_for_pca = patch_embeddings
        
    elif n_patches > target_patches:
        excess = n_patches - target_patches
        print(f"Note: Found {n_patches} tokens, expected {target_patches} for 14x14 grid.")
        print(f"Assuming {excess} extra tokens are Register Tokens at the beginning.")
        
        # Register tokens are usually after CLS but before Patches.
        # If input excluded CLS, registers are at indices [0:excess]
        # We want the LAST 196 tokens
        patches_for_pca = patch_embeddings[-target_patches:]
        print(f"Kept last {len(patches_for_pca)} tokens.")
        spatial_dims = (target_grid_size, target_grid_size)
        
    else:
        # Fallback for other grid sizes
        grid_size = int(np.sqrt(n_patches))
        if grid_size * grid_size != n_patches:
            print(f"Warning: Number of patches ({n_patches}) is not a perfect square and not a known DINOv3 pattern.")
        spatial_dims = (grid_size, grid_size)
        patches_for_pca = patch_embeddings

    # --- 5. Perform PCA ---
    print("\nPerforming PCA...")
    scaler = StandardScaler()
    patches_scaled = scaler.fit_transform(patches_for_pca)

    pca = PCA(n_components=3, random_state=42, whiten=True)
    patches_pca = pca.fit_transform(patches_scaled)
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.1%}")

    # --- 6. Reshape to Spatial Grid ---
    # Reshape to (H, W, 3)
    pca_spatial = patches_pca.reshape(*spatial_dims, 3)
    print(f"Spatial PCA shape: {pca_spatial.shape}")

    # Normalize PCA components to [0, 1] for visualization
    pca_spatial_norm = np.zeros_like(pca_spatial)
    for i in range(3):
        comp = pca_spatial[..., i]
        pca_spatial_norm[..., i] = (comp - comp.min()) / (comp.max() - comp.min() + 1e-8)

    # --- 7. Visualization ---
    print("\nVisualizing...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 1. Original MRI Slice
    # Rotate 90 degrees to match typical orientation if needed
    axes[0].imshow(slice_2d, cmap='gray')
    axes[0].set_title(f"Original MRI Slice\n(Slice {slice_idx})", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # 2-4. PCA Components
    # Note: The PCA grid is 14x14, representing the 224x224 resized input.
    # The original slice might be a different size (e.g., 193x229).
    # We display the 14x14 grid directly.
    cmaps = ['Reds', 'Blues', 'Greens']
    for i in range(3):
        # Rotate PCA map to match image orientation assumption
        # (ViT patches usually fill row-by-row)
        pca_slice = pca_spatial_norm[..., i]
        
        # Upsample for smoother visualization? Or keep blocky to show patches?
        # Keeping blocky is more honest to the resolution.
        im = axes[i+1].imshow(pca_slice, cmap=cmaps[i], vmin=0, vmax=1) # interpolation='nearest' by default for small
        
        axes[i+1].set_title(f"Component {i+1}\n({pca.explained_variance_ratio_[i]:.1%} var)", 
                           fontsize=12, fontweight='bold')
        axes[i+1].axis('off')
        plt.colorbar(im, ax=axes[i+1], fraction=0.046)

    plt.suptitle(f'DINOv3 2D Spatial PCA Analysis: Subject {subject_id}\n(14x14 Patch Grid from Middle Slice)', 
                 fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    print(f"Saving plot to {output_plot}")
    plt.savefig(output_plot, bbox_inches='tight', dpi=300)
    plt.close()
    print("Done!")

if __name__ == "__main__":
    main()

