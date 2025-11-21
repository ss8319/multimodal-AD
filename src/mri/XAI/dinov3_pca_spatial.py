import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import nibabel as nib
from PIL import Image
from scipy.ndimage import zoom

def main():
    # --- Configuration ---
    base_path_features = '/home/ssim0068/multimodal-AD/src/mri/XAI/features'
    base_path_images = '/home/ssim0068/data/multimodal-dataset/all_icbm/images'
    subject_id = '007_S_1206'  # Using the first default subject
    
    # Files
    patch_embedding_file = os.path.join(base_path_features, f'{subject_id}_dinov3_patch_embeddings.npy')
    image_file = os.path.join(base_path_images, f'{subject_id}.nii.gz')
    output_plot = os.path.join(base_path_features, f'{subject_id}_dinov3_spatial_pca_foreground.png')

    print("=" * 70)
    print(f"DINOv3 2D Spatial PCA Analysis (Foreground Only) for Subject: {subject_id}")
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

    # --- 5. Generate Foreground Mask ---
    print("\nGenerating Foreground Mask...")
    # Resize slice to 14x14 to match patch grid
    # Input image was resized to 224x224, then patchified to 14x14 (16x16 patches)
    # So we need to resize original slice to 14x14 directly to approximate patch content
    
    # Normalize slice to [0,1]
    slice_norm = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8)
    
    # Use PIL to resize (consistent with extraction)
    slice_pil = Image.fromarray((slice_norm * 255).astype(np.uint8))
    # Resize directly to grid size (14x14) to estimate patch intensity
    mask_pil = slice_pil.resize((target_grid_size, target_grid_size), Image.BILINEAR)
    mask_arr = np.array(mask_pil) / 255.0
    
    # Threshold: Since skull stripped, background is 0. Let's use a small threshold.
    fg_mask = mask_arr > 0.05
    fg_mask_flat = fg_mask.flatten()
    
    print(f"Foreground patches: {np.sum(fg_mask_flat)} / {len(fg_mask_flat)}")
    
    if np.sum(fg_mask_flat) < 10:
        print("Warning: Too few foreground patches found. Using all patches instead.")
        fg_mask_flat[:] = True

    # --- 6. Perform PCA (Foreground Only) ---
    print("\nPerforming PCA on Foreground patches...")
    scaler = StandardScaler()
    
    # Fit scaler and PCA ONLY on foreground patches
    fg_patches = patches_for_pca[fg_mask_flat]
    patches_scaled_fg = scaler.fit_transform(fg_patches)

    pca = PCA(n_components=3, random_state=42, whiten=True)
    pca.fit(patches_scaled_fg)
    
    print(f"Explained variance ratio (Foreground): {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.1%}")
    
    # Transform ALL patches using the foreground-fitted PCA
    # (We need to transform background patches too just to place them in the grid, even if masked later)
    all_patches_scaled = scaler.transform(patches_for_pca)
    patches_pca_all = pca.transform(all_patches_scaled)

    # --- 7. Reshape to Spatial Grid ---
    # Reshape to (H, W, 3)
    pca_spatial = patches_pca_all.reshape(*spatial_dims, 3)
    print(f"Spatial PCA shape: {pca_spatial.shape}")

    # Normalize PCA components to [0, 1] for visualization
    pca_spatial_norm = np.zeros_like(pca_spatial)
    for i in range(3):
        comp = pca_spatial[..., i]
        # Normalize based on FOREGROUND range only to keep contrast in brain
        # But clip background outliers
        fg_comp = comp[fg_mask]
        vmin, vmax = fg_comp.min(), fg_comp.max()
        
        pca_spatial_norm[..., i] = (comp - vmin) / (vmax - vmin + 1e-8)
        pca_spatial_norm[..., i] = np.clip(pca_spatial_norm[..., i], 0, 1)
        
    # Apply mask to visualization (set background to Black)
    for i in range(3):
        pca_spatial_norm[..., i][~fg_mask] = 0.0

    # --- 8. Visualization ---
    print("\nVisualizing...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 1. Original MRI Slice
    axes[0].imshow(slice_2d, cmap='gray')
    axes[0].set_title(f"Original MRI Slice\n(Slice {slice_idx})", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # 2-4. PCA Components
    cmaps = ['Reds', 'Blues', 'Greens']
    for i in range(3):
        pca_slice = pca_spatial_norm[..., i]
        
        # Nearest neighbor interpolation to show grid clearly
        im = axes[i+1].imshow(pca_slice, cmap=cmaps[i], vmin=0, vmax=1, interpolation='nearest')
        
        axes[i+1].set_title(f"Component {i+1}\n({pca.explained_variance_ratio_[i]:.1%} var)", 
                           fontsize=12, fontweight='bold')
        axes[i+1].axis('off')
        plt.colorbar(im, ax=axes[i+1], fraction=0.046)

    plt.suptitle(f'DINOv3 2D Spatial PCA (Foreground Only): Subject {subject_id}\n(PCA fit on brain tissue only)', 
                 fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    print(f"Saving plot to {output_plot}")
    plt.savefig(output_plot, bbox_inches='tight', dpi=300)
    plt.close()
    print("Done!")

if __name__ == "__main__":
    main()

