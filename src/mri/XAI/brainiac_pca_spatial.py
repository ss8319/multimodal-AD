import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import nibabel as nib

def main():
    # --- Configuration ---
    base_path_features = '/home/ssim0068/multimodal-AD/src/mri/XAI/features'
    base_path_images = '/home/ssim0068/data/multimodal-dataset/all_icbm/images'
    subject_id = '002_S_0413'
    patch_embedding_file = os.path.join(base_path_features, f'{subject_id}_patch_embeddings.npy')
    image_file = os.path.join(base_path_images, f'{subject_id}.nii.gz')
    output_plot = os.path.join(base_path_features, f'{subject_id}_spatial_pca.png')

    print("=" * 70)
    print(f"Spatial PCA Analysis for Subject: {subject_id}")
    print("=" * 70)

    # --- 1. Load Original MRI Image ---
    print(f"Loading MRI image: {image_file}")
    try:
        img_nii = nib.load(image_file)
        img_data = img_nii.get_fdata()
        print(f"Image shape: {img_data.shape}")
    except Exception as e:
        print(f"Error loading MRI image: {e}")
        return

    # --- 2. Load Patch Embeddings ---
    print(f"Loading patch embeddings: {patch_embedding_file}")
    try:
        patch_embeddings = np.load(patch_embedding_file)
        n_patches, feature_dim = patch_embeddings.shape
        print(f"Loaded: {n_patches} patches, {feature_dim} features per patch")
    except Exception as e:
        print(f"Error loading patch embeddings: {e}")
        return

    # --- 3. Infer Spatial Dimensions (Handle 215 vs 216) ---
    expected_patches = 216 # 6x6x6
    spatial_dims = (6, 6, 6)

    if n_patches != expected_patches:
        print(f"Note: Patch count {n_patches} != expected {expected_patches}. Adjusting...")
        if n_patches < expected_patches:
            # Pad with zeros
            padding = np.zeros((expected_patches - n_patches, feature_dim))
            patch_embeddings_adjusted = np.vstack([patch_embeddings, padding])
        else:
            # Truncate
            patch_embeddings_adjusted = patch_embeddings[:expected_patches]
    else:
        patch_embeddings_adjusted = patch_embeddings

    # --- 4. Perform PCA ---
    print("\nPerforming PCA...")
    scaler = StandardScaler()
    patches_scaled = scaler.fit_transform(patch_embeddings_adjusted)

    pca = PCA(n_components=3, random_state=42, whiten=True)
    patches_pca = pca.fit_transform(patches_scaled)
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

    # --- 5. Reshape to Spatial Grid ---
    # Reshape to (D, H, W, 3)
    pca_spatial = patches_pca.reshape(*spatial_dims, 3)
    print(f"Spatial PCA shape: {pca_spatial.shape}")

    # Normalize PCA components to [0, 1] for visualization
    pca_spatial_norm = np.zeros_like(pca_spatial)
    for i in range(3):
        comp = pca_spatial[..., i]
        pca_spatial_norm[..., i] = (comp - comp.min()) / (comp.max() - comp.min() + 1e-8)

    # --- 6. Visualization ---
    print("\nVisualizing middle slices...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Middle slice indices
    slice_idx_img = img_data.shape[2] // 2
    slice_idx_pca = spatial_dims[2] // 2

    # 1. Original MRI (Middle Slice)
    # Rotate 90 degrees to align with common view if needed, depending on orientation
    axes[0].imshow(np.rot90(img_data[:, :, slice_idx_img]), cmap='gray')
    axes[0].set_title(f"Original MRI\n(Slice {slice_idx_img})", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # 2-4. PCA Components (Middle Slice)
    cmaps = ['Reds', 'Blues', 'Greens']
    for i in range(3):
        pca_slice = np.rot90(pca_spatial_norm[:, :, slice_idx_pca, i])
        im = axes[i+1].imshow(pca_slice, cmap=cmaps[i], vmin=0, vmax=1) # extent can be added here to match aspect ratio
        axes[i+1].set_title(f"Component {i+1}\n({pca.explained_variance_ratio_[i]:.1%} var)", 
                           fontsize=12, fontweight='bold')
        axes[i+1].axis('off')
        plt.colorbar(im, ax=axes[i+1], fraction=0.046)

    plt.suptitle(f'Spatial PCA Analysis: Subject {subject_id}\n(Mapping PCA components to brain regions)', 
                 fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    print(f"Saving plot to {output_plot}")
    plt.savefig(output_plot, bbox_inches='tight', dpi=300)
    plt.close()
    print("Done!")

if __name__ == "__main__":
    main()

