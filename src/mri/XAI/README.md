# XAI: Vision Transformer Analysis for MRI

This directory contains tools for visualizing and analyzing Vision Transformer features (BrainIAC and DINOv3) on 3D MRI data.

## Scripts

### 1. DINOv3 Feature Extraction
`extract_dinov3_features.py`
Extracts patch embeddings from the middle slice of 3D MRI scans using the DINOv3 model.

**Usage:**
```bash
# Run on default ICBM images
python extract_dinov3_features.py

# Run on specific images
python extract_dinov3_features.py --images /path/to/img1.nii.gz /path/to/img2.nii.gz --output_dir features/
```
**Output:** Saves `.npy` files containing feature vectors (e.g., 768-dim) for each 16x16 patch.

### 2. DINOv3 Spatial PCA
`dinov3_pca_spatial.py`
Performs Principal Component Analysis (PCA) on the extracted DINOv3 patch embeddings to visualize the dominant features in the image.

**Key Features:**
*   **Foreground-Only PCA:** Fits PCA only on brain tissue patches (ignoring black background) to maximize contrast.
*   **Spatial Mapping:** Reshapes feature vectors back into a 2D grid to show where specific features are located.

**Usage:**
```bash
# Before running, ensure you have generated features using the extraction script above.
# The script currently has hardcoded paths for a demo subject (007_S_1206).
python dinov3_pca_spatial.py
```
**Output:** Generates a PNG visualization showing the original slice and the top 3 PCA components (Red, Blue, Green channels).

