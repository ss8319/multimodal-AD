# Patch Embedding Spatial Mapping in BrainIAC

## Overview

Yes, **each patch embedding corresponds to a unique spatial region** in the 3D MRI image. Here's how it works:

## Spatial Correspondence

### Image Division

For a **96×96×96** MRI image with **16×16×16** patch size:

```
Image: [96, 96, 96] voxels
Patch size: [16, 16, 16] voxels
Patches per dimension: 96 / 16 = 6
Total patches: 6 × 6 × 6 = 216 (theoretically)
Actual patches: 215 (from MONAI ViT)
```

### Patch-to-Region Mapping

Each patch embedding corresponds to a **non-overlapping 16×16×16 voxel region**:

```
Patch 0:  Voxels [0:16,   0:16,   0:16]   → Embedding[0]
Patch 1:  Voxels [0:16,   0:16,   16:32]  → Embedding[1]
Patch 2:  Voxels [0:16,   0:16,   32:48]  → Embedding[2]
...
Patch 215: Voxels [80:96, 80:96, 80:96]  → Embedding[215]
```

### Spatial Arrangement

The patches are arranged in a **3D grid**:

```
Z (depth)
  ↑
  │  ┌─────┬─────┬─────┐
  │  │  0  │  1  │  2  │  Y (height)
  │  ├─────┼─────┼─────┤  ↑
  │  │  3  │  4  │  5  │  │
  │  ├─────┼─────┼─────┤  │
  │  │  6  │  7  │  8  │  │
  └──┴─────┴─────┴─────┘  │
                          │
                          └──→ X (width)
```

## Important Nuance: Initial vs Final Embeddings

### 1. **Initial Patch Embeddings** (Input to Transformer)
- **Direct spatial correspondence**: Each embedding directly represents its 16×16×16 voxel region
- **No cross-patch information**: Only contains information from that specific region
- **Shape**: `[215, 768]` - 215 unique spatial regions

### 2. **Final Patch Embeddings** (What We Extract)
- **Still spatially mapped**: Each embedding still corresponds to its original spatial region
- **Enriched with context**: Contains information aggregated from other patches via self-attention
- **Shape**: `[215, 768]` - Same 215 regions, but with global context

## Why This Matters for PCA

When you do **spatial PCA** on patch embeddings:

1. **Each embedding** = One spatial region (16×16×16 voxels)
2. **PCA components** = Principal directions of variation across patches
3. **Spatial visualization** = Shows which brain regions have similar/different features

### Example:

```python
# Patch embeddings: [215, 768]
# After PCA: [215, 3]
# Reshape to spatial: [6, 6, 6, 3]

# Component 1 might highlight:
# - High values in frontal regions (patches 0-50)
# - Low values in occipital regions (patches 150-200)

# This tells you: "Frontal and occipital regions differ along PC1"
```

## Spatial Ordering

The patch embeddings are ordered **spatially**:

```
For 3D grid (D, H, W) = (6, 6, 6):
- Patch index = d * (H * W) + h * W + w
- Where (d, h, w) are the spatial coordinates

Example:
- Patch 0:  (d=0, h=0, w=0) → Top-left-front
- Patch 1:  (d=0, h=0, w=1) → Top-left-center
- Patch 6:  (d=0, h=1, w=0) → Top-center-front
- Patch 36: (d=1, h=0, w=0) → Middle-left-front
```

## Verification

You can verify spatial correspondence by:

1. **Visualizing patch locations**: Map patch indices back to voxel coordinates
2. **Checking attention maps**: See which patches attend to which spatial regions
3. **Upsampling PCA components**: Reshape `[6,6,6,3]` → `[96,96,96,3]` to see spatial patterns

## Summary

✅ **Yes, each patch embedding corresponds to a unique 16×16×16 voxel region**

✅ **The spatial mapping is preserved** even after transformer processing

✅ **For PCA analysis**, you're analyzing variation across these spatial regions

✅ **Spatial visualization** shows which brain regions contribute to each principal component

This is why patch embeddings are perfect for **spatial XAI** - you can visualize which brain regions are important for the model's decisions!

