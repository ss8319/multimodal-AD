# XAI Methods Comparison: PCA Feature Maps vs Attention Visualization

## Overview
This document compares two XAI methods for understanding region contributions towards diagnosis across multiple SSL foundation models (DINOv3, MAE, JEPA, SimCLR).

---

## Method 1: PCA Feature Maps over Patch Embeddings

### How It Works
1. Extract patch embeddings from the model (spatial features for each image region)
2. Apply PCA to reduce dimensionality and find principal directions of variation
3. Map principal components back to spatial locations
4. Visualize spatial patterns of the principal components

### Pros ✅

1. **Universal Compatibility**
   - ✅ Works with **ANY** model architecture that produces embeddings
   - ✅ Compatible with: ViT, CNN, ResNet, ConvNeXt, etc.
   - ✅ Works with all SSL objectives: MAE, JEPA, SimCLR, DINOv3, SwAV, etc.

2. **Model-Agnostic**
   - No need to modify model architecture
   - No need to hook into specific layers (attention, etc.)
   - Works with black-box models (just need embeddings)

3. **Captures Intrinsic Data Structure**
   - PCA finds principal directions of variation in feature space
   - Reveals natural groupings and patterns in the data
   - Can identify regions with similar feature characteristics

4. **Spatial Interpretability**
   - Direct mapping from feature space to spatial locations
   - Shows which brain regions have similar/dissimilar representations
   - Can identify spatial patterns (e.g., left-right symmetry, anatomical structures)

5. **Robust to Model Differences**
   - Same analysis pipeline works across different models
   - Allows fair comparison between models
   - Can aggregate across multiple models

6. **No Model-Specific Implementation**
   - Single codebase works for all models
   - Easier to maintain and extend

### Cons ❌

1. **Post-Hoc Analysis**
   - Requires additional computation after feature extraction
   - Not captured during inference (offline analysis)

2. **May Not Reflect Decision-Making**
   - PCA finds variance in features, not necessarily what model uses for decisions
   - Principal components might not align with diagnostic features
   - Could capture noise or irrelevant variations

3. **Requires Multiple Samples (for cross-sample analysis)**
   - Single-sample PCA is less meaningful
   - Better with population-level analysis
   - However, single-sample spatial PCA (like we implemented) works well

4. **Less Direct**
   - Doesn't show what model "attends to" during inference
   - Requires interpretation of principal components

5. **Dimensionality Assumptions**
   - Need to choose number of components
   - May lose information if too few components

---

## Method 2: Attention Visualization

### How It Works
1. Hook into attention layers during forward pass
2. Extract attention weights (typically CLS token attention to patches)
3. Aggregate attention across layers/heads
4. Map attention weights to spatial locations
5. Visualize attention maps as saliency maps

### Pros ✅

1. **Direct Model Introspection**
   - Shows what the model actually attends to during inference
   - Captures model's decision-making process
   - Real-time visualization (during forward pass)

2. **Single-Sample Analysis**
   - Works with individual samples
   - No need for population data
   - Patient-specific explanations

3. **Interpretable Weights**
   - Attention weights directly show importance/contribution
   - Higher attention = more important region
   - Intuitive interpretation

4. **Model-Specific Insights**
   - Tailored to how the model processes information
   - Reveals model-specific patterns

### Cons ❌

1. **Model-Dependent**
   - ❌ **Only works with attention-based models (ViTs)**
   - ❌ **Does NOT work with CNNs, ResNets, ConvNeXt (unless they have attention)**
   - ❌ **Limited to transformer architectures**

2. **Implementation Complexity**
   - Different models store attention differently:
     - MONAI ViT: `att_mat` attribute (if `save_attn=True`)
     - HuggingFace: `outputs.attentions` tuple
     - Custom ViTs: May need to modify forward pass
   - Requires model-specific hooks/wrappers
   - May need to recompute attention if not saved

3. **SSL Objective Compatibility**
   - ✅ **DINOv3**: Works (ViT with attention)
   - ✅ **MAE**: Works (ViT with attention, but masked tokens complicate)
   - ⚠️ **JEPA**: Depends on architecture (may use ViT or CNN)
   - ❌ **SimCLR**: Typically uses ResNet/CNN (no attention) - **WON'T WORK**

4. **Attention May Be Misleading**
   - Attention doesn't always correlate with importance
   - Early layers may attend to low-level features
   - Late layers may attend to high-level abstractions
   - CLS token attention may not reflect patch importance

5. **Requires Model Modification**
   - May need to modify forward pass to save attention
   - Need to ensure attention is accessible
   - Can affect model performance/memory

6. **Inconsistent Across Models**
   - Different attention mechanisms (self-attention, cross-attention)
   - Different aggregation strategies needed
   - Hard to compare across models fairly

---

## Model-by-Model Compatibility Analysis

### DINOv3
- **Architecture**: ViT (Vision Transformer)
- **PCA**: ✅ Works perfectly (patch embeddings available)
- **Attention**: ✅ Works (standard ViT attention)

### MAE (Masked Autoencoder)
- **Architecture**: ViT encoder-decoder
- **PCA**: ✅ Works (encoder patch embeddings)
- **Attention**: ⚠️ Works but complicated (masked tokens, need to handle masking)

### JEPA (Joint-Embedding Predictive Architecture)
- **Architecture**: Varies (could be ViT or CNN)
- **PCA**: ✅ Works (if produces embeddings)
- **Attention**: ⚠️ Depends on architecture (may not have attention if CNN-based)

### SimCLR
- **Architecture**: Typically ResNet or CNN backbone
- **PCA**: ✅ Works (can extract spatial features from CNN layers)
- **Attention**: ❌ **DOES NOT WORK** (no attention mechanism in ResNet/CNN)

### BrainIAC (SimCLR-trained ViT)
- **Architecture**: ViT (MONAI implementation)
- **PCA**: ✅ Works (patch embeddings available)
- **Attention**: ✅ Works (MONAI ViT with `save_attn=True`)

---

## Recommendation: **PCA Feature Maps** 🏆

### Why PCA is the Better Universal Method

1. **Universal Compatibility** ⭐⭐⭐⭐⭐
   - Works with **ALL** models and SSL objectives
   - No exceptions - every model produces embeddings/features

2. **Consistent Analysis Pipeline** ⭐⭐⭐⭐⭐
   - Same code works for all models
   - Fair comparison across models
   - Easier to maintain and extend

3. **Captures Spatial Patterns** ⭐⭐⭐⭐⭐
   - Direct mapping to brain regions
   - Identifies regions with similar representations
   - Reveals anatomical structures

4. **Robust and Reliable** ⭐⭐⭐⭐
   - No model-specific implementation needed
   - No need to modify models
   - Works with pre-trained models out-of-the-box

5. **Interpretable for Diagnosis** ⭐⭐⭐⭐
   - Principal components can identify disease-related patterns
   - Can compare healthy vs diseased regions
   - Spatial visualization maps directly to anatomy

### When to Use Attention Visualization

- **Supplementary analysis** for ViT-based models
- **Model-specific insights** when you need to understand attention patterns
- **Real-time visualization** during inference
- **When you only have ViT models** (not mixed architectures)

---

## Hybrid Approach (Best of Both Worlds)

### Recommended Strategy:

1. **Primary Method: PCA Feature Maps**
   - Use for all models (universal)
   - Standardized analysis pipeline
   - Fair comparison across models

2. **Secondary Method: Attention Visualization**
   - Use for ViT-based models only (DINOv3, MAE, BrainIAC)
   - Provides complementary insights
   - Validates PCA findings

3. **Combined Analysis**
   - Compare PCA components with attention maps
   - Identify regions where both methods agree (high confidence)
   - Investigate discrepancies (may reveal interesting patterns)

---

## Implementation Recommendations

### For PCA Feature Maps:
```python
# Universal pipeline for all models
1. Extract patch embeddings (model-specific extraction)
2. Standardize embeddings (StandardScaler)
3. Apply PCA (n_components=3-5)
4. Map to spatial locations
5. Visualize spatial PCA maps
```

### For Attention Visualization:
```python
# Model-specific implementation
1. Check if model has attention (ViT only)
2. Hook into attention layers
3. Extract CLS token attention weights
4. Aggregate across layers/heads
5. Map to spatial locations
6. Visualize attention maps
```

---

## Conclusion

**PCA Feature Maps** is the **universal method** that works best across all models and SSL objectives. It provides:
- ✅ Universal compatibility
- ✅ Consistent analysis
- ✅ Spatial interpretability
- ✅ Fair model comparison
- ✅ Easy implementation

**Attention Visualization** is valuable as a **supplementary method** for ViT-based models, but cannot be used as the primary method due to:
- ❌ Limited to attention-based architectures
- ❌ Incompatible with SimCLR (CNN/ResNet)
- ❌ Model-specific implementation complexity

**Final Recommendation**: Use **PCA Feature Maps as the primary XAI method** for understanding region contributions across all models, with attention visualization as a supplementary method for ViT-based models.

