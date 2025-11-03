# Unimodal Baselines for Multimodal Fusion

This directory contains scripts to evaluate unimodal baselines (sMRI-only and protein-only) on the same cross-validation splits used for multimodal fusion experiments.

## Purpose

Establish baseline performance for individual modalities to compare against fusion models:
- **sMRI-only**: DINOv3 vision transformer + linear probe OR BrainIAC 3D CNN
- **Protein-only**: Pre-trained MLP/Transformer models

---

## DINOv3 sMRI-Only Baseline

### Overview

The DINOv3 baseline uses:
1. **Pretrained DINOv3 ViT-S/16** frozen backbone (trained on 142M images)
2. **Slice-wise aggregation**: Extract 2D slices from 3D MRI volumes, process through DINOv3, aggregate features
3. **Sklearn LogisticRegression** trained on ADNI dataset (separate from multimodal data to avoid leakage)
4. **Inference-only** on multimodal test sets per fold

### Key Differences from BrainIAC Baseline

| Aspect | BrainIAC | DINOv3 |
|--------|----------|---------|
| Architecture | 3D CNN (native) | 2D ViT + slice aggregation |
| Pretraining | None | 142M natural images |
| Training | PyTorch Lightning | Sklearn (on ADNI) |
| Input | Full 3D volume | 2D slices (aggregated) |

### Files

- `dinov3_linear_probe_baseline.py`: Main evaluation script (refactored to use DINOv3 dataset infrastructure)
- `run_dinov3_baseline.sh`: SLURM job script

### Refactored Architecture (October 2025)

The baseline now leverages the **exact same preprocessing pipeline** as DINOv3 linear probe experiments:

1. **Temporary CSV Conversion**: Converts multimodal CSV to ADNI format per fold
   - Extracts `pat_id` from `mri_path` filename
   - Converts `research_group` to binary `label`
   
2. **Dataset Creation via `make_dataset()`**: 
   - Uses DINOv3's `loaders.py:make_dataset()` 
   - Automatically wraps ADNI dataset with `SliceAggregationDataset`
   - Applies MONAI pipeline: LoadImaged → EnsureChannelFirstd → Orientationd → ScaleIntensityRangePercentilesd
   
3. **Feature Extraction via `extract_features()`**:
   - Uses DINOv3's `eval/utils.py:extract_features()`
   - Automatically detects `SliceAggregationDataset` and routes to slice-wise aggregation
   - Identical to linear probe feature extraction

**Benefits:**
- ✅ **Zero preprocessing drift**: Uses same MONAI pipeline as linear probe training
- ✅ **Maintainable**: Changes to preprocessing automatically propagate
- ✅ **Correct**: Leverages battle-tested DINOv3 data pipeline

### Usage

```bash
# Edit run_dinov3_baseline.sh to set paths, then:
sbatch run_dinov3_baseline.sh
```

### Requirements

- Trained DINOv3 logistic regression model (`.pkl` from joblib)
- Pretrained DINOv3 weights
- Multimodal dataset CSV with columns: `mri_path`, `research_group` (no `pat_id` needed)
- CV splits JSON (from fusion experiment)

### Paths Configuration

```bash
# In run_dinov3_baseline.sh:
DATA_CSV="/home/ssim0068/data/multimodal-dataset/all_mni.csv"
MRI_ROOT="/home/ssim0068/data/multimodal-dataset/all_mni/images"
CHECKPOINT_PATH="/path/to/trained/final_logreg_model.pkl"
CV_SPLITS_JSON="/path/to/cv_splits.json"
```

### Output

Results saved to `{SAVE_DIR}/dinov3_linear_probe_baseline_results.json` with:
- Per-fold metrics: accuracy, balanced accuracy, AUC, precision, recall, F1, MCC
- Aggregated metrics: mean ± std across folds
- Confusion matrices: per-fold and aggregated

### Implementation Details

**CSV Format Conversion:**

Multimodal CSV format:
```csv
RID,Subject,research_group,mri_path,...
606,126_S_0606,AD,/path/to/126_S_0606.nii.gz,...
```

Temporary ADNI-format CSV (per fold):
```csv
pat_id,label
126_S_0606,1
007_S_1206,0
```

**Dataset String Construction:**

```python
dataset_str = f"ADNI:split=TEST:root={mri_root}:extra={temp_csv_dir}:csv_filename=fold{i}_test.csv"
dataset = make_dataset(dataset_str, transform=make_classification_eval_transform(224, 224))
```

This automatically:
1. Creates base `ADNI` dataset (parses CSV, provides `pat_id` and `label`)
2. Wraps with `SliceAggregationDataset` (loads volumes via MONAI, extracts 2D slices)
3. Applies 2D ImageNet transforms to each slice

---

## BrainIAC MRI Baseline

### Usage

```bash
uv run python -m src.integration.unimodal_baseline.mri_linear_probe_baseline \
  --data-csv /home/ssim0068/data/multimodal-dataset/all.csv \
  --cv-splits-json /home/ssim0068/multimodal-AD/runs/fusion_weighted_attention_nn_5fold_cv/cv_splits.json \
  --checkpoint-path /home/ssim0068/multimodal-AD/src/mri/BrainIAC/src/logs/mni305/cn_ad_linear_probe_mni305_batch32-epoch=84-val_auc=0.77.ckpt \
  --simclr-ckpt-path /home/ssim0068/multimodal-AD/src/mri/BrainIAC/src/checkpoints/BrainIAC.ckpt \
  --save-dir /home/ssim0068/multimodal-AD/runs/mri_baseline_eval \
  --device cpu \
  --image-size 96 96 96
```

---

## Protein Baseline Evaluation

### Features

- **No retraining**: Uses pre-trained protein models (inference only)
- **Pre-fitted scaler**: Reuses the scaler from training
- **Same CV splits**: Uses identical splits as fusion for fair comparison
- **Test-only evaluation**: Only evaluates on test sets (no validation)
- **Modular design**: Supports both MLP and Transformer models

### Usage

**MLP Baseline:**
```bash
uv run python src/integration/unimodal_baseline/protein_baseline.py \
  --model-type mlp \
  --model-path src/protein/runs/run_20251009_113510/models/neural_network.pkl \
  --scaler-path src/protein/runs/run_20251009_113510/scaler.pkl \
  --data-csv /home/ssim0068/data/multimodal-dataset/all.csv \
  --cv-splits-json runs/fusion_mlp_5fold_cv/cv_splits.json \
  --save-dir runs/baseline_protein_mlp_5fold_cv
```

**Transformer Baseline:**
```bash
uv run python src/integration/unimodal_baseline/protein_baseline.py \
  --model-type transformer \
  --model-path src/protein/runs/run_20251003_133215/models/protein_transformer.pth \
  --scaler-path src/protein/runs/run_20251003_133215/scaler.pkl \
  --data-csv /home/ssim0068/data/multimodal-dataset/all.csv \
  --cv-splits-json runs/fusion_transformer_5fold_cv/cv_splits.json \
  --save-dir runs/baseline_protein_transformer_5fold_cv
```

### Arguments

- `--model-type`: Type of protein model (`mlp` or `transformer`)
- `--model-path`: Path to pre-trained model (`.pkl` for MLP, `.pth` for Transformer)
- `--scaler-path`: Path to pre-fitted scaler (`.pkl` file)
- `--data-csv`: Path to CSV with protein data
- `--cv-splits-json`: Path to `cv_splits.json` from fusion experiment
- `--save-dir`: Directory to save baseline results

### Output

Results saved to `{save_dir}/protein_baseline_results.json` with:
- Per-fold test metrics (Accuracy, Balanced Accuracy, AUC, Precision, Recall, F1)
- Aggregated metrics (mean ± std across folds)
- Aggregated confusion matrix

---

## Comparison with Fusion

Compare results directly:
- **Fusion**: `runs/fusion_{model_type}_5fold_cv/aggregated_results.json`
- **Protein baseline**: `runs/baseline_protein_{model_type}_5fold_cv/protein_baseline_results.json`



## Unimodal MRI Baseline Evaluation
MRI baseline utilizes the pretrained SimCLR BrainIAC model followed by the simple MLP model used for linear probing. 

```bash
uv run python -m src.integration.unimodal_baseline.mri_linear_probe_baseline \
  --data-csv /home/ssim0068/data/multimodal-dataset/all.csv \
  --cv-splits-json /home/ssim0068/multimodal-AD/runs/fusion_weighted_attention_nn_5fold_cv/cv_splits.json \
  --checkpoint-path /home/ssim0068/multimodal-AD/src/mri/BrainIAC/src/logs/mni305/cn_ad_linear_probe_mni305_batch32-epoch=84-val_auc=0.77.ckpt \
  --simclr-ckpt-path /home/ssim0068/multimodal-AD/src/mri/BrainIAC/src/checkpoints/BrainIAC.ckpt \
  --save-dir /home/ssim0068/multimodal-AD/runs/mri_baseline_eval \
  --device cpu \
  --image-size 96 96 96
```
