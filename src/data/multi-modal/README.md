# Multimodal Dataset Preparation

This directory contains scripts for preparing multimodal datasets that combine protein and MRI data.

## prepare_multimodal_dataset.py

Prepares a multimodal dataset by flattening nested MRI directory structures and creating balanced train/test splits.

### What it does:

1. **Finds MRI files**: Searches for `.nii.gz` files in nested directory structures
2. **Creates balanced splits**: Generates train/test splits maintaining:
   - Research group balance (AD/CN)
   - Sex distribution
   - Age distribution
3. **Organizes files**: Copies MRI files to flat directory structure
4. **Generates CSVs**: Creates train.csv and test.csv with all metadata

### Usage:

```bash
python prepare_multimodal_dataset.py
```

### Input:
- Merged CSV file with protein and MRI metadata
- Nested MRI directory structure

### Output:
- `train.csv` and `test.csv` with balanced splits
- `train/images/` and `test/images/` directories with MRI files
- Flat file structure for easy loading

### Configuration:
- Test size: 8 samples (configurable)
- Random seed: 42
- Output directory: `/home/ssim0068/data/multimodal-dataset`

### Features:
- **Balanced sampling**: Maintains demographic balance across splits
- **File validation**: Checks for missing MRI files
- **Progress tracking**: Shows detailed progress and statistics
- **Error handling**: Graceful handling of stratification failures
