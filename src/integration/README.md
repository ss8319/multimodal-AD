# Multimodal Integration Module

Fusion of protein and MRI data for AD/CN classification.

## Components

### 1. `multimodal_dataset.py`
PyTorch Dataset that loads protein + MRI features **on-the-fly** during training.

**Features:**
- Extracts protein features from CSV
- Extracts MRI latents using BrainIAC model (on-the-fly, no pre-processing needed)
- Concatenates both modalities
- Returns fused features + label

**Usage:**
```python
from multimodal_dataset import MultimodalDataset
from load_brainiac import load_brainiac

# Load BrainIAC once
mri_model = load_brainiac('path/to/checkpoint.ckpt', 'cpu')

# Create dataset
dataset = MultimodalDataset(
    csv_path='data/multimodal-dataset/train.csv',
    brainiac_model=mri_model,
    device='cpu'
)

# Get a sample
sample = dataset[0]
# sample contains: protein_features, mri_features, fused_features, label, subject_id
```

### 2. `fusion_model.py`
Simple neural network models for classification.

**Model:**
- `SimpleFusionClassifier`: Fusion model (protein + MRI → FCNN)

**Architecture:**
```
Input (protein_dim + 768) 
  → Linear(hidden_dim) → BatchNorm → ReLU → Dropout
  → Linear(hidden_dim/2) → BatchNorm → ReLU → Dropout
  → Linear(2) → Logits
```

**Usage:**
```python
from fusion_model import get_model

# Fusion model
model = get_model(protein_dim=320, mri_dim=768)
```

### 3. `train_fusion.py`
Complete training script.

**Usage:**
```bash
python src/integration/train_fusion.py
```

**Configuration** (edit in `main()`):
- `train_csv`: Path to train.csv
- `test_csv`: Path to test.csv
- `brainiac_checkpoint`: Path to BrainIAC checkpoint
# Model is always fusion
- `batch_size`: 4 (default)
- `num_epochs`: 50 (default)
- `learning_rate`: 0.001
- `device`: 'cpu' or 'cuda'

**Output:**
- Saves best model to `runs/fusion/best_model.pth`
- Prints train/test metrics each epoch
- Tracks AUC, accuracy, sensitivity, specificity

## Quick Start

1. **Prepare data** (already done):
```bash
# Data should be at:
# /home/ssim0068/data/multimodal-dataset/
#   ├── train.csv (30 samples)
#   ├── test.csv (8 samples)
#   ├── train/images/*.nii.gz
#   └── test/images/*.nii.gz
```

2. **Train fusion model**:
```bash
conda activate brainiac
cd /home/ssim0068/multimodal-AD
python src/integration/train_fusion.py
```

3. **Monitor training**:
- Watch epoch progress
- Best model saved based on test AUC
- Results printed each epoch

## Dataset Details

- **Train**: 30 samples (18 CN, 12 AD)
- **Test**: 8 samples (5 CN, 3 AD)
- **Protein features**: 320 columns (auto-detected from CSV)
- **MRI features**: 768 (from BrainIAC)
- **Total fused dim**: 1088

## Key Design Choices

✅ **On-the-fly inference**: No need to pre-extract all MRI latents
✅ **Simple concatenation**: Direct fusion without learned projections
✅ **Small batch size**: 4 samples (due to on-the-fly MRI processing)
✅ **Subject-level split**: No data leakage between train/test
✅ **Balanced split**: Stratified by research_group and sex

## Next Steps

- Run training and check results
- Experiment with different hyperparameters
- Add validation set if needed

