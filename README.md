# Protein Autoencoder Pipeline

A config-driven pipeline for training autoencoders on proteomic data and visualizing extracted features.

## Overview

This pipeline provides:
- **Autoencoder training** for feature extraction from proteomic data
- **Feature visualization** using PCA, t-SNE, UMAP, and clustering
- **Config-driven approach** using YAML files for easy parameter management
- **Multiple dataset support** for different proteomic datasets
- **Comprehensive logging** with unique experiment directories
- **Experiment tracking** with timestamps and detailed summaries

## File Structure

```
├── main.py                 # Main pipeline controller
├── config.py              # Configuration classes
├── protein_model.py       # Autoencoder model and training
├── protein_feature_vis.py # Feature visualization
├── configs/
│   └── default.yml        # Default configuration
└── README.md              # This file
```

## Installation

1. **Install dependencies:**
```bash
poetry add pyyaml
```

2. **Verify installation:**
```bash
python main.py --help
```

## Usage

### 1. Command Line Interface

**Train autoencoder (auto-generated experiment name):**
```bash
python main.py --dataset mrm_small --train
```

**Train with custom experiment name:**
```bash
python main.py --dataset mrm_small --train --experiment-name "my_experiment_001"
```

**Visualize features (uses most recent experiment):**
```bash
python main.py --dataset mrm_small --visualize
```

**Train and visualize:**
```bash
python main.py --dataset mrm_small --train --visualize
```

**Use custom configuration:**
```bash
python main.py --config configs/my_config.yml --dataset mrm_small --train
```

**Override output directory:**
```bash
python main.py --dataset mrm_small --train --output-dir "D:/my_output"
```

### 2. Programmatic Usage

```python
from main import train_dataset, visualize_dataset, process_all_datasets

# Train a specific dataset
results = train_dataset('mrm_small', 'configs/default.yml')

# Train with custom experiment name
results = train_dataset('mrm_small', 'configs/default.yml', 'my_experiment')

# Visualize features
viz_results = visualize_dataset('mrm_small', 'configs/default.yml')

# Process all datasets
all_results = process_all_datasets('configs/default.yml')
```

### 3. Configuration Files

The pipeline uses YAML configuration files. Example `configs/default.yml`:

```yaml
# Model parameters
hidden_size: 8
dropout_rate: 0.1

# Training parameters
num_epochs: 200
learning_rate: 0.001
patience: 20
batch_size: 16

# Dataset configurations
datasets:
  mrm_small:
    name: mrm_small
    metadata_path: "D:\\ADNI\\AD_CN\\proteomics\\Biomarkers Consortium Plasma Proteomics MRM\\metadata.csv"
    exclude_columns:
      - RID
      - VISCODE
      - MRI_acquired
      - research_group
      - subject_age
```

## Configuration Parameters

### Model Parameters
- `hidden_size`: Size of the latent space (default: 8)
- `dropout_rate`: Dropout rate for regularization (default: 0.1)

### Training Parameters
- `num_epochs`: Maximum training epochs (default: 200)
- `learning_rate`: Learning rate (default: 0.001)
- `patience`: Early stopping patience (default: 20)
- `batch_size`: Batch size (default: 16)
- `weight_decay`: L2 regularization (default: 1e-5)

### Data Parameters
- `test_size`: Test set fraction (default: 0.2)
- `random_state`: Random seed (default: 42)

### Dataset Configuration
- `name`: Dataset identifier
- `metadata_path`: Path to CSV file with proteomic data
- `exclude_columns`: Columns to exclude from features

## Experiment Management

### Unique Experiment Directories

Each training run creates a unique experiment directory with timestamp:

```
dataset_name/
└── dataset_name_YYYYMMDD_HHMMSS/
    ├── train_features_autoencoder.npy    # Extracted training features
    ├── test_features_autoencoder.npy     # Extracted test features
    ├── best_autoencoder.pth              # Best trained model
    ├── scaler.pkl                        # Data scaler
    ├── training_curves.png               # Training loss curves
    ├── feature_quality_analysis.png      # Feature quality plots
    ├── dimensionality_reduction_analysis.png  # PCA, t-SNE, UMAP plots
    ├── visualization_data.npy            # Visualization data
    ├── training_log.txt                  # Detailed training log
    ├── config_snapshot.yml              # Configuration snapshot
    └── experiment_summary.txt            # Experiment summary
```

### Logging Features

- **Training Log**: Detailed log of training progress, epochs, losses
- **Config Snapshot**: Exact configuration used for the experiment
- **Experiment Summary**: Comprehensive summary of parameters and results
- **Console Output**: Real-time progress updates during training

### Example Experiment Directory

```
mrm_small/
└── mrm_small_20241201_143022/
    ├── training_log.txt
    ├── config_snapshot.yml
    ├── experiment_summary.txt
    ├── best_autoencoder.pth
    ├── training_curves.png
    ├── train_features_autoencoder.npy
    ├── test_features_autoencoder.npy
    ├── scaler.pkl
    ├── feature_quality_analysis.png
    ├── dimensionality_reduction_analysis.png
    └── visualization_data.npy
```

## Output Files

### Training Outputs
- `best_autoencoder.pth`: Best trained model weights
- `training_curves.png`: Training and validation loss curves
- `train_features_autoencoder.npy`: Extracted training features
- `test_features_autoencoder.npy`: Extracted test features
- `scaler.pkl`: Data scaler for preprocessing

### Logging Outputs
- `training_log.txt`: Detailed training log with timestamps
- `config_snapshot.yml`: Configuration used for the experiment
- `experiment_summary.txt`: Human-readable experiment summary

### Visualization Outputs
- `feature_quality_analysis.png`: Feature distribution, variance, correlation
- `dimensionality_reduction_analysis.png`: PCA, t-SNE, UMAP visualizations
- `visualization_data.npy`: Raw visualization data for further analysis

## Adding New Datasets

1. **Add to YAML config:**
```yaml
datasets:
  my_new_dataset:
    name: my_new_dataset
    metadata_path: "path/to/your/metadata.csv"
    exclude_columns:
      - RID
      - VISCODE
      - MRI_acquired
      - research_group
      - subject_age
```

2. **Run pipeline:**
```bash
python main.py --dataset my_new_dataset --train --visualize
```

## Visualization Features

The pipeline provides comprehensive feature analysis:

1. **Feature Quality Analysis:**
   - Feature distribution
   - Feature variance
   - Feature correlation matrix
   - Feature importance (LDA)

2. **Dimensionality Reduction:**
   - PCA visualization
   - t-SNE visualization
   - UMAP visualization
   - K-means clustering
   - Sample distance matrix

3. **Performance Metrics:**
   - KNN classification accuracy
   - Silhouette score for clustering
   - Feature statistics

## Example Workflow

```bash
# 1. Create custom configuration
cp configs/default.yml configs/my_experiment.yml
# Edit my_experiment.yml with your parameters

# 2. Train autoencoder with custom experiment name
python main.py --config configs/my_experiment.yml --dataset mrm_small --train --experiment-name "baseline_model"

# 3. Check the experiment directory
ls "D:\ADNI\AD_CN\proteomics\Biomarkers Consortium Plasma Proteomics MRM\mrm_small\mrm_small_20241201_143022\"

# 4. View training log
cat "D:\ADNI\AD_CN\proteomics\Biomarkers Consortium Plasma Proteomics MRM\mrm_small\mrm_small_20241201_143022\training_log.txt"

# 5. View experiment summary
cat "D:\ADNI\AD_CN\proteomics\Biomarkers Consortium Plasma Proteomics MRM\mrm_small\mrm_small_20241201_143022\experiment_summary.txt"

# 6. Visualize results
python main.py --config configs/my_experiment.yml --dataset mrm_small --visualize
```

## Experiment Tracking

### Automatic Experiment Naming
- Format: `{dataset_name}_{YYYYMMDD}_{HHMMSS}`
- Example: `mrm_small_20241201_143022`

### Custom Experiment Names
- Use `--experiment-name` flag
- Example: `--experiment-name "baseline_model_v1"`

### Finding Previous Experiments
- Experiments are stored in `{base_path}/{dataset_name}/{experiment_name}/`
- Visualization automatically finds the most recent experiment
- Manual specification: `--experiment-dir "path/to/experiment"`

## Troubleshooting

**Configuration file not found:**
- The pipeline will create a default configuration automatically
- Check the path in `--config` argument

**Dataset not found:**
- Ensure the dataset name exists in your YAML configuration
- Check the `metadata_path` exists and is accessible

**Memory issues:**
- Reduce `batch_size` in configuration
- Reduce `hidden_size` for smaller latent space

**Training issues:**
- Adjust `learning_rate` and `patience`
- Check data quality and missing values

**Experiment directory issues:**
- Check that the base path exists and is writable
- Ensure sufficient disk space for experiment files
- Verify that the dataset directory structure is correct