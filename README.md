# Protein Autoencoder and TabNet Pipeline

A config-driven pipeline for training autoencoders and TabNet models on proteomic data and visualizing extracted features.

## Overview

This pipeline provides:
- **Autoencoder training** for unsupervised feature extraction from proteomic data
- **TabNet pretraining** for unsupervised feature extraction using TabNet architecture
- **Feature visualization** using PCA, t-SNE, UMAP, and clustering
- **Feature interpretation** for biological insights
- **Config-driven approach** using YAML files for easy parameter management
- **Multiple dataset support** for different proteomic datasets
- **Comprehensive logging** with unique experiment directories
- **Experiment tracking** with timestamps and detailed summaries

## Script Architecture

The pipeline consists of 7 interconnected Python scripts organized into two main workflows:

### **Main Pipeline (Training & Visualization)**
```
main.py (Central Controller)
├── config.py (Configuration Management)
├── base_model.py (Shared Functionality)
├── protein_model.py (Autoencoder Training Engine)
├── tabnet_model.py (TabNet Training Engine)
└── protein_feature_vis.py (Visualization Engine)
```

### **Interpretability Pipeline (Separate Workflow)**
```
interpret_features.py (Standalone Runner)
└── protein_feature_interpretation.py (Interpretability Engine)
```

### **Script Roles & Connections**

#### **Core Pipeline Scripts**

1. **`main.py`** - Central Pipeline Controller
   - **Role:** Orchestrates the entire training and visualization workflow
   - **Connections:** Imports and calls functions from `config.py`, `base_model.py`, `protein_model.py`, `tabnet_model.py`, and `protein_feature_vis.py`
   - **Functions:** 
     - Loads YAML configuration files
     - Calls `train_autoencoder_pipeline()` or `train_tabnet_pipeline()` for training
     - Calls `visualize_features_pipeline()` for visualization
     - Manages experiment directories and logging

2. **`config.py`** - Configuration Management
   - **Role:** Defines data structures and configuration classes
   - **Classes:** `AutoencoderConfig`, `TabNetConfig`, `DatasetConfig`, `DATASET_CONFIGS`
   - **Connections:** Used by all other scripts for configuration management
   - **Functions:** `get_paths()` generates unique experiment directory structures

3. **`base_model.py`** - Shared Functionality
   - **Role:** Contains common functions used by both autoencoder and TabNet models
   - **Functions:** `setup_logging()`, `save_config_snapshot()`, `load_and_prepare_data()`, `plot_training_curves()`, `create_experiment_summary()`
   - **Purpose:** Eliminates code duplication between models

4. **`protein_model.py`** - Autoencoder Training Engine
   - **Role:** Defines autoencoder architecture and handles training
   - **Classes:** `ProteinAutoencoder` (PyTorch neural network)
   - **Connections:** Called by `main.py` for autoencoder training workflow
   - **Functions:**
     - `train_autoencoder_pipeline()` - Complete autoencoder training workflow
     - `extract_features()` - Feature extraction from trained model

5. **`tabnet_model.py`** - TabNet Training Engine
   - **Role:** Defines TabNet pretraining and handles unsupervised feature extraction
   - **Classes:** Uses `TabNetPretrainer` from pytorch-tabnet
   - **Connections:** Called by `main.py` for TabNet training workflow
   - **Functions:**
     - `train_tabnet_pipeline()` - Complete TabNet pretraining workflow
     - `extract_tabnet_features()` - Feature extraction from trained TabNet

6. **`protein_feature_vis.py`** - Visualization Engine
   - **Role:** Creates comprehensive feature analysis visualizations
   - **Connections:** Called by `main.py` for visualization workflow
   - **Functions:**
     - `plot_feature_quality_analysis()` - Feature distribution, variance, correlation
     - `plot_dimensionality_reduction()` - PCA, t-SNE, UMAP, clustering
     - `analyze_features()` - Performance metrics and statistics

#### **Interpretability Scripts (Separate Workflow)**

7. **`interpret_features.py`** - Standalone Interpretability Runner
   - **Role:** Independent script for feature interpretation (separate from main pipeline)
   - **Connections:** Imports functions from `protein_feature_interpretation.py`
   - **Functions:** Automatically finds most recent experiment and runs interpretation

8. **`protein_feature_interpretation.py`** - Interpretability Engine
   - **Role:** Provides biological interpretation of learned features
   - **Connections:** Called by `interpret_features.py` (separate workflow)
   - **Functions:**
     - `analyze_feature_contributions()` - Maps features back to original proteins
     - `analyze_feature_correlations_with_diagnosis()` - Correlates features with AD diagnosis
     - `create_feature_summary_report()` - Generates detailed interpretation reports

### **Workflow Separation**

- **Main Pipeline:** `main.py` → `protein_model.py`/`tabnet_model.py` → `protein_feature_vis.py`
- **Interpretability Pipeline:** `interpret_features.py` → `protein_feature_interpretation.py`
- **No direct connection** between main pipeline and interpretability scripts
- **Shared resources:** Both workflows use the same experiment directories and saved models

## File Structure

```
├── main.py                 # Main pipeline controller
├── config.py              # Configuration classes
├── base_model.py          # Shared functionality
├── protein_model.py       # Autoencoder model and training
├── tabnet_model.py        # TabNet model and training
├── protein_feature_vis.py # Feature visualization
├── interpret_features.py   # Standalone interpretability runner
├── protein_feature_interpretation.py # Feature interpretation engine
├── configs/
│   ├── default.yml        # Default configuration
│   ├── autoencoder_experiment.yml # Autoencoder-specific config
│   └── tabnet_experiment.yml # TabNet-specific config
└── README.md              # This file
```

## Installation

1. **Install dependencies:**
```bash
poetry add pyyaml pytorch-tabnet
```

2. **Verify installation:**
```bash
python main.py --help
```

## Complete Workflow

### **Two-Stage Pipeline**

The pipeline operates in two distinct stages:

#### **Stage 1: Training & Visualization (Main Pipeline)**
```bash
# Train autoencoder and visualize features
python main.py --dataset proteomic_full --model-type autoencoder --train --visualize

# Train TabNet and visualize features
python main.py --dataset proteomic_full --model-type tabnet --train --visualize
```

**What happens:**
1. `main.py` loads configuration from YAML
2. `protein_model.py` or `tabnet_model.py` trains model and extracts 8-dimensional features
3. `protein_feature_vis.py` creates comprehensive visualizations
4. Results saved to unique experiment directory

#### **Stage 2: Feature Interpretation (Separate Workflow)**
```bash
# Interpret the learned features (run after training)
python interpret_features.py
```

**What happens:**
1. `interpret_features.py` automatically finds your most recent experiment
2. `protein_feature_interpretation.py` analyzes feature-to-protein mappings
3. Creates biological interpretation reports and correlation analyses

### **Key Workflow Separation**

- **Main Pipeline:** Training → Visualization (connected workflow)
- **Interpretability Pipeline:** Feature interpretation (separate, independent workflow)
- **Shared Resources:** Both use the same experiment directories and saved models
- **No Direct Connection:** Interpretability is a separate analysis step

## Usage

### 1. Command Line Interface

**Train autoencoder (auto-generated experiment name):**
```bash
python main.py --dataset proteomic_full --model-type autoencoder --train
```

**Train TabNet (auto-generated experiment name):**
```bash
python main.py --dataset proteomic_full --model-type tabnet --train
```

**Train with custom experiment name:**
```bash
python main.py --dataset proteomic_full --model-type autoencoder --train --experiment-name "my_experiment_001"
```

**Visualize features (uses most recent experiment):**
```bash
python main.py --dataset proteomic_full --model-type autoencoder --visualize
```

**Train and visualize:**
```bash
python main.py --dataset proteomic_full --model-type autoencoder --train --visualize
```

**Use custom configuration:**
```bash
python main.py --config configs/autoencoder_experiment.yml --dataset proteomic_full --model-type autoencoder --train
```

**Override output directory:**
```bash
python main.py --dataset proteomic_full --model-type autoencoder --train --output-dir "D:/my_output"
```

### 2. Programmatic Usage

```python
from main import train_dataset, visualize_dataset, process_all_datasets

# Train a specific dataset with autoencoder
results = train_dataset('proteomic_full', 'autoencoder', 'configs/default.yml')

# Train a specific dataset with TabNet
results = train_dataset('proteomic_full', 'tabnet', 'configs/default.yml')

# Train with custom experiment name
results = train_dataset('proteomic_full', 'autoencoder', 'configs/default.yml', 'my_experiment')

# Visualize features
viz_results = visualize_dataset('proteomic_full', 'configs/default.yml')

# Process all datasets
all_results = process_all_datasets('configs/default.yml', 'autoencoder')
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

# TabNet parameters
tabnet_params:
  num_features: 8
  feature_dim: 64
  output_dim: 2
  num_decision_steps: 3
  relaxation_factor: 1.5
  sparsity_coefficient: 1.0e-5
  batch_momentum: 0.98
  virtual_batch_size: 128
  mask_type: "sparsemax"

# Dataset configurations
datasets:
  proteomic_full:
    name: proteomic_full
    metadata_path: "D:\\ADNI\\AD_CN\\proteomics\\Biomarkers Consortium Plasma Proteomics MRM\\proteomic_w_labels.csv"
    exclude_columns:
      - RID
      - VISCODE
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

### TabNet Parameters
- `num_features`: Number of features to select (default: 8)
- `feature_dim`: Dimension of features (default: 64)
- `output_dim`: Number of classes (default: 2)
- `num_decision_steps`: Number of decision steps (default: 3)
- `relaxation_factor`: Relaxation factor (default: 1.5)
- `sparsity_coefficient`: Sparsity coefficient (default: 1e-5)
- `batch_momentum`: Batch momentum (default: 0.98)
- `virtual_batch_size`: Virtual batch size (default: 128)
- `mask_type`: Mask type ("sparsemax" or "entmax") (default: "sparsemax")

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
proteomic_full/
└── proteomic_full_YYYYMMDD_HHMMSS/
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

## Model Comparison

### Autoencoder vs TabNet

| Aspect | Autoencoder | TabNet |
|--------|-------------|--------|
| **Architecture** | Neural network with bottleneck | Attention-based tabular learning |
| **Learning** | Unsupervised reconstruction | Unsupervised pretraining |
| **Feature Extraction** | Bottleneck layer (8 dims) | Encoder + dimensionality handling (8 dims) |
| **Interpretability** | Feature-to-protein mapping | Built-in attention mechanisms |
| **Use Case** | Standard autoencoder approach | Tabular data with attention |

Both models:
- ✅ Extract **8-dimensional features**
- ✅ Use **unsupervised learning**
- ✅ Work with the same **visualization pipeline**
- ✅ Support **config-driven experimentation**

## Output Files

### Training Outputs
- `best_autoencoder.pth` / `best_tabnet.pth`: Best trained model weights
- `training_curves.png`: Training and validation loss curves
- `train_features_autoencoder.npy` / `train_features_tabnet.npy`: Extracted training features
- `test_features_autoencoder.npy` / `test_features_tabnet.npy`: Extracted test features
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
      - research_group
      - subject_age
```

2. **Run pipeline:**
```bash
python main.py --dataset my_new_dataset --model-type autoencoder --train --visualize
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

## Interpretability Features

The interpretability pipeline provides biological insights:

1. **Feature-to-Protein Mapping:**
   - Maps each of the 8 learned features back to original proteins
   - Identifies top positive and negative protein contributors
   - Calculates effective weights through neural network layers

2. **Diagnosis Correlation Analysis:**
   - Correlates each feature with AD/CN diagnosis
   - Ranks features by importance for disease classification
   - Creates correlation plots and importance rankings

3. **Biological Interpretation:**
   - Generates detailed text reports for each feature
   - Identifies protein patterns and biological pathways
   - Provides clinical relevance insights

4. **Output Files:**
   - `feature_interpretation.png`: Visual breakdown of each feature
   - `feature_interpretation_report.txt`: Detailed text report
   - `feature_diagnosis_correlations.png`: Feature vs AD diagnosis correlations

## Example Workflow

```bash
# 1. Create custom configuration
cp configs/default.yml configs/my_experiment.yml
# Edit my_experiment.yml with your parameters

# 2. Train autoencoder with custom experiment name
python main.py --config configs/my_experiment.yml --dataset proteomic_full --model-type autoencoder --train --experiment-name "baseline_autoencoder"

# 3. Train TabNet with custom experiment name
python main.py --config configs/my_experiment.yml --dataset proteomic_full --model-type tabnet --train --experiment-name "baseline_tabnet"

# 4. Check the experiment directories
ls "D:\ADNI\AD_CN\proteomics\Biomarkers Consortium Plasma Proteomics MRM\proteomic_full\"

# 5. View training log
cat "D:\ADNI\AD_CN\proteomics\Biomarkers Consortium Plasma Proteomics MRM\proteomic_full\proteomic_full_YYYYMMDD_HHMMSS\training_log.txt"

# 6. View experiment summary
cat "D:\ADNI\AD_CN\proteomics\Biomarkers Consortium Plasma Proteomics MRM\proteomic_full\proteomic_full_YYYYMMDD_HHMMSS\experiment_summary.txt"

# 7. Visualize results
python main.py --config configs/my_experiment.yml --dataset proteomic_full --model-type autoencoder --visualize

# 8. Interpret features (separate workflow)
python interpret_features.py

# 9. View interpretability results
cat "D:\ADNI\AD_CN\proteomics\Biomarkers Consortium Plasma Proteomics MRM\proteomic_full\proteomic_full_YYYYMMDD_HHMMSS\feature_interpretation_report.txt"
```

## Experiment Tracking

### Automatic Experiment Naming
- Format: `{dataset_name}_{YYYYMMDD}_{HHMMSS}`
- Example: `proteomic_full_20241201_143022`

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

**TabNet-specific issues:**
- Ensure `pytorch-tabnet` is properly installed
- Check that the model parameters are compatible with your data
- Verify that the pretraining ratio is appropriate for your dataset