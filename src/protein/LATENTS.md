# Latent Extraction and Analysis Workflow

## Overview
This workflow extracts latent representations from trained protein classification models and performs PCA analysis for visualization and interpretation.

## Step-by-Step Process

### 1. Train Models and Save Checkpoints
```bash
uv run python src/protein/main.py --save-models
```
- Trains all classifiers (Logistic Regression, Neural Network, Protein Transformer, etc.)
- Saves trained models to `src/protein/runs/run_YYYYMMDD_HHMMSS/models/`

### 2. Extract Latent Representations
```bash
uv run python src/protein/extract_latents.py
```
- Extracts hidden layer activations from Neural Network model
- Extracts transformer embeddings from Protein Transformer model
- Saves latents to `src/protein/runs/run_YYYYMMDD_HHMMSS/latents/`

### 3. PCA Analysis and Visualization
```bash
uv run python src/protein/nn_latents_pca.py
```
- Reduces Neural Network latents to 4 principal components
- Creates pairwise scatterplots colored by AD/CN class
- Saves visualizations to `latents/neural_network/visualisation/pca_analysis_YYYYMMDD_HHMMSS/`

## Output Structure
```
src/protein/runs/run_YYYYMMDD_HHMMSS/
├── models/                          # Trained model checkpoints
├── latents/
│   ├── neural_network/              # NN hidden layer activations
│   │   ├── hidden_layer_1.npy
│   │   ├── hidden_layer_2.npy
│   │   ├── pre_output.npy
│   │   └── visualisation/           # PCA analysis outputs
│   └── protein_transformer/         # Transformer embeddings
└── cv_results_summary.csv           # Cross-validation results
```
