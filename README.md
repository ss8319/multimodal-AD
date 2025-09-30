# Imaging Proteomics and Genomics Integration
*In-progress -- this repo is under active development*
Our goal is to train multimodal classifcation method that leverages protein and MRI data to classify Alzheimer's disease and to leverage existing XAI work to understand the interactions between imaging biomarkers and protein biomarkers. 

## Installation
Clone the repo, install [uv](https://docs.astral.sh/uv/getting-started/installation/), and run

```bash
uv sync
```

This will create a new virtual environment for the project with all the required dependencies. Activate the environment with

```bash
source .venv/bin/activate
```

or use `uv run`.

# Add new dependencies
```bash
uv add package-name
```

## Code Organisation

- **`src/data/`**: Dataset pipelines
- **`src/mri/BrainIAC/`**: BrainIAC foundation model 
- **`src/protein/`**: Proteomics training 
- **`src/integration/`**: Experiments for fusing latent representations between modalities

## 📊 Data Sources

- **ADNI**: Alzheimer's Disease Neuroimaging Initiative
