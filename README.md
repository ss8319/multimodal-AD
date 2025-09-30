# Imaging Proteomics and Genomics Integration
*In-progress -- this repo is under active development*

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
