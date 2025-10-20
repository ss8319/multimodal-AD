#!/bin/bash
# Tip: set a meaningful job name at submission time, e.g.
#   sbatch --job-name=fusion_mlp_hidden_layer_2 train_fusion_slurm.sh
# The %x token below will be replaced by the job name
#SBATCH --job-name=fusion_train_NN
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=39G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4


# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs

# Change to project directory
cd /home/ssim0068/multimodal-AD

# Verify environment
echo "Python version: $(uv run python --version)"
echo "Python path: $(uv run which python)"
echo "CUDA available: $(uv run python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(uv run python -c 'import torch; print(torch.cuda.device_count())')"
echo "MONAI available: $(uv run python -c 'import monai; print(monai.__version__)')"
echo "NumPy version: $(uv run python -c 'import numpy; print(numpy.__version__)')"
echo "Scikit-learn version: $(uv run python -c 'import sklearn; print(sklearn.__version__)')"

# Check if wandb is available, install if needed
echo "Checking for wandb..."
HAS_WANDB=$(uv run python -c "import importlib.util; print('Yes' if importlib.util.find_spec('wandb') is not None else 'No')" 2>/dev/null || echo "No")
echo "wandb available: $HAS_WANDB"

# Set up wandb if available
echo "Setting up wandb logging..."
# You can set your API key here or use wandb login in advance
# export WANDB_API_KEY="your-api-key-here"
export WANDB_PROJECT="multimodal-ad"
# export WANDB_ENTITY="your-username"  # Remove placeholder; set this in environment if needed
export WANDB_GROUP="cv_$(date +%Y%m%d_%H%M%S)"
echo "wandb configuration: project=$WANDB_PROJECT, entity=${WANDB_ENTITY:-<unset>}, group=$WANDB_GROUP"

# Run training using uv environment
echo "Starting multimodal fusion training with merged uv environment..."
uv run python src/integration/train_fusion.py

# Print completion info
echo "Job completed at: $(date)"
echo "Exit code: $?"
