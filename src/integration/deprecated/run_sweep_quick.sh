#!/bin/bash
# Quick hyperparameter sweep script using SLURM for GPU acceleration

#SBATCH --job-name=fusion_sweep_quick
#SBATCH --output=sweeps/%x_%j.out
#SBATCH --error=sweeps/%x_%j.err
#SBATCH --time=2:00:00                     # Shorter time for quick test
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
mkdir -p sweeps

# Change to project directory
cd /home/ssim0068/multimodal-AD

# Verify environment
echo "Python version: $(uv run python --version)"
echo "Python path: $(uv run which python)"
echo "CUDA available: $(uv run python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(uv run python -c 'import torch; print(torch.cuda.device_count())')"

# Check if wandb is available
echo "Checking for wandb..."
HAS_WANDB=$(uv run python -c "import importlib.util; print('Yes' if importlib.util.find_spec('wandb') is not None else 'No')" 2>/dev/null || echo "No")
echo "wandb available: $HAS_WANDB"

# Set up wandb if available
echo "Setting up wandb logging..."
export WANDB_PROJECT="multimodal-ad"
export WANDB_GROUP="sweep_quick_$(date +%Y%m%d_%H%M%S)"
echo "wandb configuration: project=$WANDB_PROJECT, entity=${WANDB_ENTITY:-<unset>}, group=$WANDB_GROUP"

echo "Starting QUICK hyperparameter sweep..."

# Run quick sweep using uv environment
uv run python src/integration/hyperparameter_sweep.py quick

echo "Quick sweep completed!"
echo "Job completed at: $(date)"
echo "Exit code: $?"
