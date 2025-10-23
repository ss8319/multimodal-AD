#!/bin/bash
# W&B Hyperparameter Sweep using SLURM for GPU acceleration

#SBATCH --job-name=wandb_sweep
#SBATCH --output=sweeps/%x_%j.out
#SBATCH --error=sweeps/%x_%j.err
#SBATCH --time=6:00:00                     # 6 hours for 30 trials
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

# Set up wandb
echo "Setting up wandb..."
export WANDB_PROJECT="AD-multimodal"
export WANDB_ENTITY="shamussim"  # Replace with your W&B username
echo "wandb configuration: project=$WANDB_PROJECT, entity=$WANDB_ENTITY"

echo "Starting W&B hyperparameter sweep..."

# Run sweep using uv environment
if [ "$1" = "quick" ]; then
    echo "Running quick W&B sweep..."
    uv run python src/integration/wandb_sweep.py quick
else
    echo "Running full W&B sweep..."
    uv run python src/integration/wandb_sweep.py
fi

echo "W&B sweep completed!"
echo "Job completed at: $(date)"
echo "Exit code: $?"
