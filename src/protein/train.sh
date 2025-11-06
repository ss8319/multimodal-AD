#!/bin/bash
#SBATCH --job-name=protein_train
#SBATCH --partition=gpu            # GPU partition (required on your cluster)
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=39G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Change to project directory
cd /home/ssim0068/multimodal-AD

# Verify GPU availability
echo "Checking CUDA availability..."
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}' if torch.cuda.is_available() else 'No GPU')"

# Run training - GPU will be auto-detected
echo "Starting protein classification training..."
uv run python src/protein/main.py --save-models

echo "Job completed at: $(date)"

