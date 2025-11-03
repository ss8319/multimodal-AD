#!/bin/bash
#SBATCH --job-name=dinov3_baseline
#SBATCH --output=logs/dinov3_baseline_%j.out
#SBATCH --error=logs/dinov3_baseline_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=39G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

set -e  # Exit on error

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs

# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# =============================================================================

# Data paths
DATA_CSV="/home/ssim0068/data/multimodal-dataset/all_mni.csv"
MRI_ROOT="/home/ssim0068/data/multimodal-dataset/all_mni/images"

# CV splits from fusion experiment (must match protein/fusion baselines)
CV_SPLITS_JSON="/home/ssim0068/multimodal-AD/runs/fusion_weighted_attention_nn_5fold_cv/cv_splits.json"

# DINOv3 paths
DINOV3_HUB_DIR="/home/ssim0068/multimodal-AD/src/mri/dinov3"
DINOV3_MODEL="dinov3_vits16"  # Options: dinov3_vits16, dinov3_vitb16, dinov3_vitl16
PRETRAINED_WEIGHTS="/home/ssim0068/multimodal-AD/src/mri/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

# Trained logistic regression checkpoint (from DINOv3 linear probe training)
# Use the model trained on ADNI data (not contaminated with multimodal dataset)
CHECKPOINT_PATH="/home/ssim0068/multimodal-AD/src/mri/dinov3/runs/adni_full_test_20251030_151635/final_logreg_model.pkl"

# Output directory
SAVE_DIR="/home/ssim0068/multimodal-AD/runs/dinov3_smri_baseline_$(date +%Y%m%d_%H%M%S)"

# =============================================================================
# DINOv3-SPECIFIC PARAMETERS
# =============================================================================

IMAGE_SIZE=224        # DINOv3 image size
SLICE_AXIS=0          # 0=sagittal, 1=coronal, 2=axial
STRIDE=2              # Slice stride for aggregation
DEVICE="cuda"         # cuda or cpu

# =================================================== ==========================
# RUN EVALUATION
# =============================================================================

echo "==================================="
echo "DINOv3 sMRI-ONLY BASELINE EVALUATION"
echo "==================================="
echo "Data CSV: ${DATA_CSV}"
echo "MRI Root: ${MRI_ROOT}"
echo "CV Splits: ${CV_SPLITS_JSON}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "DINOv3 Model: ${DINOV3_MODEL}"
echo "Save Dir: ${SAVE_DIR}"
echo "Device: ${DEVICE}"
echo "==================================="
echo ""

# Navigate to multimodal root for uv environment
cd /home/ssim0068/multimodal-AD

# Run baseline evaluation
uv run python src/integration/unimodal_baseline/dinov3_linear_probe_baseline.py \
  --data-csv "${DATA_CSV}" \
  --cv-splits-json "${CV_SPLITS_JSON}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --dinov3-hub-dir "${DINOV3_HUB_DIR}" \
  --dinov3-model "${DINOV3_MODEL}" \
  --pretrained-weights "${PRETRAINED_WEIGHTS}" \
  --mri-root "${MRI_ROOT}" \
  --save-dir "${SAVE_DIR}" \
  --device "${DEVICE}" \
  --image-size "${IMAGE_SIZE}" \
  --slice-axis "${SLICE_AXIS}" \
  --stride "${STRIDE}"

echo ""
echo "==================================="
echo "Baseline evaluation completed!"
echo "Results saved to: ${SAVE_DIR}"
echo "End time: $(date)"
echo "==================================="

