# Baseline Evaluation

## Unimodal Protein Baseline Evaluation

Evaluate protein-only baselines using pre-trained models on the same CV splits as the fusion model for fair comparison.

## Features

- **No retraining**: Uses pre-trained protein models (inference only)
- **Pre-fitted scaler**: Reuses the scaler from training
- **Same CV splits**: Uses identical splits as fusion for fair comparison
- **Test-only evaluation**: Only evaluates on test sets (no validation)
- **Modular design**: Supports both MLP and Transformer models

## Usage

### MLP Baseline

```bash
uv run python src/integration/unimodal_baseline/protein_baseline.py \
  --model-type mlp \
  --model-path src/protein/runs/run_20251003_133215/models/neural_network.pkl \
  --scaler-path src/protein/runs/run_20251003_133215/scaler.pkl \
  --data-csv /home/ssim0068/data/multimodal-dataset/all.csv \
  --cv-splits-json runs/fusion_mlp_5fold_cv/cv_splits.json \
  --save-dir runs/baseline_protein_mlp_5fold_cv
```
uv run python src/integration/unimodal_baseline/protein_baseline.py \
  --cv-splits-json multimodal-AD/runs/fusion_simple_nn_5fold_cv/cv_splits.json

```bash
uv run python src/integration/unimodal_baseline/protein_baseline.py \
  --model-type mlp \
  --model-path src/protein/runs/run_20251009_113510/models/neural_network.pkl \
  --scaler-path src/protein/runs/run_20251009_113510/scaler.pkl \
  --data-csv /home/ssim0068/data/multimodal-dataset/all.csv \
  --cv-splits-json runs/fusion_mlp_5fold_cv/cv_splits.json \
  --save-dir runs/baseline_protein_mlp_5fold_cv_run_20251009_113510
```

### Transformer Baseline

```bash
uv run python src/integration/unimodal_baseline/protein_baseline.py \
  --model-type transformer \
  --model-path src/protein/runs/run_20251003_133215/models/protein_transformer.pth \
  --scaler-path src/protein/runs/run_20251003_133215/scaler.pkl \
  --data-csv /home/ssim0068/data/multimodal-dataset/all.csv \
  --cv-splits-json runs/fusion_transformer_5fold_cv/cv_splits.json \
  --save-dir runs/baseline_protein_transformer_5fold_cv
```

## Arguments

- `--model-type`: Type of protein model (`mlp` or `transformer`)
- `--model-path`: Path to pre-trained model (`.pkl` for MLP, `.pth` for Transformer)
- `--scaler-path`: Path to pre-fitted scaler (`.pkl` file)
- `--data-csv`: Path to CSV with protein data
- `--cv-splits-json`: Path to `cv_splits.json` from fusion experiment
- `--save-dir`: Directory to save baseline results

## Output

Results are saved to `{save_dir}/protein_baseline_results.json` with:
- Per-fold test metrics (Accuracy, Balanced Accuracy, AUC, Precision, Recall, F1)
- Aggregated metrics (mean ± std across folds)
- Aggregated confusion matrix
- Model configuration

## Comparison with Fusion

Compare results directly:
- **Fusion**: `runs/fusion_{model_type}_5fold_cv/aggregated_results.json`
- **Protein baseline**: `runs/baseline_protein_{model_type}_5fold_cv/protein_baseline_results.json`


## Unimodal MRI Baseline Evaluation
MRI baseline utilizes the pretrained SimCLR BrainIAC model followed by the simple MLP model used for linear probing. 

```bash
uv run python -m src.integration.unimodal_baseline.mri_linear_probe_baseline \
  --data-csv /home/ssim0068/data/multimodal-dataset/all.csv \
  --cv-splits-json /home/ssim0068/multimodal-AD/runs/fusion_weighted_attention_nn_5fold_cv/cv_splits.json \
  --checkpoint-path /home/ssim0068/multimodal-AD/src/mri/BrainIAC/src/logs/mni305/cn_ad_linear_probe_mni305_batch32-epoch=84-val_auc=0.77.ckpt \
  --simclr-ckpt-path /home/ssim0068/multimodal-AD/src/mri/BrainIAC/src/checkpoints/BrainIAC.ckpt \
  --save-dir /home/ssim0068/multimodal-AD/runs/mri_baseline_eval \
  --device cpu \
  --image-size 96 96 96
```