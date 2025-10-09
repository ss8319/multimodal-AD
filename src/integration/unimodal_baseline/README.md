# Unimodal Protein Baseline Evaluation

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