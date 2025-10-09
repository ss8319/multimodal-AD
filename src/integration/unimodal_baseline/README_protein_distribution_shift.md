bash
'''
uv run python src/integration/unimodal_baseline/protein_shift_analysis.py --train-csv src/data/protein/proteomic_encoder_train.csv --paired-csv /home/ssim0068/data/multimodal-dataset/all.csv --cv-splits-json runs/fusion_mlp_5fold_cv/cv_splits.json --save-path src/integration/unimodal_baseline/protein_shift_report.json
'''

## Overall Distribution Shift
mean_diff - Difference in mean values (Paired - Training):
Positive values: Paired samples have higher mean than training
Negative values: Paired samples have lower mean than training
Large magnitudes: Significant distribution shift
std_ratio - Ratio of standard deviations (Paired / Training):
> 1.0: Paired samples are more variable than training
< 1.0: Paired samples are less variable than training
Near 1.0: Similar variability

## Concerning Patterns
Looking at the data, I see several large shifts:
Mean differences (worst cases):
APOE_CLAVYQAGAR: -1.08 (paired much lower)
APOE_LGADMEDVR: -0.73 (paired much lower)
