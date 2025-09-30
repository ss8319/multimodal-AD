# Protein Data Organisation

## Files

- `raw/CSFMRM.csv`
  - Raw proteomics data (after QC)
  - 303 subjects (303 rows)

- `proteomic_w_labels.csv`
  - From `CSFMRM.csv`, merged with AD/CN labels
  - 158 subjects (~5 duplicates originally in raw)

- `merged_proteomic_mri_mprage.csv`
  - Proteomics subjects with corresponding MRI info and AD/CN labels
  - 38 subjects
  - Used for fusion experiments (6:2:2 split)
  - Images in `github_repos\AD_CN_MRI_final`

- `proteomic_encoder_data.csv`
  - Data to train the protein encoder
  - From `proteomic_w_labels.csv`, remove RIDs present in `merged_proteomic_mri_mprage.csv`
  - 153 − 38 = 115 subjects (target)
  - Intended for 8:1:1 train/val/test (downstream)

## Generate encoder data

```bash
uv run python src/data/protein/generate_proteomic_encoder_data.py \
  --csv1 "src/data/protein/proteomic_w_labels.csv" \
  --csv2 "src/data/protein/merged_proteomic_mri_mprage.csv" \
  --out "src/data/protein/proteomic_encoder_data.csv"
```

Reported stats for `proteomic_encoder_data.csv`:

- Age
  - Mean: 75.96 years
  - Std:  6.33 years
- Diagnosis distribution
  - CN: 63 (54.8%)
  - AD: 52 (45.2%)

## Train/test split for encoder

Script: `generate_proteomic_encoder_train_test_split.py` (reads `proteomic_encoder_data.csv` and creates balanced splits)

Split sizes:

- Train: 97 samples (84.3%) → `proteomic_encoder_train.csv`
- Test:  18 samples (15.7%) → `proteomic_encoder_test.csv`

Train set (reported):

- Age: 75.97 ± 6.24 years
- Diagnosis: CN 53 (54.6%), AD 44 (45.4%)

Test set (reported):

- Age: 75.86 ± 6.96 years
- Diagnosis: CN 10 (55.6%), AD 8 (44.4%)