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
  
  ```bash
  Proteomic + MRI Cohort Summary
  Total subjects: 38
  AD: 15
  CN: 23
  Male: 13
  Female: 25
  Age mean: 74.32
  Age std: 6.87
  ```
  - to generate statistic above run 'summarize_multimodal_cohort.py'

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

Script: `generate_proteomic_encoder_train_test_split.py` (reads `proteomic_with_demographics.csv` and creates balanced splits)

**Key Features**:
- Triple stratification by diagnosis, age (quantile-based), and sex

- Comprehensive balance verification
- Increased test size (20%) for better evaluation

**Split sizes** (106 subjects):
- Train: 84 samples (79.2%) → `proteomic_encoder_train.csv`
- Test: 22 samples (20.8%) → `proteomic_encoder_test.csv`

Train set (reported):

- Age: 75.97 ± 6.24 years
- Diagnosis: CN 53 (54.6%), AD 44 (45.4%)

Test set (reported):

- Age: 75.86 ± 6.96 years
- Diagnosis: CN 10 (55.6%), AD 8 (44.4%)

## Merge demographics with proteomic data

Script: `merge_demographics.py`

**Purpose**: Merges Subject_Demographics.csv with proteomic_encoder_data.csv based on RID to add demographic information (Sex, Age, Education) to the proteomic dataset.

**Inputs**:
- `Subject_Demographics.csv` - ADNI demographic data (6,212 rows)
- `proteomic_encoder_data.csv` - Proteomic data for encoder training (115 rows)

**Output**:
- `proteomic_with_demographics.csv` - Merged dataset with demographic columns

**Key Features**:
- Merges on RID field
- Adds Sex column from PTGENDER (1=Male, 2=Female)
- Adds Age from subject_age column
- Adds Education from PTEDUCAT
- Organized CSV output: metadata columns first, protein/peptide columns second
- Optional flag to keep/drop rows with Unknown Sex
- Calculates demographic statistics

**Usage**:
```bash
# Default: Drop unknown sex rows (106 subjects)
python src/data/protein/merge_demographics.py

# Keep all rows including unknown sex (115 subjects)
python src/data/protein/merge_demographics.py --keep-unknown-sex
```