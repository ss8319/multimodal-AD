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

**Script**: `generate_proteomic_encoder_train_test_split.py`

**Purpose**: Creates stratified train/test splits for proteomic encoder training with configurable demographic stratification

**Key Features**:
- **Configurable stratification**: Choose any combination of diagnosis, age, and sex
- **Flexible test size**: Adjustable test set proportion (default: 20%)
- **Comprehensive balance verification**: Only checks stratified features
- **Single-sample handling**: Automatically assigns single-sample groups to training set. But shouldn't happen 
- **Reproducible**: Configurable random seed

**Usage Examples**:

```bash
# Default: stratify by all demographics (diagnosis, age, sex)
uv run python generate_proteomic_encoder_train_test_split.py

# Stratify by diagnosis only
uv run python generate_proteomic_encoder_train_test_split.py --stratify-by diagnosis

# Stratify by diagnosis and age only
uv run python generate_proteomic_encoder_train_test_split.py --stratify-by diagnosis age

**Command-line Options**:

- `--stratify-by`: Demographic features to stratify by (`diagnosis`, `age`, `sex`)
- `--test-size`: Proportion for test set (default: 0.20)
- `--random-state`: Random seed (default: 42)
- `--input-path`: Input CSV file path
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