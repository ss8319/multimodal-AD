# ADNI Proteomic-MRI Data Processing Pipeline

This repository contains scripts for matching and processing ADNI proteomic and MRI data for machine learning applications.

## Main Script: `match_mprage_simple.py`

**Purpose**: Matches proteomic data with MPRAGE MRI scans and copies corresponding image files to an organized directory structure.

**Key Features**:
- Matches proteomic subjects with MRI scans based on RID and visit codes
- Filters for MPRAGE/MP-RAGE scans only
- Handles duplicate scans (keeps first occurrence per subject)
- Organizes CSV columns: metadata first, then protein data
- Copies matched MRI image folders maintaining directory structure

**Input Files**:
- `proteomic_mri_with_labels.csv` - Proteomic data with subject labels
- `AD_CN_all_available_data.csv` - MRI metadata from ADNI database

**Output**:
- `merged_proteomic_mri_mprage.csv` - Matched dataset with organized columns
- `C:\Users\User\github_repos\AD_CN_MRI_final\` - Copied MRI image folders

**Usage**:
```bash
python match_mprage_simple.py
```

## Data Files

**CSFMRM_23Jun2025.csv**: Raw proteomic data from CSF MRM measurements
**proteomic_mri_with_labels.csv**: Proteomic data merged with subject labels and visit information

## Directory Structure
```
extracted_images/          # Source MRI DICOM files
merged_proteomic_mri_mprage.csv    # Final matched dataset
AD_CN_MRI_final/           # Organized MRI images for matched subjects
```