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

## Utility Script: `mri_extract_full.py`

**Purpose**: High-performance extraction of DICOM files from large ZIP archives with Windows path length handling.

**Key Features**:
- Filters for specific scan types (MPRAGE/MP-RAGE by default)
- Handles Windows MAX_PATH limitations by intelligently shortening filenames
- Preserves original directory structure from ZIP archives
- Multiprocessing support for faster extraction
- Resume functionality (skips already extracted files)
- CRC-based filename uniqueness when shortening paths

**Usage**:
```bash
# Basic usage with defaults
python mri_extract_full.py

# Custom zip and output directory
python mri_extract_full.py <zip_file> <output_dir>

# With overwrite and custom worker count
python mri_extract_full.py <zip_file> <output_dir> --overwrite --workers 4
```

**Configuration**:
- Default input: `AD_CN_all_available_data_91gb.zip`
- Default output: `C:\Users\User\github_repos\AD_CN_all_available_data_133gb_mprage`
- Scan filter: MPRAGE/MP-RAGE scans only
- Max path length: 240 characters (Windows safe)

## Analysis Notebook: `mri_extraction.ipynb`

**Purpose**: Interactive Jupyter notebook for MRI data exploration, extraction, and subject sampling.

**Key Features**:
- **MPRAGE File Extraction**: Extract MPRAGE/MP-RAGE files from multiple ZIP archives
- **Subject ID Analysis**: Count and analyze subject IDs across multiple ZIP files
- **Data Sampling**: Randomly sample additional subjects for training (100 AD + 100 CN)
- **Subject Filtering**: Remove subjects already used in proteomic analysis
- **Interactive Analysis**: Step-by-step data exploration with visual feedback

**Main Functions**:
- `extract_mprage_files()` - Extract MPRAGE files maintaining directory structure
- `count_subject_ids_at_level()` - Analyze subject distribution across ZIP files
- Subject sampling and filtering for additional training data

**Use Cases**:
- Exploratory data analysis of MRI archives
- Creating additional training datasets
- Understanding data distribution across multiple ZIP files
- Interactive debugging of extraction processes

## Key Differences: `mri_extract_full.py` vs `mri_extraction.ipynb`

| Aspect | `mri_extract_full.py` | `mri_extraction.ipynb` |
|--------|----------------------|------------------------|
| **Type** | Production Python script | Interactive Jupyter notebook |
| **Purpose** | High-performance bulk extraction | Exploratory analysis & sampling |
| **Scale** | Large archives (91GB+) | Smaller ZIP files (5 archives) |
| **Performance** | Multiprocessing, optimized I/O | Single-threaded, simple extraction |
| **Path Handling** | Windows MAX_PATH solutions | Basic path handling |
| **Resume Support** | Yes (skip existing files) | No resume functionality |
| **Error Handling** | Robust production-level | Basic try-catch |
| **Use Case** | Initial data extraction | Data exploration & additional sampling |
| **Output** | Organized DICOM files | Subject analysis & sample lists |
| **Automation** | Fully automated | Interactive, step-by-step |

**When to use `mri_extract_full.py`:**
- Extracting from massive ZIP archives (>50GB)
- Production data processing pipelines
- Need for speed and reliability
- Windows path length issues
- Automated/batch processing

**When to use `mri_extraction.ipynb`:**
- Exploring data structure and contents
- Sampling additional subjects for training
- Interactive analysis and debugging
- Understanding data distribution
- Prototyping extraction logic

## Data Files

**CSFMRM_23Jun2025.csv**: Raw proteomic data from CSF MRM measurements
**proteomic_mri_with_labels.csv**: Proteomic data merged with subject labels and visit information

## Directory Structure
```
extracted_images/          # Source MRI DICOM files
merged_proteomic_mri_mprage.csv    # Final matched dataset
AD_CN_MRI_final/           # Organized MRI images for matched subjects
```