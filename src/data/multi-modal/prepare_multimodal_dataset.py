#!/usr/bin/env python3
"""
Prepare multimodal dataset by:
1. Flattening nested MRI directory structure
2. Creating balanced train/test splits
3. Generating CSV files with paired protein-MRI data
"""

import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np

def find_mri_file(subject_id, base_dir):
    """
    Find the MRI file for a given subject in the nested structure.
    
    Args:
        subject_id: Subject ID (e.g., '002_S_0413')
        base_dir: Base directory with nested structure
    
    Returns:
        Path to .nii.gz file or None if not found
    """
    subject_dir = Path(base_dir) / subject_id
    
    if not subject_dir.exists():
        return None
    
    # Find all .nii.gz files in subdirectories
    nii_files = list(subject_dir.rglob("*.nii.gz"))
    
    if len(nii_files) == 0:
        return None
    elif len(nii_files) > 1:
        print(f"  Warning: Multiple scans found for {subject_id}, using first one")
    
    return nii_files[0]

def create_balanced_split(df, test_size=8, random_state=42):
    """
    Create balanced train/test split considering:
    - research_group (AD/CN balance)
    - Age distribution
    - Sex distribution
    
    Args:
        df: DataFrame with metadata
        test_size: Number of test samples
        random_state: Random seed
    
    Returns:
        train_df, test_df
    """
    # Create stratification column combining group and sex
    df['stratify_col'] = df['research_group'].astype(str) + '_' + df['Sex'].astype(str)
    
    # Try to split with stratification
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['stratify_col']
        )
    except ValueError as e:
        print(f"Warning: Stratification failed ({e}), using random split")
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state
        )
    
    # Remove temporary column
    train_df = train_df.drop('stratify_col', axis=1)
    test_df = test_df.drop('stratify_col', axis=1)
    
    return train_df, test_df

def print_split_statistics(train_df, test_df):
    """Print statistics about the train/test split"""
    print("\n" + "="*60)
    print("SPLIT STATISTICS")
    print("="*60)
    
    for name, split_df in [("TRAIN", train_df), ("TEST", test_df)]:
        print(f"\n{name} SET (n={len(split_df)}):")
        print("-" * 40)
        
        # Group distribution
        group_counts = split_df['research_group'].value_counts()
        print(f"  Research groups:")
        for group, count in group_counts.items():
            pct = count / len(split_df) * 100
            print(f"    {group}: {count} ({pct:.1f}%)")
        
        # Sex distribution
        sex_counts = split_df['Sex'].value_counts()
        print(f"  Sex:")
        for sex, count in sex_counts.items():
            pct = count / len(split_df) * 100
            print(f"    {sex}: {count} ({pct:.1f}%)")
        
        # Age statistics
        print(f"  Age: {split_df['Age'].mean():.1f} ± {split_df['Age'].std():.1f} years")

def main():
    # Configuration
    input_csv = "/home/ssim0068/data/preprocessed/AD_CN_MRI_final/merged_proteomic_mri_mprage.csv"
    source_mri_dir = "/home/ssim0068/data/preprocessed/AD_CN_MRI_final"
    output_base = "/home/ssim0068/data/multimodal-dataset"
    test_size = 8
    random_state = 42
    
    print("="*60)
    print("MULTIMODAL DATASET PREPARATION")
    print("="*60)
    print(f"Input CSV: {input_csv}")
    print(f"Source MRI dir: {source_mri_dir}")
    print(f"Output dir: {output_base}")
    print(f"Test size: {test_size} samples")
    
    # Read merged CSV
    print(f"\n📊 Loading data...")
    df = pd.read_csv(input_csv)
    print(f"  Total samples: {len(df)}")
    
    # Find MRI files for each subject
    print(f"\n🔍 Finding MRI files...")
    mri_paths = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        subject_id = row['Subject']
        mri_path = find_mri_file(subject_id, source_mri_dir)
        
        if mri_path is not None:
            mri_paths.append(str(mri_path))
            valid_indices.append(idx)
            print(f"  ✓ {subject_id}: Found")
        else:
            print(f"  ✗ {subject_id}: NOT FOUND")
    
    # Filter to valid samples
    df = df.loc[valid_indices].copy()
    df['mri_source_path'] = mri_paths
    
    print(f"\n✅ Valid samples: {len(df)}/{len(valid_indices)}")
    
    # Create balanced train/test split
    print(f"\n📂 Creating train/test split...")
    train_df, test_df = create_balanced_split(df, test_size=test_size, random_state=random_state)
    
    # Print statistics
    print_split_statistics(train_df, test_df)
    
    # Create output directories
    print(f"\n📁 Creating output directories...")
    output_path = Path(output_base)
    train_img_dir = output_path / "train" / "images"
    test_img_dir = output_path / "test" / "images"
    
    train_img_dir.mkdir(parents=True, exist_ok=True)
    test_img_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Created: {train_img_dir}")
    print(f"  Created: {test_img_dir}")
    
    # Copy/link MRI files and update paths
    print(f"\n📋 Organizing MRI files...")
    
    def process_split(split_df, img_dir, split_name):
        new_paths = []
        for idx, row in split_df.iterrows():
            subject_id = row['Subject']
            source_path = Path(row['mri_source_path'])
            
            # New filename: {Subject}.nii.gz
            dest_path = img_dir / f"{subject_id}.nii.gz"
            
            # Copy file
            if not dest_path.exists():
                shutil.copy2(source_path, dest_path)
                print(f"  {split_name}: {subject_id} copied")
            else:
                print(f"  {split_name}: {subject_id} already exists")
            
            new_paths.append(str(dest_path))
        
        return new_paths
    
    train_df['mri_path'] = process_split(train_df, train_img_dir, "TRAIN")
    test_df['mri_path'] = process_split(test_df, test_img_dir, "TEST")
    
    # Save CSVs with all metadata
    print(f"\n💾 Saving CSV files...")
    train_csv_path = output_path / "train.csv"
    test_csv_path = output_path / "test.csv"
    
    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    
    print(f"  Saved: {train_csv_path}")
    print(f"  Saved: {test_csv_path}")
    
    # Print final summary
    print("\n" + "="*60)
    print("✅ DATASET PREPARATION COMPLETE!")
    print("="*60)
    print(f"Output directory: {output_path}")
    print(f"  train.csv: {len(train_df)} samples")
    print(f"  test.csv: {len(test_df)} samples")
    print(f"  train/images/: {len(list(train_img_dir.glob('*.nii.gz')))} files")
    print(f"  test/images/: {len(list(test_img_dir.glob('*.nii.gz')))} files")
    
    # Show CSV columns
    print(f"\nCSV columns available:")
    for col in train_df.columns:
        print(f"  - {col}")
    
    print(f"\n🎯 Ready for multimodal fusion training!")

if __name__ == "__main__":
    main()

