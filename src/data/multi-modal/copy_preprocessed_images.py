#!/usr/bin/env python3
"""
Copy ICBM152 preprocessed images from ADNI_v2 to multimodal dataset directory.

This script:
1) Reads 'Subject' column from data/multimodal-dataset/all.csv
2) Finds matching Subject in data/ADNI_v2/images_icbm152
3) Copies the image to data/multimodal-dataset/all_mni/images
"""

import shutil
from pathlib import Path
import pandas as pd


def main():
    # Paths
    csv_path = Path("/home/ssim0068/data/multimodal-dataset/all_mni.csv")
    source_dir = Path("/home/ssim0068/data/ADNI_v2/images_mni305")
    dest_dir = Path("/home/ssim0068/data/multimodal-dataset/all_mni/images")
    
    # Create destination directory
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Copy ICBM152 Images for Multimodal Dataset")
    print("="*80)
    print()
    
    # Read CSV
    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    subjects = df['Subject'].tolist()
    
    print(f"Found {len(subjects)} subjects in CSV")
    print()
    
    # Copy images
    copied = 0
    missing = []
    
    print(f"Source: {source_dir}")
    print(f"Destination: {dest_dir}")
    print()
    print("Copying images...")
    print("-"*80)
    
    for subject in subjects:
        source_file = source_dir / f"{subject}.nii.gz"
        dest_file = dest_dir / f"{subject}.nii.gz"
        
        if not source_file.exists():
            print(f"❌ {subject}: Source file not found")
            missing.append(subject)
            continue
        
        # Copy file
        shutil.copy2(source_file, dest_file)
        copied += 1
        print(f"✅ {subject}")
    
    print("-"*80)
    print()
    print("SUMMARY:")
    print(f"  Total subjects: {len(subjects)}")
    print(f"  Copied: {copied}")
    print(f"  Missing: {len(missing)}")
    
    if missing:
        print()
        print("Missing subjects:")
        for subj in missing:
            print(f"  - {subj}")
    
    print()
    print(f"✅ Images copied to: {dest_dir}")
    

if __name__ == "__main__":
    main()

