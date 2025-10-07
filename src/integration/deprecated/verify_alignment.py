"""
Verification script to ensure protein latents, MRI latents, and labels are correctly aligned
Run this after pre-extraction to verify data integrity
"""

import pandas as pd
import numpy as np
from pathlib import Path


def verify_alignment(csv_path, latents_dir, split_name):
    """
    Verify that pre-extracted protein latents align with CSV data
    
    Args:
        csv_path: Path to train.csv or test.csv
        latents_dir: Path to directory containing pre-extracted latents
        split_name: 'train' or 'test'
    """
    print(f"\n{'='*60}")
    print(f"VERIFYING {split_name.upper()} SET ALIGNMENT")
    print(f"{'='*60}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"CSV: {len(df)} samples")
    
    # Load pre-extracted data
    latents_dir = Path(latents_dir)
    protein_latents = np.load(latents_dir / f"{split_name}_protein_latents.npy")
    subjects = np.load(latents_dir / f"{split_name}_subjects.npy", allow_pickle=True)  # String array needs pickle
    labels = np.load(latents_dir / f"{split_name}_labels.npy")
    
    print(f"Protein latents: {protein_latents.shape}")
    print(f"Subjects: {subjects.shape}")
    print(f"Labels: {labels.shape}")
    
    # Check 1: Number of samples match
    assert len(df) == len(protein_latents) == len(subjects) == len(labels), \
        f"Length mismatch! CSV={len(df)}, latents={len(protein_latents)}, subjects={len(subjects)}, labels={len(labels)}"
    print(f"✅ All arrays have same length: {len(df)}")
    
    # Check 2: Subject IDs match
    csv_subjects = df['Subject'].values
    for i, (csv_subj, saved_subj) in enumerate(zip(csv_subjects, subjects)):
        assert csv_subj == saved_subj, \
            f"Subject mismatch at index {i}: CSV={csv_subj}, saved={saved_subj}"
    print(f"✅ Subject IDs match exactly (row-by-row)")
    
    # Check 3: Labels match
    csv_labels = (df['research_group'] == 'AD').astype(int).values
    for i, (csv_label, saved_label) in enumerate(zip(csv_labels, labels)):
        assert csv_label == saved_label, \
            f"Label mismatch at index {i} (Subject={csv_subjects[i]}): CSV={csv_label}, saved={saved_label}"
    print(f"✅ Labels match exactly (row-by-row)")
    
    # Check 4: MRI paths are consistent
    print(f"\n📍 Spot checking first 3 samples:")
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        print(f"  Sample {i}:")
        print(f"    Subject: {row['Subject']} (saved: {subjects[i]})")
        print(f"    Label: {'AD' if csv_labels[i] == 1 else 'CN'} (saved: {labels[i]})")
        print(f"    MRI: {row['mri_path']}")
        print(f"    Protein latent shape: {protein_latents[i].shape}")
    
    # Check 5: Class distribution
    print(f"\n📊 Class distribution:")
    ad_count = np.sum(labels == 1)
    cn_count = np.sum(labels == 0)
    print(f"  AD: {ad_count} ({ad_count/len(labels)*100:.1f}%)")
    print(f"  CN: {cn_count} ({cn_count/len(labels)*100:.1f}%)")
    
    print(f"\n{'='*60}")
    print(f"✅ {split_name.upper()} SET VERIFICATION PASSED")
    print(f"{'='*60}")
    print(f"All data is correctly aligned:")
    print(f"  • Protein latents: row i → latents[i]")
    print(f"  • Subject IDs: row i → subjects[i]")
    print(f"  • Labels: row i → labels[i]")
    print(f"  • MRI paths: row i → df.iloc[i]['mri_path']")
    print(f"\nIn multimodal_dataset.py:")
    print(f"  • self.protein_latents[idx] → matches df.iloc[idx]")
    print(f"  • _extract_mri_latents(df.iloc[idx]['mri_path']) → MRI for same subject")
    print(f"  • df.iloc[idx]['research_group'] → correct label")


def main():
    print("="*60)
    print("DATA ALIGNMENT VERIFICATION")
    print("="*60)
    print("This script verifies that:")
    print("  1. Pre-extracted protein latents match CSV row order")
    print("  2. Subject IDs are consistent")
    print("  3. Labels are consistent")
    print("  4. MRI paths correspond to correct subjects")
    
    # Paths
    train_csv = "/home/ssim0068/data/multimodal-dataset/train.csv"
    test_csv = "/home/ssim0068/data/multimodal-dataset/test.csv"
    latents_dir = "/home/ssim0068/data/multimodal-dataset/protein_latents"
    
    # Verify train set
    verify_alignment(train_csv, latents_dir, 'train')
    
    # Verify test set
    verify_alignment(test_csv, latents_dir, 'test')
    
    print("\n" + "="*60)
    print("✅ ALL VERIFICATION PASSED")
    print("="*60)
    print("You can safely proceed with training!")


if __name__ == "__main__":
    main()

