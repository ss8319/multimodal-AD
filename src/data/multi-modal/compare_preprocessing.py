#!/usr/bin/env python3
"""
Compare NIfTI headers to determine which preprocessing (MNI305 or ICBM152)
was used for the multimodal dataset images: multimodal-dataset/all/images.
"""

import os
from pathlib import Path
import nibabel as nib
import numpy as np

# Directories
MULTIMODAL_DIR = Path("/home/ssim0068/data/multimodal-dataset/all_mni/images")
MNI305_DIR = Path("/home/ssim0068/data/ADNI_v2/images_mni305")
ICBM152_DIR = Path("/home/ssim0068/data/ADNI_v2/images_icbm152")


def get_header_info(nifti_path):
    """Extract key header information from a NIfTI file."""
    img = nib.load(str(nifti_path))
    header = img.header
    
    return {
        'shape': img.shape,
        'affine': img.affine,
        'voxel_size': header.get_zooms(),
        'qform_code': int(header['qform_code']),
        'sform_code': int(header['sform_code']),
    }


def compare_headers(h1, h2):
    """Compare two headers and return similarity metrics."""
    shape_match = h1['shape'] == h2['shape']
    affine_close = np.allclose(h1['affine'], h2['affine'], atol=1e-3)
    voxel_close = np.allclose(h1['voxel_size'], h2['voxel_size'], atol=1e-3)
    qform_match = h1['qform_code'] == h2['qform_code']
    sform_match = h1['sform_code'] == h2['sform_code']
    
    return {
        'shape_match': shape_match,
        'affine_close': affine_close,
        'voxel_close': voxel_close,
        'qform_match': qform_match,
        'sform_match': sform_match,
        'all_match': all([shape_match, affine_close, voxel_close]),
    }


def main():
    # Get subjects from multimodal dataset
    multimodal_files = sorted(MULTIMODAL_DIR.glob("*.nii.gz"))
    
    print("="*80)
    print("NIfTI Header Comparison: Multimodal vs MNI305 vs ICBM152")
    print("="*80)
    print()
    
    mni305_matches = 0
    icbm152_matches = 0
    total = 0
    
    # Sample first 5 subjects for detailed comparison
    print("DETAILED COMPARISON (first 5 subjects):")
    print("-"*80)
    
    for i, mm_file in enumerate(multimodal_files[:5]):
        subject = mm_file.stem.replace('.nii', '')
        print(f"\n[{i+1}] Subject: {subject}")
        
        # Paths
        mni305_path = MNI305_DIR / f"{subject}.nii.gz"
        icbm152_path = ICBM152_DIR / f"{subject}.nii.gz"
        
        # Check existence
        mni305_exists = mni305_path.exists()
        icbm152_exists = icbm152_path.exists()
        
        if not mni305_exists and not icbm152_exists:
            print(f"  ⚠️  Not found in either MNI305 or ICBM152")
            continue
        
        # Load headers
        mm_header = get_header_info(mm_file)
        print(f"  Multimodal shape: {mm_header['shape']}, voxel: {mm_header['voxel_size']}")
        
        if mni305_exists:
            mni305_header = get_header_info(mni305_path)
            mni305_comp = compare_headers(mm_header, mni305_header)
            print(f"  MNI305     shape: {mni305_header['shape']}, voxel: {mni305_header['voxel_size']}")
            print(f"    → Match: shape={mni305_comp['shape_match']}, "
                  f"affine={mni305_comp['affine_close']}, "
                  f"voxel={mni305_comp['voxel_close']}")
        
        if icbm152_exists:
            icbm152_header = get_header_info(icbm152_path)
            icbm152_comp = compare_headers(mm_header, icbm152_header)
            print(f"  ICBM152    shape: {icbm152_header['shape']}, voxel: {icbm152_header['voxel_size']}")
            print(f"    → Match: shape={icbm152_comp['shape_match']}, "
                  f"affine={icbm152_comp['affine_close']}, "
                  f"voxel={icbm152_comp['voxel_close']}")
    
    print("\n" + "="*80)
    print("SUMMARY (all subjects):")
    print("="*80)
    
    for mm_file in multimodal_files:
        subject = mm_file.stem.replace('.nii', '')
        
        mni305_path = MNI305_DIR / f"{subject}.nii.gz"
        icbm152_path = ICBM152_DIR / f"{subject}.nii.gz"
        
        if not mni305_path.exists() and not icbm152_path.exists():
            continue
        
        mm_header = get_header_info(mm_file)
        
        mni305_match = False
        icbm152_match = False
        
        if mni305_path.exists():
            mni305_header = get_header_info(mni305_path)
            mni305_comp = compare_headers(mm_header, mni305_header)
            mni305_match = mni305_comp['all_match']
        
        if icbm152_path.exists():
            icbm152_header = get_header_info(icbm152_path)
            icbm152_comp = compare_headers(mm_header, icbm152_header)
            icbm152_match = icbm152_comp['all_match']
        
        if mni305_match:
            mni305_matches += 1
        if icbm152_match:
            icbm152_matches += 1
        
        total += 1
    
    print(f"\nTotal subjects compared: {total}")
    print(f"MNI305 matches:  {mni305_matches} ({mni305_matches/total*100:.1f}%)")
    print(f"ICBM152 matches: {icbm152_matches} ({icbm152_matches/total*100:.1f}%)")
    print()
    
    if mni305_matches > icbm152_matches:
        print("✅ CONCLUSION: Multimodal dataset uses MNI305 preprocessing")
    elif icbm152_matches > mni305_matches:
        print("✅ CONCLUSION: Multimodal dataset uses ICBM152 preprocessing")
    else:
        print("⚠️  CONCLUSION: Mixed or unclear preprocessing")


if __name__ == "__main__":
    main()

