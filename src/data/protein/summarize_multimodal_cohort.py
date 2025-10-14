#!/usr/bin/env python3
"""
Summarize proteomic cohort statistics from merged_proteomic_mri_mprage.csv
"""

import csv
import math
from statistics import mean, pstdev

def main():
    # Read CSV
    rows = []
    with open('/home/ssim0068/multimodal-AD/src/data/protein/merged_proteomic_mri_mprage.csv', 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Count AD vs CN (research_group)
    ad_count = sum(1 for r in rows if r.get('research_group', '').strip() == 'AD')
    cn_count = sum(1 for r in rows if r.get('research_group', '').strip() == 'CN')
    
    # Count M vs F (Sex)
    male_count = sum(1 for r in rows if r.get('Sex', '').strip().upper() == 'M')
    female_count = sum(1 for r in rows if r.get('Sex', '').strip().upper() == 'F')
    
    # Age statistics
    ages = []
    for r in rows:
        try:
            age = float(r.get('Age', ''))
            if not math.isnan(age):
                ages.append(age)
        except:
            pass
    
    age_mean = mean(ages) if ages else float('nan')
    age_std = pstdev(ages) if len(ages) > 1 else 0.0
    
    # Print summary
    print("Proteomic Cohort Summary")
    print(f"Total subjects: {len(rows)}")
    print(f"AD: {ad_count}")
    print(f"CN: {cn_count}")
    print(f"Male: {male_count}")
    print(f"Female: {female_count}")
    print(f"Age mean: {age_mean:.2f}")
    print(f"Age std: {age_std:.2f}")

if __name__ == "__main__":
    main()
