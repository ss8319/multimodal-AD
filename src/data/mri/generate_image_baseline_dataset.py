#!/usr/bin/env python3
"""
Combine MCI CSV files and remove subjects that overlap with proteomic dataset
"""

import csv
import os

def read_csv_file(filepath):
    """Read CSV file and return list of dictionaries"""
    data = []
    with open(filepath, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

def main():
    # File paths
    mci_files = [
        '/home/ssim0068/data/ADNI_v2/csvs/mci_train.csv',
        '/home/ssim0068/data/ADNI_v2/csvs/mci_val.csv',
        '/home/ssim0068/data/ADNI_v2/csvs/mci_test.csv'
    ]
    
    proteomic_file = '/home/ssim0068/multimodal-AD/src/data/protein/merged_proteomic_mri_mprage.csv'
    output_file = '/home/ssim0068/data/ADNI_v2/csvs/image_baseline.csv'
    
    print("Reading MCI CSV files...")
    combined_mci = []
    for file_path in mci_files:
        data = read_csv_file(file_path)
        combined_mci.extend(data)
        print(f"  {os.path.basename(file_path)}: {len(data)} rows")
    
    print(f"Combined MCI: {len(combined_mci)} rows")
    
    print("Reading proteomic dataset...")
    proteomic_data = read_csv_file(proteomic_file)
    print(f"Proteomic dataset: {len(proteomic_data)} rows")
    
    # Get list of subjects in proteomic dataset
    proteomic_subjects = set()
    for row in proteomic_data:
        subject = row.get('Subject', '').strip()
        if subject:
            proteomic_subjects.add(subject)
    
    print(f"Proteomic subjects: {len(proteomic_subjects)}")
    
    # Filter out subjects that are in proteomic dataset
    filtered_mci = []
    removed_count = 0
    
    for row in combined_mci:
        pat_id = row.get('pat_id', '').strip()
        if pat_id not in proteomic_subjects:
            filtered_mci.append(row)
        else:
            removed_count += 1
    
    print(f"Removed {removed_count} overlapping subjects")
    print(f"Remaining MCI subjects: {len(filtered_mci)}")
    
    # Calculate demographics for filtered dataset
    male_count = sum(1 for r in filtered_mci if r.get('Sex', '').strip().upper() == 'M')
    female_count = sum(1 for r in filtered_mci if r.get('Sex', '').strip().upper() == 'F')
    ad_count = sum(1 for r in filtered_mci if str(r.get('label', '')).strip() == '1')
    cn_count = sum(1 for r in filtered_mci if str(r.get('label', '')).strip() == '0')
    
    # Age statistics
    ages = []
    for r in filtered_mci:
        try:
            age = float(r.get('Age', ''))
            if age > 0:
                ages.append(age)
        except:
            pass
    
    age_mean = sum(ages) / len(ages) if ages else 0
    age_std = (sum((x - age_mean) ** 2 for x in ages) / len(ages)) ** 0.5 if len(ages) > 1 else 0
    
    print("\nFiltered MCI Dataset Demographics:")
    print(f"Total subjects: {len(filtered_mci)}")
    print(f"Male: {male_count}")
    print(f"Female: {female_count}")
    print(f"AD (label=1): {ad_count}")
    print(f"CN (label=0): {cn_count}")
    print(f"Age mean: {age_mean:.2f}")
    print(f"Age std: {age_std:.2f}")
    
    # Save filtered dataset
    if filtered_mci:
        with open(output_file, 'w', newline='') as file:
            fieldnames = ['pat_id', 'label', 'Sex', 'Age']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(filtered_mci)
        
        print(f"\nFiltered dataset saved to: {output_file}")
    else:
        print("\nNo data to save")

if __name__ == "__main__":
    main()
