#!/usr/bin/env python3
"""
Merge Subject_Demographics.csv with proteomic_encoder_data.csv based on RID
Add Sex column from PTGENDER (1=Male, 2=Female)
Calculate demographic statistics

Usage:
    python merge_demographics.py                    # Drop unknown sex rows (default)
    python merge_demographics.py --keep-unknown-sex # Keep unknown sex rows
"""

import csv
import math
from statistics import mean, pstdev

def read_csv_file(filepath):
    """Read CSV file and return list of dictionaries"""
    data = []
    with open(filepath, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

def merge_data(demographics, proteomic, keep_unknown_sex=False):
    """Merge demographics and proteomic data based on RID"""
    # Create lookup dictionary for demographics by RID
    demo_lookup = {}
    for row in demographics:
        rid = row.get('RID', '').strip()
        if rid:
            demo_lookup[rid] = row
    
    # Merge proteomic data with demographics
    merged_data = []
    unknown_sex_count = 0
    
    for row in proteomic:
        rid = row.get('RID', '').strip()
        if rid in demo_lookup:
            # Create new row with proteomic data
            new_row = row.copy()
            
            # Add Sex column from PTGENDER (1=Male, 2=Female)
            ptgender = demo_lookup[rid].get('PTGENDER', '').strip()
            if ptgender == '1':
                new_row['Sex'] = 'M'
            elif ptgender == '2':
                new_row['Sex'] = 'F'
            else:
                new_row['Sex'] = 'Unknown'
                unknown_sex_count += 1
            
            # Add other useful demographic info
            new_row['Age'] = demo_lookup[rid].get('PTDOBYY', '')
            new_row['Education'] = demo_lookup[rid].get('PTEDUCAT', '')
            
            # Only add row if keeping unknown sex or if sex is known
            if keep_unknown_sex or new_row['Sex'] != 'Unknown':
                merged_data.append(new_row)
        else:
            print(f"Warning: RID {rid} not found in demographics data")
    
    if not keep_unknown_sex and unknown_sex_count > 0:
        print(f"Dropped {unknown_sex_count} rows with Unknown Sex")
    
    return merged_data

def calculate_demographics(data):
    """Calculate demographic statistics"""
    print("Merged Dataset Demographics:")
    print(f"Total subjects: {len(data)}")
    
    # Count Male vs Female
    male_count = sum(1 for r in data if r.get('Sex', '').strip().upper() == 'M')
    female_count = sum(1 for r in data if r.get('Sex', '').strip().upper() == 'F')
    unknown_sex = len(data) - male_count - female_count
    
    print(f"Male: {male_count}")
    print(f"Female: {female_count}")
    if unknown_sex > 0:
        print(f"Unknown Sex: {unknown_sex}")
    
    # Count AD vs CN (using research_group column)
    ad_count = sum(1 for r in data if str(r.get('research_group', '')).strip().upper() == 'AD')
    cn_count = sum(1 for r in data if str(r.get('research_group', '')).strip().upper() == 'CN')
    unknown_label = len(data) - ad_count - cn_count
    
    print(f"AD: {ad_count}")
    print(f"CN: {cn_count}")
    if unknown_label > 0:
        print(f"Unknown Label: {unknown_label}")
    
    # Age statistics (from subject_age column)
    ages = []
    for r in data:
        try:
            age = float(r.get('subject_age', ''))
            if not math.isnan(age) and age > 0:
                ages.append(age)
        except:
            pass
    
    if ages:
        age_mean = mean(ages)
        age_std = pstdev(ages) if len(ages) > 1 else 0.0
        print(f"Age mean: {age_mean:.2f}")
        print(f"Age std: {age_std:.2f}")
        print(f"Age range: {min(ages)}-{max(ages)}")
    else:
        print("Age: No valid age data found")

def save_merged_data(data, output_file):
    """Save merged data to CSV with organized column order"""
    if not data:
        print("No data to save")
        return
    
    # Get all unique column names
    all_columns = set()
    for row in data:
        all_columns.update(row.keys())
    
    # Define metadata columns (put these first)
    metadata_columns = [
        'RID', 'Subject', 'VISCODE', 'Visit', 'research_group', 'Group', 
        'Sex', 'Age', 'subject_age', 'Education', 'PTGENDER', 'PTDOBYY', 'PTEDUCAT',
        'Image Data ID', 'Description', 'Type', 'Modality', 'Format', 
        'Acq Date', 'Downloaded', 'MRI_acquired'
    ]
    
    # Get protein/peptide columns (everything else)
    protein_columns = []
    for col in sorted(all_columns):
        if col not in metadata_columns:
            protein_columns.append(col)
    
    # Combine: metadata first, then protein columns
    columns = []
    for col in metadata_columns:
        if col in all_columns:
            columns.append(col)
    
    columns.extend(protein_columns)
    
    with open(output_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"\nMerged data saved to: {output_file}")
    print(f"Total columns: {len(columns)}")
    print(f"Metadata columns: {len([c for c in columns if c in metadata_columns])}")
    print(f"Protein/peptide columns: {len(protein_columns)}")

def main():
    import sys
    
    # Parse command line arguments
    keep_unknown_sex = False
    if len(sys.argv) > 1 and sys.argv[1] == '--keep-unknown-sex':
        keep_unknown_sex = True
        print("Flag: Keeping rows with Unknown Sex")
    else:
        print("Flag: Dropping rows with Unknown Sex (default)")
    
    # File paths
    demographics_file = '/home/ssim0068/multimodal-AD/src/data/protein/Subject_Demographics.csv'
    proteomic_file = '/home/ssim0068/multimodal-AD/src/data/protein/proteomic_encoder_data.csv'
    output_file = '/home/ssim0068/multimodal-AD/src/data/protein/proteomic_with_demographics.csv'
    
    print("Reading Subject_Demographics.csv...")
    demographics = read_csv_file(demographics_file)
    print(f"Demographics data: {len(demographics)} rows")
    
    print("Reading proteomic_encoder_data.csv...")
    proteomic = read_csv_file(proteomic_file)
    print(f"Proteomic data: {len(proteomic)} rows")
    
    print("\nMerging data based on RID...")
    merged_data = merge_data(demographics, proteomic, keep_unknown_sex)
    print(f"Merged data: {len(merged_data)} rows")
    
    print("\n" + "="*50)
    calculate_demographics(merged_data)
    
    print("\n" + "="*50)
    save_merged_data(merged_data, output_file)

if __name__ == "__main__":
    main()
