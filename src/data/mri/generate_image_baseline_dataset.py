#!/usr/bin/env python3
"""
Combine MCI CSV files and remove subjects that overlap with proteomic dataset
"""

import csv
import os
from typing import List, Dict, Tuple

try:
    from sklearn.model_selection import train_test_split
except Exception:
    # If sklearn isn't available, the script will raise a clear error when splitting
    train_test_split = None

def read_csv_file(filepath):
    """Read CSV file and return list of dictionaries"""
    data = []
    with open(filepath, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

def main():
    # File paths (inputs moved to deprecated_csvs)
    mci_files = [
        '/home/ssim0068/data/ADNI_v2/deprecated_csvs/mci_train.csv',
        '/home/ssim0068/data/ADNI_v2/deprecated_csvs/mci_val.csv',
        '/home/ssim0068/data/ADNI_v2/deprecated_csvs/mci_test.csv'
    ]
    
    proteomic_file = '/home/ssim0068/multimodal-AD/src/data/protein/merged_proteomic_mri_mprage.csv'
    # Consolidated dataset
    output_file = '/home/ssim0068/data/ADNI_v2/csvs/image_external_dataset.csv'
    
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

        # ---------- New: create stratified train/val/test splits ----------
        if train_test_split is None:
            raise RuntimeError("scikit-learn is required for stratified splitting. Please install scikit-learn.")

        def _quantile_thresholds(values: List[float], quantiles: List[float]) -> List[float]:
            """Compute approximate quantile thresholds from a list of floats.
            quantiles should be in [0,1], strictly increasing. Returns thresholds list.
            """
            if not values:
                return []
            vals = sorted(values)
            n = len(vals)
            cuts = []
            for q in quantiles:
                # nearest-rank method
                idx = int(round(q * (n - 1)))
                idx = max(0, min(n - 1, idx))
                cuts.append(vals[idx])
            return cuts

        def build_strat_key(rows: List[Dict]) -> List[str]:
            # Age bins by quartiles (4 bins). Missing/invalid ages -> 'UNK'
            ages = []
            for r in rows:
                try:
                    a = float(r.get('Age', ''))
                    if a > 0:
                        ages.append(a)
                except Exception:
                    pass
            # quartile thresholds at 25%, 50%, 75%
            qcuts = _quantile_thresholds(ages, [0.25, 0.50, 0.75]) if ages else []

            def age_to_bin(a: float) -> str:
                if not qcuts or a <= 0:
                    return 'UNK'
                if a <= qcuts[0]:
                    return 'A0'
                elif a <= qcuts[1]:
                    return 'A1'
                elif a <= qcuts[2]:
                    return 'A2'
                else:
                    return 'A3'

            keys = []
            for r in rows:
                label = str(r.get('label', '')).strip()
                sex = str(r.get('Sex', '')).strip().upper()
                try:
                    a = float(r.get('Age', ''))
                except Exception:
                    a = -1
                age_bin = age_to_bin(a)
                keys.append(f"{label}_{sex}_{age_bin}")
            return keys

        def safe_split(rows: List[Dict], test_size: float, random_state: int) -> Tuple[List[Dict], List[Dict]]:
            # Try full stratification: label + sex + age_bin
            keys = build_strat_key(rows)
            try:
                train_idx, test_idx = _split_with_keys(rows, keys, test_size, random_state)
                return [rows[i] for i in train_idx], [rows[i] for i in test_idx]
            except Exception:
                # Fallback: label + sex
                keys = [f"{str(r.get('label','')).strip()}_{str(r.get('Sex','')).strip().upper()}" for r in rows]
                try:
                    train_idx, test_idx = _split_with_keys(rows, keys, test_size, random_state)
                    return [rows[i] for i in train_idx], [rows[i] for i in test_idx]
                except Exception:
                    # Fallback: label only
                    keys = [str(r.get('label','')).strip() for r in rows]
                    train_idx, test_idx = _split_with_keys(rows, keys, test_size, random_state)
                    return [rows[i] for i in train_idx], [rows[i] for i in test_idx]

        def _split_with_keys(rows: List[Dict], keys: List[str], test_size: float, random_state: int):
            idx = list(range(len(rows)))
            train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=random_state, stratify=keys)
            return train_idx, test_idx

        # 80/10/10 split: first carve out test 10%, then val 10% of original from remaining
        all_rows = filtered_mci
        train_rows, test_rows = safe_split(all_rows, test_size=0.10, random_state=42)
        train_rows, val_rows = safe_split(train_rows, test_size=0.1111, random_state=43)  # 0.1111 of 90% ≈ 10% overall

        def write_csv(path: str, rows: List[Dict]):
            with open(path, 'w', newline='') as f:
                fieldnames = ['pat_id', 'label', 'Sex', 'Age']
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(rows)

        out_dir = '/home/ssim0068/data/ADNI_v2/csvs'
        train_path = os.path.join(out_dir, 'ad_train.csv')
        val_path = os.path.join(out_dir, 'ad_val.csv')
        test_path = os.path.join(out_dir, 'ad_test.csv')

        write_csv(train_path, train_rows)
        write_csv(val_path, val_rows)
        write_csv(test_path, test_rows)

        print("\nCreated stratified splits:")
        print(f"  Train: {len(train_rows)} -> {train_path}")
        print(f"  Val:   {len(val_rows)} -> {val_path}")
        print(f"  Test:  {len(test_rows)} -> {test_path}")
        
        # Summary demographics per split
        def summarize(rows: List[Dict]) -> Dict[str, float]:
            total = len(rows)
            male = sum(1 for r in rows if str(r.get('Sex','')).strip().upper() == 'M')
            female = sum(1 for r in rows if str(r.get('Sex','')).strip().upper() == 'F')
            ad = sum(1 for r in rows if str(r.get('label','')).strip() == '1')
            cn = sum(1 for r in rows if str(r.get('label','')).strip() == '0')
            ages = []
            for r in rows:
                try:
                    a = float(r.get('Age',''))
                    if a > 0:
                        ages.append(a)
                except Exception:
                    pass
            mean_age = sum(ages)/len(ages) if ages else 0.0
            std_age = (sum((x-mean_age)**2 for x in ages)/len(ages))**0.5 if len(ages)>1 else 0.0
            return {
                'total': total,
                'male': male,
                'female': female,
                'ad': ad,
                'cn': cn,
                'age_mean': mean_age,
                'age_std': std_age,
            }

        tr = summarize(train_rows)
        va = summarize(val_rows)
        te = summarize(test_rows)
        print("\nSplit demographics:")
        print(f"  Train: N={tr['total']}, M={tr['male']}, F={tr['female']}, AD={tr['ad']}, CN={tr['cn']}, Age={tr['age_mean']:.2f}±{tr['age_std']:.2f}")
        print(f"  Val:   N={va['total']}, M={va['male']}, F={va['female']}, AD={va['ad']}, CN={va['cn']}, Age={va['age_mean']:.2f}±{va['age_std']:.2f}")
        print(f"  Test:  N={te['total']}, M={te['male']}, F={te['female']}, AD={te['ad']}, CN={te['cn']}, Age={te['age_mean']:.2f}±{te['age_std']:.2f}")
        # -------------------------------------------------------------------
    else:
        print("\nNo data to save")

if __name__ == "__main__":
    main()
