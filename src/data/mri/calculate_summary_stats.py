#!/usr/bin/env python3
"""
Reads MCI CSV splits, combines them, and prints:
- Count of Male vs Female
- Mean age
- Standard deviation of age

Input CSVs (columns: pat_id,label,Sex,Age):
- /home/ssim0068/data/ADNI_v2/csvs/mci_test.csv
- /home/ssim0068/data/ADNI_v2/csvs/mci_train.csv
- /home/ssim0068/data/ADNI_v2/csvs/mci_val.csv
"""

import csv
import math
from statistics import mean, pstdev


def read_rows(path):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def to_float(value):
    try:
        return float(value)
    except Exception:
        return math.nan


def main():
    paths = [
        "/home/ssim0068/data/ADNI_v2/csvs/mci_test.csv",
        "/home/ssim0068/data/ADNI_v2/csvs/mci_train.csv",
        "/home/ssim0068/data/ADNI_v2/csvs/mci_val.csv",
    ]

    # Read and combine
    combined = []
    for p in paths:
        combined.extend(read_rows(p))

    # Compute male/female counts (case-insensitive on Sex)
    male = sum(1 for r in combined if str(r.get("Sex", "")).strip().upper() == "M")
    female = sum(1 for r in combined if str(r.get("Sex", "")).strip().upper() == "F")
    unknown = len(combined) - male - female

    # Compute AD (1) vs CN (0) counts
    ad_count = sum(1 for r in combined if str(r.get("label", "")).strip() == "1")
    cn_count = sum(1 for r in combined if str(r.get("label", "")).strip() == "0")
    unknown_labels = len(combined) - ad_count - cn_count

    # Compute age stats (ignore non-numeric)
    ages = [to_float(r.get("Age", "")) for r in combined]
    ages = [a for a in ages if not math.isnan(a)]

    age_mean = mean(ages) if ages else float("nan")
    # Use population std dev to match full dataset behavior; change to stdev if sample desired
    age_std = pstdev(ages) if len(ages) > 1 else 0.0

    # Report
    print("MCI combined summary")
    print(f"Total rows: {len(combined)}")
    print(f"Male: {male}")
    print(f"Female: {female}")
    if unknown:
        print(f"Unknown/Other: {unknown}")
    print(f"AD (label=1): {ad_count}")
    print(f"CN (label=0): {cn_count}")
    if unknown_labels:
        print(f"Unknown labels: {unknown_labels}")
    print(f"Age mean: {age_mean:.2f}" if not math.isnan(age_mean) else "Age mean: NaN")
    print(f"Age std (population): {age_std:.2f}")


if __name__ == "__main__":
    main()


