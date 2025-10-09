import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METADATA_COLS = [
    'RID', 'Subject', 'VISCODE', 'Visit', 'research_group', 'Group', 'Sex', 'Age',
    'subject_age', 'Image Data ID', 'Description', 'Type', 'Modality', 'Format',
    'Acq Date', 'Downloaded', 'MRI_acquired', 'mri_source_path', 'mri_path'
]


def summarize_stats(df: pd.DataFrame):
    means = df.mean(numeric_only=True)
    stds = df.std(numeric_only=True)
    return means, stds


def main():
    parser = argparse.ArgumentParser(description="Protein distribution shift analysis")
    parser.add_argument('--train-csv', required=True, type=str, help='Protein encoder training CSV (115)')
    parser.add_argument('--paired-csv', required=True, type=str, help='Paired multimodal CSV (38)')
    parser.add_argument('--cv-splits-json', required=True, type=str, help='CV splits used for paired data')
    parser.add_argument('--save-path', required=True, type=str)
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_csv)
    paired_df = pd.read_csv(args.paired_csv)
    with open(args.cv_splits_json, 'r') as f:
        cv_splits = json.load(f)

    protein_cols = [c for c in paired_df.columns if c not in METADATA_COLS]

    # Overall shift
    train_means, train_stds = summarize_stats(train_df[protein_cols])
    paired_means, paired_stds = summarize_stats(paired_df[protein_cols])

    mean_diff = (paired_means - train_means).to_dict()
    std_ratio = (paired_stds / (train_stds.replace(0, np.nan))).replace([np.inf, -np.inf], np.nan).to_dict()

    # Per-fold shift (paired only)
    folds = []
    for fold_idx, split in enumerate(cv_splits, start=1):
        idx = np.array(split['train'] + split['test'], dtype=int)
        fold_df = paired_df.iloc[idx]
        f_means, f_stds = summarize_stats(fold_df[protein_cols])
        folds.append({
            'fold': fold_idx,
            'means': f_means.to_dict(),
            'stds': f_stds.to_dict(),
        })

    report = {
        'n_train': int(len(train_df)),
        'n_paired': int(len(paired_df)),
        'overall': {
            'mean_diff': mean_diff,
            'std_ratio': std_ratio,
        },
        'per_fold': folds
    }

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    print("Saved protein shift analysis to:", save_path)


if __name__ == '__main__':
    main()


