import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "mri" / "BrainIAC" / "src"))
from load_brainiac import load_brainiac  # type: ignore

import torch
import nibabel as nib
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    Resized, NormalizeIntensityd, EnsureTyped
)


warnings.filterwarnings("ignore", message="X does not have valid feature names")


METADATA_COLS = [
    'RID', 'Subject', 'VISCODE', 'Visit', 'research_group', 'Group', 'Sex', 'Age',
    'subject_age', 'Image Data ID', 'Description', 'Type', 'Modality', 'Format',
    'Acq Date', 'Downloaded', 'MRI_acquired', 'mri_source_path', 'mri_path'
]


def build_mri_transform():
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(96, 96, 96), mode="trilinear"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image"]) 
    ])


def extract_mri_latent(mri_model, mri_path: Path, device: str):
    transform = build_mri_transform()
    sample = {"image": str(mri_path)}
    sample = transform(sample)
    image = sample["image"].unsqueeze(0).to(device)  # [1, 1, 96, 96, 96]
    with torch.no_grad():
        feat = mri_model(image)  # [1, 768]
        return feat.cpu().numpy().reshape(-1)


def auc_standardized(y_true, y_pred, y_proba):
    # Standardized AUC handling
    n_unique_labels = len(np.unique(y_true))
    n_unique_preds = len(np.unique(y_pred))
    if n_unique_labels < 2:
        return float('nan')
    if n_unique_preds < 2:
        return 0.5
    try:
        return float(roc_auc_score(y_true, y_proba[:, 1]))
    except ValueError:
        return float('nan')


def evaluate_fold(train_idx, test_idx, df, mri_model, device):
    # Prepare labels
    y = (df['research_group'] == 'AD').astype(int).values
    
    # Extract latents (cache per index)
    latents = {}
    for idx in np.concatenate([train_idx, test_idx]):
        if idx not in latents:
            mri_path = Path(df.iloc[idx]['mri_path'])
            latents[idx] = extract_mri_latent(mri_model, mri_path, device)
    
    X_train = np.stack([latents[i] for i in train_idx])
    y_train = y[train_idx]
    X_test = np.stack([latents[i] for i in test_idx])
    y_test = y[test_idx]
    
    # Classifier
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    if hasattr(clf, 'predict_proba'):
        y_proba = clf.predict_proba(X_test)
    else:
        # Fallback probabilities if not available
        y_proba = np.column_stack([1 - y_pred, y_pred]).astype(float)
    
    # Metrics
    acc = float(accuracy_score(y_test, y_pred))
    bacc = float(balanced_accuracy_score(y_test, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()
    auc = auc_standardized(y_test, y_pred, y_proba)
    
    return {
        'accuracy': acc,
        'balanced_accuracy': bacc,
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'auc': auc,
        'confusion_matrix': cm
    }


def main():
    parser = argparse.ArgumentParser(description="MRI-only baseline using BrainIAC latents with CV splits")
    parser.add_argument('--data-csv', required=True, type=str)
    parser.add_argument('--cv-splits-json', required=True, type=str)
    parser.add_argument('--brainiac-ckpt', required=True, type=str)
    parser.add_argument('--save-dir', required=True, type=str)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()

    df = pd.read_csv(args.data_csv)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.cv_splits_json, 'r') as f:
        cv_splits = json.load(f)
    n_folds = len(cv_splits)
    
    # Load BrainIAC model
    mri_model = load_brainiac(args.brainiac_ckpt, args.device)
    mri_model.eval()

    per_fold = []
    for fold_idx, split in enumerate(cv_splits, start=1):
        train_idx = np.array(split['train_idx'], dtype=int)
        test_idx = np.array(split['test_idx'], dtype=int)
        res = evaluate_fold(train_idx, test_idx, df, mri_model, args.device)
        res['fold'] = fold_idx
        per_fold.append(res)
        # Save per-fold
        with open(save_dir / f"fold_{fold_idx}_results.json", 'w') as f:
            json.dump(res, f, indent=2)

    # Aggregate
    def agg(key):
        vals = [r[key] for r in per_fold if r[key] == r[key]]  # exclude NaNs
        return float(np.mean(vals)) if len(vals) > 0 else float('nan')
    aggregated = {
        'n_folds': n_folds,
        'accuracy_mean': agg('accuracy'),
        'balanced_accuracy_mean': agg('balanced_accuracy'),
        'precision_mean': agg('precision'),
        'recall_mean': agg('recall'),
        'f1_mean': agg('f1'),
        'auc_mean': agg('auc'),
        'per_fold': per_fold
    }
    with open(save_dir / 'aggregated_results.json', 'w') as f:
        json.dump(aggregated, f, indent=2)
    print("Saved MRI-only baseline results to:", save_dir)


if __name__ == '__main__':
    main()





