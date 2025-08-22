import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import os
from pathlib import Path

# Load and filter data
print("🔄 Loading and filtering MPRAGE data...")
data_path = r"D:\ADNI\AD_CN\proteomics\Biomarkers Consortium Plasma Proteomics MRM\MRI\Baseline_AD_CN_8_21_2025.csv"
df = pd.read_csv(data_path)

# Filter for MPRAGE sequences and remove duplicates
allowed_mprage = ['MPRAGE', 'MP-RAGE', 'ADNI       MPRAGEadni2', 'ADNI       MPRAGE', 'ADNI_new   MPRAGE', 'ADNI       MPRAGEadni2B']
df = df[df['Description'].isin(allowed_mprage)].drop_duplicates('Subject').copy()

print(f"✅ Filtered to {len(df)} unique subjects")
print(f"📊 Class distribution: {df['Group'].value_counts().to_dict()}")

# Create splits
print("\n🔀 Creating train/test splits with cross-validation...")
X, y = df.drop('Group', axis=1), df['Group']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create CV folds
cv_folds = 5
skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
cv_fold_indices = [{'fold': i+1, 'train_indices': train_idx.tolist(), 'val_indices': val_idx.tolist()} 
                   for i, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train))]

print(f"✅ Created {len(X_train)} train, {len(X_test)} test samples with {cv_folds}-fold CV")

# Add DICOM paths
def find_dicom_path(subject_id, description):
    """Find DICOM folder path for subject and description"""
    adni_base = Path(r"D:\ADNI\AD_CN\proteomics\Biomarkers Consortium Plasma Proteomics MRM\MRI\ADNI")
    subject_folder = adni_base / subject_id
    
    if not subject_folder.exists():
        return None
    
    # Mapping descriptions to folder patterns
    desc_to_pattern = {
        'MPRAGE': 'MPRAGE', 'MP-RAGE': 'MP-RAGE',
        'ADNI       MPRAGEadni2': 'ADNI_______MPRAGEadni2',
        'ADNI       MPRAGE': 'ADNI_______MPRAGE', 
        'ADNI_new   MPRAGE': 'ADNI_new___MPRAGE',
        'ADNI       MPRAGEadni2B': 'ADNI_______MPRAGEadni2B'
    }
    
    pattern = desc_to_pattern.get(description, 'MPRAGE')
    
    # Find matching folder with DICOM files
    for folder in subject_folder.iterdir():
        if folder.is_dir() and pattern in folder.name:
            for root, _, files in os.walk(folder):
                if any(f.lower().endswith('.dcm') for f in files):
                    return os.path.relpath(root, adni_base)
    return None

def add_dicom_paths(df_split, split_name):
    """Add DICOM paths to dataframe"""
    paths = [find_dicom_path(row['Subject'], row['Description']) for _, row in df_split.iterrows()]
    df_split['dicom_folder_path'] = paths
    found = sum(1 for p in paths if p)
    print(f"   • {split_name}: {found}/{len(df_split)} subjects have DICOM paths ({found/len(df_split)*100:.1f}%)")
    return df_split

print("\n🔍 Adding DICOM folder paths...")
train_df = X_train.copy()
train_df['Group'] = y_train
test_df = X_test.copy() 
test_df['Group'] = y_test

train_df = add_dicom_paths(train_df, "Training")
test_df = add_dicom_paths(test_df, "Test")

# Save results
output_folder = Path(r"D:\ADNI\AD_CN\proteomics\Biomarkers Consortium Plasma Proteomics MRM\MRI\splits")
output_folder.mkdir(exist_ok=True)

train_df.to_csv(output_folder / "train_split.csv", index=False)
test_df.to_csv(output_folder / "test_split.csv", index=False)
pd.DataFrame(cv_fold_indices).to_csv(output_folder / "cv_fold_indices.csv", index=False)

print(f"\n💾 All files saved to: {output_folder}")
print(f"🎉 Setup complete! Ready for MRI classification training.")

# Show sample DICOM paths
valid_paths = train_df[train_df['dicom_folder_path'].notna()]
if len(valid_paths) > 0:
    print(f"\n📋 Sample DICOM paths:")
    for _, row in valid_paths.head(3).iterrows():
        print(f"   • {row['Subject']}: {row['dicom_folder_path']}")
