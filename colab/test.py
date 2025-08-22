import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import os
from pathlib import Path

AD_CN_baseline_add_training_data = r"D:\ADNI\AD_CN\proteomics\Biomarkers Consortium Plasma Proteomics MRM\MRI\Baseline_AD_CN_8_21_2025.csv"
train_df= pd.read_csv(AD_CN_baseline_add_training_data)
# print(train_df.head())

print('No of unique Subject in train_df: ', train_df['Subject'].nunique())

print(f"\n📋 Original data shape: {train_df.shape}")
print(f"📋 Unique Description values:")
print(train_df['Description'].value_counts())

# Filter for specific MPRAGE sequences only (strict filtering)
print(f"\n🔍 Filtering for specific MPRAGE sequences...")
allowed_mprage = [
    'MPRAGE',
    'MP-RAGE',
    'ADNI       MPRAGEadni2',
    'ADNI       MPRAGE',
    'ADNI_new   MPRAGE',
    'ADNI       MPRAGEadni2B'
]

# Create mask for exact matches
mprage_mask = train_df['Description'].isin(allowed_mprage)
train_df_mprage = train_df[mprage_mask].copy()
# there are duplicates in the train_df_mprage, drop them
print(f"✅ Dropping duplicates in train_df_mprage...")
train_df_mprage = train_df_mprage.drop_duplicates('Subject')

print(f"✅ Allowed MPRAGE sequences:")
for seq in allowed_mprage:
    count = len(train_df[train_df['Description'] == seq])
    print(f"   • '{seq}': {count} scans")

print(f"\n✅ Filtered data shape: {train_df_mprage.shape}")
print(f"📊 Remaining Description values:")
print(train_df_mprage['Description'].value_counts())

print(f"\n📈 Data reduction summary:")
print(f"   • Original rows: {len(train_df)}")
print(f"   • MPRAGE rows: {len(train_df_mprage)}")
print(f"   • Rows removed: {len(train_df) - len(train_df_mprage)}")

print(f"\n👥 Subject count after filtering:")
print(f"   • Unique subjects: {train_df_mprage['Subject'].nunique()}")
print(f"   • Class distribution:")
print(train_df_mprage['Group'].value_counts())

train_df_mprage.to_csv(r"D:\ADNI\AD_CN\proteomics\Biomarkers Consortium Plasma Proteomics MRM\MRI\Baseline_AD_CN_8_21_2025_MPRAGE.csv", index=False)

# Update train_df to be the filtered version
train_df = train_df_mprage
print(f"\n🔀 Creating train/test splits with cross-validation...")

# Create train/test split (80/20)
X = train_df_mprage.drop(['Group'], axis=1)  # Features (all columns except target)
y = train_df_mprage['Group']  # Target variable

# Stratified split to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"✅ Train/test split created:")
print(f"   • Training set: {len(X_train)} samples")
print(f"   • Test set: {len(X_test)} samples")
print(f"   • Training class distribution: {y_train.value_counts().to_dict()}")
print(f"   • Test class distribution: {y_test.value_counts().to_dict()}")

# Create cross-validation folds
cv_folds = 5
skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

# Create CV fold indices
cv_fold_indices = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    cv_fold_indices.append({
        'fold': fold,
        'train_indices': train_idx.tolist(),
        'val_indices': val_idx.tolist()
    })

print(f"✅ Cross-validation folds created:")
print(f"   • Number of folds: {cv_folds}")
print(f"   • Each fold maintains class distribution")

# Save the splits
output_folder = r"D:\ADNI\AD_CN\proteomics\Biomarkers Consortium Plasma Proteomics MRM\MRI\splits"
#make the folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Save train/test splits
train_df = X_train.copy()
train_df['Group'] = y_train
test_df = X_test.copy()
test_df['Group'] = y_test

train_df.to_csv(f"{output_folder}/train_split.csv", index=False)
test_df.to_csv(f"{output_folder}/test_split.csv", index=False)

# Save CV fold indices
cv_df = pd.DataFrame(cv_fold_indices)
cv_df.to_csv(f"{output_folder}/cv_fold_indices.csv", index=False)

print(f"\n💾 Data splits saved to:")
print(f"   • {output_folder}/train_split.csv")
print(f"   • {output_folder}/test_split.csv")
print(f"   • {output_folder}/cv_fold_indices.csv")

print(f"\n�� Ready for training with:")
print(f"   • Training set: {len(train_df)} samples")
print(f"   • Test set: {len(test_df)} samples")
print(f"   • {cv_folds}-fold cross-validation")
print(f"   • Target classes: {y.unique()}")

# Add DICOM folder paths to train and test splits
def find_dicom_folder_path(subject_id, description, adni_base_path):
    """
    Find the DICOM folder path for a given subject and description
    """
    subject_folder = Path(adni_base_path) / subject_id
    
    if not subject_folder.exists():
        return None
    
    # Map description to folder name patterns
    description_to_folder = {
        'MPRAGE': ['MPRAGE'],
        'MP-RAGE': ['MP-RAGE'],
        'ADNI       MPRAGEadni2': ['ADNI_______MPRAGEadni2'],
        'ADNI       MPRAGE': ['ADNI_______MPRAGE'],
        'ADNI_new   MPRAGE': ['ADNI_new___MPRAGE'],
        'ADNI       MPRAGEadni2B': ['ADNI_______MPRAGEadni2B']
    }
    # Get possible folder patterns for this description
    folder_patterns = description_to_folder.get(description, ['MPRAGE', 'MP-RAGE'])
    
    # Look for matching folders
    for folder in subject_folder.iterdir():
        if folder.is_dir():
            folder_name = folder.name
            # Check if folder name contains any of the patterns
            for pattern in folder_patterns:
                if pattern in folder_name:
                    # Look for the deepest folder with DICOM files
                    for root, dirs, files in os.walk(folder):
                        dcm_files = [f for f in files if f.lower().endswith('.dcm')]
                        if dcm_files:
                            # Return relative path from ADNI base
                            relative_path = os.path.relpath(root, adni_base_path)
                            return relative_path
    
    return None

def add_dicom_paths_to_splits(train_df, test_df, output_folder):
    """
    Add DICOM folder paths to train and test dataframes
    """
    print(f"\n🔍 Adding DICOM folder paths to splits...")
    
    # Base path for ADNI folder
    adni_base_path = r"D:\ADNI\AD_CN\proteomics\Biomarkers Consortium Plasma Proteomics MRM\MRI\ADNI"
    
    # Add paths to training set
    train_paths = []
    train_found = 0
    for idx, row in train_df.iterrows():
        dicom_path = find_dicom_folder_path(row['Subject'], row['Description'], adni_base_path)
        train_paths.append(dicom_path)
        if dicom_path:
            train_found += 1
    
    train_df['dicom_folder_path'] = train_paths
    
    # Add paths to test set
    test_paths = []
    test_found = 0
    for idx, row in test_df.iterrows():
        dicom_path = find_dicom_folder_path(row['Subject'], row['Description'], adni_base_path)
        test_paths.append(dicom_path)
        if dicom_path:
            test_found += 1
    
    test_df['dicom_folder_path'] = test_paths
    
    # Save updated splits
    train_df.to_csv(f"{output_folder}/train_split.csv", index=False)
    test_df.to_csv(f"{output_folder}/test_split.csv", index=False)
    
    print(f"✅ DICOM paths added:")
    print(f"   • Training set: {train_found}/{len(train_df)} subjects have DICOM paths ({(train_found/len(train_df)*100):.1f}%)")
    print(f"   • Test set: {test_found}/{len(test_df)} subjects have DICOM paths ({(test_found/len(test_df)*100):.1f}%)")
    
    # Show subjects without DICOM paths
    train_missing = train_df[train_df['dicom_folder_path'].isna()]
    test_missing = test_df[test_df['dicom_folder_path'].isna()]
    
    if len(train_missing) > 0:
        print(f"\n⚠️ Training subjects without DICOM paths ({len(train_missing)}):")
        for _, row in train_missing.head(5).iterrows():
            print(f"   • {row['Subject']} (Description: '{row['Description']}')")
        if len(train_missing) > 5:
            print(f"   • ... and {len(train_missing) - 5} more")
    
    if len(test_missing) > 0:
        print(f"\n⚠️ Test subjects without DICOM paths ({len(test_missing)}):")
        for _, row in test_missing.head(5).iterrows():
            print(f"   • {row['Subject']} (Description: '{row['Description']}')")
        if len(test_missing) > 5:
            print(f"   • ... and {len(test_missing) - 5} more")
    
    # Show sample paths
    print(f"\n📋 Sample DICOM paths:")
    valid_train = train_df[train_df['dicom_folder_path'].notna()]
    for _, row in valid_train.head(3).iterrows():
        print(f"   • {row['Subject']}: {row['dicom_folder_path']}")
    
    return train_df, test_df

# Execute the DICOM path addition
if os.path.exists(r"D:\ADNI\AD_CN\proteomics\Biomarkers Consortium Plasma Proteomics MRM\MRI\ADNI"):
    train_df, test_df = add_dicom_paths_to_splits(train_df, test_df, output_folder)
    print(f"\n💾 Updated splits saved with DICOM paths!")
else:
    print(f"\n❌ ADNI folder not found - DICOM paths not added")

print(f"\n📁 All files saved in: {output_folder}")
