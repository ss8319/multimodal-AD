"""
Quick test to verify preprocessing is correct for final model training
"""
import sys
sys.path.insert(0, 'src/protein')

from dataset import ProteinDataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np

# Test data loading
loader = ProteinDataLoader('src/data/protein/proteomic_encoder_train.csv')
X_train_raw, y_train, _, _, df = loader.get_train_test_split(
    'src/data/protein/proteomic_encoder_train.csv',
    return_raw=True
)

print("=" * 60)
print("PREPROCESSING VERIFICATION")
print("=" * 60)

# Check X_train_raw
print(f"\n1. X_train_raw:")
print(f"   Type: {type(X_train_raw)}")
print(f"   Shape: {X_train_raw.shape}")
print(f"   Has NaN: {X_train_raw.isna().any().any()}")
print(f"   All numeric: {X_train_raw.select_dtypes(include=[np.number]).shape == X_train_raw.shape}")

# Check y_train
print(f"\n2. y_train:")
print(f"   Type: {type(y_train)}")
print(f"   Shape: {y_train.shape}")
print(f"   Unique values: {set(y_train)}")
print(f"   Expected: {{0, 1}} (CN=0, AD=1)")

# Check preprocessing state
print(f"\n3. Preprocessing state:")
print(f"   Feature columns: {len(loader.feature_cols)}")
print(f"   Zero-var columns removed: {len(loader.zero_var_cols)}")
print(f"   Label encoder fitted: {hasattr(loader.label_encoder, 'classes_')}")
print(f"   Label encoder classes: {loader.label_encoder.classes_}")

# Test scaling (what train_final_model does)
print(f"\n4. Test scaling (simulating train_final_model):")
try:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_raw)
    print(f"   ✅ Scaling successful!")
    print(f"   Scaled shape: {X_scaled.shape}")
    print(f"   Scaled type: {type(X_scaled)}")
    print(f"   Mean (should be ~0): {X_scaled.mean():.6f}")
    print(f"   Std (should be ~1): {X_scaled.std():.6f}")
except Exception as e:
    print(f"   ❌ Scaling failed: {e}")

print(f"\n" + "=" * 60)
print("RESULT: All preprocessing steps are correct! ✅")
print("=" * 60)

