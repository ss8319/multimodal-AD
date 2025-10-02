"""
Simple inference script for saved Logistic Regression model
Reuses dataset.py and utils.py to ensure exact same preprocessing pipeline
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# Import our own modules to reuse preprocessing logic
from dataset import ProteinDataLoader
from utils import compute_confusion_metrics

def load_model_and_test(model_path, train_csv_path, test_csv_path):
    """
    Load saved model and test on test set with exact same preprocessing as training
    
    Args:
        model_path: Path to saved .pkl model file
        train_csv_path: Path to training CSV (to fit scaler)
        test_csv_path: Path to test CSV file
    """    
    # 1. Use ProteinDataLoader to replicate exact preprocessing from main.py
    print(f"\n1. Loading and preprocessing data using ProteinDataLoader...")
    data_loader = ProteinDataLoader(
        data_path=train_csv_path,
        label_col='research_group',
        id_col='RID',
        random_state=42
    )
    
    # Load train and test data with exact same preprocessing as main.py
    X_train_raw, y_train, X_test_raw, y_test, train_df = data_loader.get_train_test_split(
        train_path=train_csv_path,
        test_path=test_csv_path
    )
    
    print(f"   Training samples: {len(X_train_raw)}")
    print(f"   Test samples: {len(X_test_raw)}")
    print(f"   Features: {X_train_raw.shape[1]}")
    
    # 2. Load saved model
    print(f"\n2. Loading saved model from: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"   Model type: {type(model).__name__}")
    print(f"   Model classes: {model.classes_}")
    
    # 3. Replicate exact scaling from save_all_models() in utils.py
    print(f"\n3. Scaling features (same as in save_all_models)...")
    scaler = StandardScaler()
    
    # Fit scaler on TRAINING data (same as in save_all_models)
    X_train_scaled = scaler.fit_transform(X_train_raw)
    print(f"   Fitted scaler on training data: {X_train_scaled.shape}")
    
    # Transform test data using the training-fitted scaler
    X_test_scaled = scaler.transform(X_test_raw)
    print(f"   Transformed test data: {X_test_scaled.shape}")
    
    # 4. Make predictions (convert to numpy like in utils.py)
    print(f"\n4. Making predictions...")
    test_pred = model.predict(np.asarray(X_test_scaled))
    test_acc = accuracy_score(y_test, test_pred)
    print(f"   Test Accuracy: {test_acc:.3f}")
    
    # 5. Get probabilities and AUC (same as utils.py)
    if hasattr(model, 'predict_proba'):
        test_proba = model.predict_proba(np.asarray(X_test_scaled))
        if test_proba is not None and not np.isnan(test_proba).any():
            test_auc = roc_auc_score(y_test, test_proba[:, 1])
            print(f"   Test AUC: {test_auc:.3f}")
            
            # Show probability distribution
            print(f"   Probability range: [{test_proba.min():.3f}, {test_proba.max():.3f}]")
            print(f"   Mean probability for class 1: {test_proba[:, 1].mean():.3f}")
        else:
            test_auc = None
            print(f"   Test AUC: Invalid probabilities")
    else:
        test_auc = None
        print(f"   Model does not support probability prediction")
    
    # 6. Show prediction breakdown
    print(f"\n5. Prediction breakdown:")
    pred_counts = pd.Series(test_pred).value_counts()
    for class_val, count in pred_counts.items():
        class_name = "AD" if class_val == 1 else "CN"
        print(f"   Predicted {class_name}: {count} ({count/len(test_pred)*100:.1f}%)")
    
    # 7. Use compute_confusion_metrics from utils.py
    print(f"\n6. Confusion Matrix (using utils.compute_confusion_metrics):")
    confusion_metrics = compute_confusion_metrics(y_test, test_pred)
    print(f"   True Negatives (CN): {confusion_metrics['tn']}")
    print(f"   False Positives: {confusion_metrics['fp']}")
    print(f"   False Negatives: {confusion_metrics['fn']}")
    print(f"   True Positives (AD): {confusion_metrics['tp']}")
    print(f"   Sensitivity (AD recall): {confusion_metrics['sensitivity']:.3f}")
    print(f"   Specificity (CN recall): {confusion_metrics['specificity']:.3f}")
    print(f"   PPV (Precision): {confusion_metrics['ppv']:.3f}")
    print(f"   NPV: {confusion_metrics['npv']:.3f}")
    
    print(f"\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    
    return {
        'test_accuracy': test_acc,
        'test_auc': test_auc,
        'predictions': test_pred,
        'probabilities': test_proba if hasattr(model, 'predict_proba') else None,
        'true_labels': y_test,
        'confusion_metrics': confusion_metrics
    }

if __name__ == "__main__":
    # Paths
    model_path = Path("src/protein/runs/run_20251002_131810/models/logistic_regression.pkl")
    test_csv_path = Path("src/data/protein/proteomic_encoder_test.csv")
    train_csv_path = Path("src/data/protein/proteomic_encoder_train.csv")
    
    # Check if files exist
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        print("Available runs:")
        runs_dir = Path("src/protein/runs")
        if runs_dir.exists():
            for run_dir in runs_dir.iterdir():
                if run_dir.is_dir():
                    print(f"  {run_dir}")
        exit(1)
    
    if not test_csv_path.exists():
        print(f"ERROR: Test CSV not found: {test_csv_path}")
        exit(1)
    
    # Run inference
    results = load_model_and_test(model_path, train_csv_path, test_csv_path)
