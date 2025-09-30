"""
Quick test script to verify the refactored code works
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from dataset import ProteinDataset, ProteinDataLoader
        print("[OK] dataset.py imports work")
    except Exception as e:
        print(f"[FAIL] dataset.py import failed: {e}")
        return False
    
    try:
        from model import ProteinTransformer, ProteinTransformerClassifier, get_classifiers
        print("[OK] model.py imports work")
    except Exception as e:
        print(f"[FAIL] model.py import failed: {e}")
        return False
    
    try:
        from utils import save_cv_fold_indices, evaluate_model_cv, print_results_summary, save_results
        print("[OK] utils.py imports work")
    except Exception as e:
        print(f"[FAIL] utils.py import failed: {e}")
        return False
    
    return True


def test_data_loader():
    """Test ProteinDataLoader basic functionality"""
    print("\nTesting ProteinDataLoader...")
    
    try:
        from dataset import ProteinDataLoader
        import pandas as pd
        import numpy as np
        
        # Create mock data
        mock_df = pd.DataFrame({
            'RID': [1, 2, 3, 4, 5],
            'research_group': ['AD', 'CN', 'AD', 'CN', 'AD'],
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [0.5, 1.5, 2.5, 3.5, 4.5],
        })
        
        loader = ProteinDataLoader("mock.csv")
        X, y = loader.prepare_features(mock_df, fit=True)
        
        assert X.shape == (5, 2), f"Expected shape (5, 2), got {X.shape}"
        assert len(y) == 5, f"Expected 5 labels, got {len(y)}"
        assert set(y) == {0, 1}, f"Expected labels {{0, 1}}, got {set(y)}"
        
        print("[OK] ProteinDataLoader works correctly")
        return True
    except Exception as e:
        print(f"[FAIL] ProteinDataLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_get_classifiers():
    """Test get_classifiers function"""
    print("\nTesting get_classifiers...")
    
    try:
        from model import get_classifiers
        
        classifiers = get_classifiers(random_state=42)
        
        expected_models = [
            'Logistic Regression',
            'Random Forest',
            'SVM (RBF)',
            'Gradient Boosting',
            'Neural Network',
            'Protein Transformer'
        ]
        
        for model_name in expected_models:
            assert model_name in classifiers, f"Missing model: {model_name}"
        
        print(f"[OK] get_classifiers returns {len(classifiers)} models")
        return True
    except Exception as e:
        print(f"[FAIL] get_classifiers test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("TESTING REFACTORED PROTEIN MODULE")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_data_loader,
        test_get_classifiers,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    
    if all(results):
        print("[PASS] ALL TESTS PASSED")
    else:
        print("[FAIL] SOME TESTS FAILED")
    
    print("=" * 60)
    
    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
