"""
Pytest tests for protein classification data pipeline
Tests data loading, preprocessing, and pipeline integrity
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dataset import ProteinDataLoader
from model import get_classifiers
from utils import save_cv_fold_indices, evaluate_model_cv
from sklearn.model_selection import StratifiedKFold


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_protein_data():
    """Create mock protein data with realistic structure"""
    np.random.seed(42)
    n_samples = 50
    n_features = 10
    
    data = {
        'RID': list(range(1, n_samples + 1)),
        'VISCODE': ['bl'] * n_samples,
        'research_group': ['AD' if i % 2 == 0 else 'CN' for i in range(n_samples)],
        'subject_age': np.random.normal(75, 6, n_samples),
    }
    
    # Add protein features
    for i in range(n_features):
        data[f'PROTEIN_{i}'] = np.random.normal(20, 5, n_samples)
    
    return pd.DataFrame(data)


@pytest.fixture
def real_train_data():
    """Load real training data if available"""
    train_path = Path("src/data/protein/proteomic_encoder_train.csv")
    if train_path.exists():
        return pd.read_csv(train_path)
    return None


@pytest.fixture
def real_test_data():
    """Load real test data if available"""
    test_path = Path("src/data/protein/proteomic_encoder_test.csv")
    if test_path.exists():
        return pd.read_csv(test_path)
    return None


# ============================================================================
# Data Structure Tests
# ============================================================================

class TestDataStructure:
    """Test that data has correct structure and columns"""
    
    def test_mock_data_structure(self, mock_protein_data):
        """Test mock data has expected columns"""
        assert 'RID' in mock_protein_data.columns
        assert 'VISCODE' in mock_protein_data.columns
        assert 'research_group' in mock_protein_data.columns
        assert 'subject_age' in mock_protein_data.columns
        assert len(mock_protein_data) == 50
    
    def test_real_train_data_structure(self, real_train_data):
        """Test real training data has required columns"""
        if real_train_data is None:
            pytest.skip("Real training data not available")
        
        required_cols = ['RID', 'VISCODE', 'research_group', 'subject_age']
        for col in required_cols:
            assert col in real_train_data.columns, f"Missing column: {col}"
        
        # Should have protein features
        protein_cols = [c for c in real_train_data.columns if c not in required_cols]
        assert len(protein_cols) > 0, "No protein features found"
        print(f"Found {len(protein_cols)} protein features")
    
    def test_real_test_data_structure(self, real_test_data):
        """Test real test data has required columns"""
        if real_test_data is None:
            pytest.skip("Real test data not available")
        
        required_cols = ['RID', 'VISCODE', 'research_group', 'subject_age']
        for col in required_cols:
            assert col in real_test_data.columns, f"Missing column: {col}"
    
    def test_diagnosis_labels(self, real_train_data):
        """Test that diagnosis labels are valid"""
        if real_train_data is None:
            pytest.skip("Real training data not available")
        
        unique_labels = set(real_train_data['research_group'].unique())
        assert unique_labels.issubset({'AD', 'CN'}), f"Invalid labels: {unique_labels}"


# ============================================================================
# Data Loader Tests
# ============================================================================

class TestProteinDataLoader:
    """Test ProteinDataLoader preprocessing"""
    
    def test_loader_initialization(self):
        """Test loader can be initialized"""
        loader = ProteinDataLoader(
            data_path="dummy.csv",
            label_col="research_group",
            id_col="RID",
            random_state=42
        )
        assert loader.label_col == "research_group"
        assert loader.id_col == "RID"
        assert loader.random_state == 42
    
    def test_feature_extraction_excludes_metadata(self, mock_protein_data):
        """Test that metadata columns are excluded from features"""
        loader = ProteinDataLoader("dummy.csv")
        X, y = loader.prepare_features(mock_protein_data, fit=True)
        
        # X should only have protein features (not RID, VISCODE, research_group, subject_age)
        expected_n_features = 10  # From mock_protein_data fixture
        assert X.shape[1] == expected_n_features, f"Expected {expected_n_features} features, got {X.shape[1]}"
        assert X.shape[0] == len(mock_protein_data), "Number of samples mismatch"
    
    def test_label_encoding(self, mock_protein_data):
        """Test that labels are encoded correctly: AD=1, CN=0"""
        loader = ProteinDataLoader("dummy.csv")
        X, y = loader.prepare_features(mock_protein_data, fit=True)
        
        # Check encoding
        assert set(y) == {0, 1}, f"Expected labels {{0, 1}}, got {set(y)}"
        
        # Verify AD=1, CN=0
        for idx in range(len(mock_protein_data)):
            expected_label = 1 if mock_protein_data.iloc[idx]['research_group'] == 'AD' else 0
            assert y[idx] == expected_label, f"Label encoding mismatch at index {idx}"
    
    def test_feature_scaling(self, mock_protein_data):
        """Test that features are scaled (mean ~0, std ~1)"""
        loader = ProteinDataLoader("dummy.csv")
        X, y = loader.prepare_features(mock_protein_data, fit=True)
        
        # Check scaling (StandardScaler should give mean~0, std~1)
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        
        # Mean should be close to 0
        assert np.allclose(mean, 0, atol=1e-10), f"Features not centered: mean={mean}"
        # Std should be close to 1
        assert np.allclose(std, 1, atol=1e-10), f"Features not scaled: std={std}"
    
    def test_missing_value_handling(self):
        """Test that missing values are imputed with median"""
        data = pd.DataFrame({
            'RID': [1, 2, 3, 4, 5],
            'research_group': ['AD', 'CN', 'AD', 'CN', 'AD'],
            'feature1': [1.0, 2.0, np.nan, 4.0, 5.0],  # Missing value
            'feature2': [0.5, 1.5, 2.5, np.nan, 4.5],  # Missing value
        })
        
        loader = ProteinDataLoader("dummy.csv")
        X, y = loader.prepare_features(data, fit=True)
        
        # Should have no NaN values after preprocessing
        assert not np.isnan(X).any(), "NaN values remain after preprocessing"
    
    def test_zero_variance_removal(self):
        """Test that zero-variance features are removed"""
        data = pd.DataFrame({
            'RID': [1, 2, 3, 4, 5],
            'research_group': ['AD', 'CN', 'AD', 'CN', 'AD'],
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],  # Variance
            'feature2': [5.0, 5.0, 5.0, 5.0, 5.0],  # Zero variance
            'feature3': [0.5, 1.5, 2.5, 3.5, 4.5],  # Variance
        })
        
        loader = ProteinDataLoader("dummy.csv")
        X, y = loader.prepare_features(data, fit=True)
        
        # Should only have 2 features (feature2 removed)
        assert X.shape[1] == 2, f"Expected 2 features after zero-var removal, got {X.shape[1]}"
        assert len(loader.zero_var_cols) == 1, "Should have detected 1 zero-variance column"


# ============================================================================
# Train/Test Split Tests
# ============================================================================

class TestTrainTestSplit:
    """Test train/test split properties"""
    
    def test_no_data_leakage(self, real_train_data, real_test_data):
        """Test that train and test sets have no overlapping subjects"""
        if real_train_data is None or real_test_data is None:
            pytest.skip("Real data not available")
        
        train_rids = set(real_train_data['RID'])
        test_rids = set(real_test_data['RID'])
        
        overlap = train_rids.intersection(test_rids)
        assert len(overlap) == 0, f"Data leakage: {len(overlap)} subjects in both train and test"
    
    def test_diagnosis_balance_preserved(self, real_train_data, real_test_data):
        """Test that AD/CN ratio is similar in train and test"""
        if real_train_data is None or real_test_data is None:
            pytest.skip("Real data not available")
        
        train_ad_ratio = (real_train_data['research_group'] == 'AD').mean()
        test_ad_ratio = (real_test_data['research_group'] == 'AD').mean()
        
        # Ratios should be within 10% of each other
        diff = abs(train_ad_ratio - test_ad_ratio)
        assert diff < 0.10, f"AD ratio difference too large: train={train_ad_ratio:.2f}, test={test_ad_ratio:.2f}"
    
    def test_age_distribution_similar(self, real_train_data, real_test_data):
        """Test that age distribution is similar in train and test"""
        if real_train_data is None or real_test_data is None:
            pytest.skip("Real data not available")
        
        train_age_mean = real_train_data['subject_age'].mean()
        test_age_mean = real_test_data['subject_age'].mean()
        
        # Mean age should be within 3 years
        diff = abs(train_age_mean - test_age_mean)
        assert diff < 3.0, f"Age mean difference too large: {diff:.2f} years"
    
    def test_split_ratio(self, real_train_data, real_test_data):
        """Test that split ratio is approximately 85/15"""
        if real_train_data is None or real_test_data is None:
            pytest.skip("Real data not available")
        
        total = len(real_train_data) + len(real_test_data)
        test_ratio = len(real_test_data) / total
        
        # Should be close to 15%
        assert 0.10 < test_ratio < 0.20, f"Test ratio {test_ratio:.2%} not in expected range (10-20%)"


# ============================================================================
# Cross-Validation Tests
# ============================================================================

class TestCrossValidation:
    """Test cross-validation setup and fold properties"""
    
    def test_cv_fold_integrity(self, mock_protein_data):
        """Test that CV folds have no overlap and cover all samples"""
        loader = ProteinDataLoader("dummy.csv")
        X, y = loader.prepare_features(mock_protein_data, fit=True)
        
        cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        all_train_indices = []
        all_val_indices = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y)):
            # Check no overlap within fold
            overlap = set(train_idx).intersection(set(val_idx))
            assert len(overlap) == 0, f"Fold {fold_idx}: train/val overlap"
            
            all_train_indices.extend(train_idx)
            all_val_indices.extend(val_idx)
        
        # Each sample should appear in validation exactly once
        assert len(set(all_val_indices)) == len(X), "Not all samples in validation"
        assert len(all_val_indices) == len(X), "Duplicate validation samples"
    
    def test_cv_stratification(self, mock_protein_data):
        """Test that CV folds maintain class balance"""
        loader = ProteinDataLoader("dummy.csv")
        X, y = loader.prepare_features(mock_protein_data, fit=True)
        
        overall_ad_ratio = np.mean(y)
        cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y)):
            train_ad_ratio = np.mean(y[train_idx])
            val_ad_ratio = np.mean(y[val_idx])
            
            # Each fold should have similar class balance
            assert abs(train_ad_ratio - overall_ad_ratio) < 0.15, \
                f"Fold {fold_idx} train balance off: {train_ad_ratio:.2f} vs {overall_ad_ratio:.2f}"
            assert abs(val_ad_ratio - overall_ad_ratio) < 0.20, \
                f"Fold {fold_idx} val balance off: {val_ad_ratio:.2f} vs {overall_ad_ratio:.2f}"


# ============================================================================
# Model Input/Output Tests
# ============================================================================

class TestModelInputOutput:
    """Test that model inputs and outputs have correct shapes and types"""
    
    def test_classifier_input_shape(self, mock_protein_data):
        """Test that classifiers can accept the preprocessed data"""
        loader = ProteinDataLoader("dummy.csv")
        X, y = loader.prepare_features(mock_protein_data, fit=True)
        
        classifiers = get_classifiers(random_state=42)
        
        # Test with Logistic Regression (fast)
        clf = classifiers['Logistic Regression']
        clf.fit(X, y)
        
        # Check predictions
        y_pred = clf.predict(X)
        assert len(y_pred) == len(y), "Prediction length mismatch"
        assert set(y_pred).issubset({0, 1}), f"Invalid predictions: {set(y_pred)}"
    
    def test_predict_proba_output(self, mock_protein_data):
        """Test that predict_proba returns valid probabilities"""
        loader = ProteinDataLoader("dummy.csv")
        X, y = loader.prepare_features(mock_protein_data, fit=True)
        
        classifiers = get_classifiers(random_state=42)
        clf = classifiers['Logistic Regression']
        clf.fit(X, y)
        
        y_proba = clf.predict_proba(X)
        
        # Check shape
        assert y_proba.shape == (len(X), 2), f"Probability shape mismatch: {y_proba.shape}"
        
        # Check probabilities sum to 1
        sums = y_proba.sum(axis=1)
        assert np.allclose(sums, 1.0), "Probabilities don't sum to 1"
        
        # Check probabilities in [0, 1]
        assert np.all(y_proba >= 0) and np.all(y_proba <= 1), "Probabilities out of range"


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """End-to-end integration tests"""
    
    def test_full_pipeline_mock_data(self, mock_protein_data, tmp_path):
        """Test full pipeline with mock data"""
        # Save mock data to temp file
        train_path = tmp_path / "train.csv"
        mock_protein_data.to_csv(train_path, index=False)
        
        # Load and preprocess
        loader = ProteinDataLoader(
            data_path=str(train_path),
            label_col="research_group",
            id_col="RID"
        )
        X, y, _, _, df = loader.get_train_test_split(str(train_path), test_path=None)
        
        # Setup CV
        cv_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        # Evaluate a single classifier
        clf = get_classifiers(random_state=42)['Logistic Regression']
        result = evaluate_model_cv(
            clf=clf,
            X_train=X,
            y_train=y,
            X_test=None,
            y_test=None,
            cv_splitter=cv_splitter,
            clf_name="Logistic Regression"
        )
        
        # Check result structure
        assert 'cv_auc_mean' in result
        assert 'cv_acc_mean' in result
        assert 0 <= result['cv_auc_mean'] <= 1, f"Invalid AUC: {result['cv_auc_mean']}"
        assert 0 <= result['cv_acc_mean'] <= 1, f"Invalid accuracy: {result['cv_acc_mean']}"
    
    def test_real_data_pipeline(self, real_train_data, real_test_data, tmp_path):
        """Test full pipeline with real data"""
        if real_train_data is None:
            pytest.skip("Real data not available")
        
        # Use real data paths
        train_path = "src/data/protein/proteomic_encoder_train.csv"
        test_path = "src/data/protein/proteomic_encoder_test.csv" if real_test_data is not None else None
        
        # Load and preprocess
        loader = ProteinDataLoader(
            data_path=train_path,
            label_col="research_group",
            id_col="RID"
        )
        X_train, y_train, X_test, y_test, train_df = loader.get_train_test_split(train_path, test_path)
        
        # Validate shapes
        assert X_train.shape[0] == len(real_train_data), "Train sample count mismatch"
        assert X_train.shape[1] > 0, "No features extracted"
        
        if X_test is not None:
            assert X_test.shape[0] == len(real_test_data), "Test sample count mismatch"
            assert X_test.shape[1] == X_train.shape[1], "Feature count mismatch between train and test"
        
        print(f"\nPipeline validation:")
        print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        if X_test is not None:
            print(f"  Test: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        print(f"  AD/CN ratio (train): {np.mean(y_train):.2%}")


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "-s"])
