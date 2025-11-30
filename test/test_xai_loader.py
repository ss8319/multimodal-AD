"""
Unit tests for XAI model loader and data utilities

Tests the refactored XAI module components:
- Model loader with registry pattern
- Data preparation utilities
- Error handling
"""
import unittest
import tempfile
import shutil
from pathlib import Path
import torch
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

# Import XAI modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from protein.XAI.model_loader import (
    load_model, 
    list_available_models, 
    ModelLoaderError,
    _registry
)
from protein.XAI.data_utils import (
    prepare_data,
    DataPreparationError
)


class TestModelLoader(unittest.TestCase):
    """Test model loading functionality"""
    
    def setUp(self):
        """Set up temporary directory for test models"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.models_dir = self.temp_dir / "models"
        self.models_dir.mkdir(parents=True)
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)
    
    def test_load_sklearn_model(self):
        """Test loading sklearn model from .pkl file"""
        # Create a dummy sklearn model
        model = LogisticRegression()
        model.fit(np.random.randn(10, 5), np.random.randint(0, 2, 10))
        
        # Save model
        model_path = self.models_dir / "logistic_regression.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Load model
        loaded_model = load_model(str(self.temp_dir), "logistic_regression")
        
        # Verify it's the same type
        self.assertIsInstance(loaded_model, LogisticRegression)
    
    def test_load_pytorch_model(self):
        """Test loading PyTorch model from .pth file"""
        from protein.model import NeuralNetwork, NeuralNetworkClassifier
        
        # Create a dummy model
        n_features = 10
        model = NeuralNetwork(n_features=n_features, hidden_sizes=(32, 16), dropout=0.1)
        
        # Create checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_config': {
                'model_class': 'NeuralNetwork',
                'model_module': 'src.protein.model',
                'n_features': n_features,
                'hidden_sizes': (32, 16),
                'dropout': 0.1,
                'random_state': 42
            }
        }
        
        # Save checkpoint
        model_path = self.models_dir / "neural_network.pth"
        torch.save(checkpoint, model_path)
        
        # Load model
        loaded_model = load_model(str(self.temp_dir), "neural_network")
        
        # Verify it's a NeuralNetworkClassifier
        self.assertIsInstance(loaded_model, NeuralNetworkClassifier)
        self.assertEqual(loaded_model.n_features, n_features)
        self.assertTrue(hasattr(loaded_model, 'model'))
    
    def test_model_not_found(self):
        """Test error handling when model doesn't exist"""
        with self.assertRaises(ModelLoaderError) as context:
            load_model(str(self.temp_dir), "nonexistent_model")
        
        self.assertIn("not found", str(context.exception).lower())
    
    def test_invalid_checkpoint_format(self):
        """Test error handling for invalid checkpoint format"""
        # Create invalid checkpoint (missing model_config)
        checkpoint = {
            'model_state_dict': {}
        }
        model_path = self.models_dir / "neural_network.pth"
        torch.save(checkpoint, model_path)
        
        with self.assertRaises(ModelLoaderError) as context:
            load_model(str(self.temp_dir), "neural_network")
        
        self.assertIn("model_config", str(context.exception))
    
    def test_list_available_models(self):
        """Test listing available models"""
        # Create some model files
        (self.models_dir / "logistic_regression.pkl").touch()
        (self.models_dir / "neural_network.pth").touch()
        (self.models_dir / "random_file.txt").touch()  # Should be ignored
        
        models = list_available_models(str(self.temp_dir))
        
        self.assertIn("logistic_regression", models)
        self.assertIn("neural_network", models)
        self.assertNotIn("random_file", models)
    
    def test_list_models_empty_directory(self):
        """Test listing models from empty directory"""
        models = list_available_models(str(self.temp_dir))
        self.assertEqual(models, [])
    
    def test_registry_list_models(self):
        """Test that registry can list registered models"""
        models = _registry.list_models()
        self.assertIn("neural_network", models)


class TestDataUtils(unittest.TestCase):
    """Test data preparation utilities"""
    
    def setUp(self):
        """Set up temporary directory for test data"""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)
    
    def test_prepare_data_missing_scaler(self):
        """Test error handling when scaler is missing"""
        # Create a dummy CSV file
        csv_path = self.temp_dir / "test.csv"
        csv_path.write_text("RID,research_group,feature1,feature2\n1,CN,1.0,2.0\n")
        
        with self.assertRaises(DataPreparationError) as context:
            prepare_data(str(self.temp_dir), str(csv_path))
        
        self.assertIn("scaler", str(context.exception).lower())
    
    def test_prepare_data_missing_csv(self):
        """Test error handling when CSV file doesn't exist"""
        # Create scaler file
        scaler_path = self.temp_dir / "scaler.pkl"
        from sklearn.preprocessing import StandardScaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(StandardScaler(), f)
        
        with self.assertRaises(FileNotFoundError):
            prepare_data(str(self.temp_dir), str(self.temp_dir / "nonexistent.csv"))


class TestErrorMessages(unittest.TestCase):
    """Test that error messages are clear and helpful"""
    
    def setUp(self):
        """Set up temporary directory"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.models_dir = self.temp_dir / "models"
        self.models_dir.mkdir(parents=True)
    
    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)
    
    def test_model_not_found_error_message(self):
        """Test that error message includes available models"""
        # Create one model
        (self.models_dir / "logistic_regression.pkl").touch()
        
        try:
            load_model(str(self.temp_dir), "neural_network")
            self.fail("Should have raised ModelLoaderError")
        except ModelLoaderError as e:
            error_msg = str(e)
            self.assertIn("neural_network", error_msg)
            self.assertIn("logistic_regression", error_msg)  # Should list available


if __name__ == '__main__':
    unittest.main()

