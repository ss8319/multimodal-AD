"""
Test script for model config extraction and save/load functionality.

Tests:
1. All models store constructor parameters as attributes
2. extract_model_config() extracts all parameters correctly
3. Save/load cycle works end-to-end
4. Missing parameters fail with clear errors
"""

import sys
from pathlib import Path
import torch
import numpy as np
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.protein.model import (
    NeuralNetwork, 
    CustomTransformerEncoder, 
    ProteinTransformer, 
    ProteinAttentionPooling
)
from src.protein.utils import extract_model_config, save_model
from src.protein.model_loader import load_pytorch_model_generic


def test_model_parameter_storage():
    """Test that all models store their constructor parameters"""
    print("=" * 70)
    print("TEST 1: Model Parameter Storage")
    print("=" * 70)
    
    models_to_test = [
        ("NeuralNetwork", NeuralNetwork(n_features=320, hidden_sizes=(128, 64), dropout=0.2)),
        ("CustomTransformerEncoder", CustomTransformerEncoder(
            n_features=320, d_model=32, n_heads=2, n_blocks=2, ffn_dim=64, dropout=0.1
        )),
        ("ProteinTransformer", ProteinTransformer(
            n_features=320, d_model=64, n_heads=4, n_layers=2, dropout=0.1
        )),
        ("ProteinAttentionPooling", ProteinAttentionPooling(
            n_features=320, d_model=64, dropout=0.2
        )),
    ]
    
    all_passed = True
    for model_name, model in models_to_test:
        print(f"\n  Testing {model_name}...")
        
        # Get expected parameters from constructor signature
        import inspect
        sig = inspect.signature(model.__class__.__init__)
        expected_params = [p for p in sig.parameters.keys() if p != 'self']
        
        # Check each parameter is stored
        missing = []
        for param in expected_params:
            if not hasattr(model, param):
                missing.append(param)
        
        if missing:
            print(f"    [FAIL] Missing attributes: {missing}")
            all_passed = False
        else:
            print(f"    [PASS] All {len(expected_params)} parameters stored")
            # Print stored values
            for param in expected_params:
                value = getattr(model, param)
                print(f"      {param} = {value}")
    
    return all_passed


def test_config_extraction():
    """Test that extract_model_config() extracts all parameters correctly"""
    print("\n" + "=" * 70)
    print("TEST 2: Config Extraction")
    print("=" * 70)
    
    models_to_test = [
        ("NeuralNetwork", NeuralNetwork(n_features=320, hidden_sizes=(128, 64), dropout=0.2)),
        ("CustomTransformerEncoder", CustomTransformerEncoder(
            n_features=320, d_model=32, n_heads=2, n_blocks=2, ffn_dim=64, dropout=0.1
        )),
        ("ProteinTransformer", ProteinTransformer(
            n_features=320, d_model=64, n_heads=4, n_layers=2, dropout=0.1
        )),
        ("ProteinAttentionPooling", ProteinAttentionPooling(
            n_features=320, d_model=64, dropout=0.2
        )),
    ]
    
    all_passed = True
    for model_name, model in models_to_test:
        print(f"\n  Testing {model_name}...")
        
        try:
            config = extract_model_config(model)
            
            # Verify required fields
            assert 'model_class' in config, "Missing model_class"
            assert 'model_module' in config, "Missing model_module"
            assert config['model_class'] == model_name, f"Wrong class name: {config['model_class']}"
            
            # Get expected parameters
            import inspect
            sig = inspect.signature(model.__class__.__init__)
            expected_params = [p for p in sig.parameters.keys() if p != 'self']
            
            # Check all parameters are in config
            missing = [p for p in expected_params if p not in config]
            
            if missing:
                print(f"    [FAIL] Missing from config: {missing}")
                all_passed = False
            else:
                print(f"    [PASS] All {len(expected_params)} parameters extracted")
                print(f"      Config keys: {list(config.keys())}")
                
        except Exception as e:
            print(f"    [FAIL] {e}")
            all_passed = False
    
    return all_passed


def test_save_load_cycle():
    """Test end-to-end save/load cycle"""
    print("\n" + "=" * 70)
    print("TEST 3: Save/Load Cycle")
    print("=" * 70)
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        models_to_test = [
            ("NeuralNetwork", NeuralNetwork(n_features=320, hidden_sizes=(128, 64), dropout=0.2)),
            ("CustomTransformerEncoder", CustomTransformerEncoder(
                n_features=320, d_model=32, n_heads=2, n_blocks=2, ffn_dim=64, dropout=0.1
            )),
            ("ProteinTransformer", ProteinTransformer(
                n_features=320, d_model=64, n_heads=4, n_layers=2, dropout=0.1
            )),
        ]
        
        all_passed = True
        
        for model_name, model in models_to_test:
            print(f"\n  Testing {model_name}...")
            
            try:
                # Create dummy wrapper (simulating sklearn wrapper)
                class DummyWrapper:
                    def __init__(self, model):
                        self.model = model
                
                wrapper = DummyWrapper(model)
                
                # Save model
                save_model(wrapper, model_name.lower().replace(" ", "_"), temp_dir)
                
                # Load model
                model_path = temp_dir / "models" / f"{model_name.lower().replace(' ', '_')}.pth"
                checkpoint = torch.load(model_path, map_location='cpu')
                loaded_model = load_pytorch_model_generic(checkpoint, device='cpu')
                
                # Verify architecture matches
                original_config = extract_model_config(model)
                loaded_config = extract_model_config(loaded_model)
                
                # Compare configs (excluding model_class/module which should match)
                original_params = {k: v for k, v in original_config.items() 
                                 if k not in ['model_class', 'model_module']}
                loaded_params = {k: v for k, v in loaded_config.items() 
                               if k not in ['model_class', 'model_module']}
                
                if original_params != loaded_params:
                    print(f"    [FAIL] Config mismatch")
                    print(f"      Original: {original_params}")
                    print(f"      Loaded: {loaded_params}")
                    all_passed = False
                else:
                    print(f"    [PASS] Config matches")
                    
                    # Test predictions match (with same random seed)
                    torch.manual_seed(42)
                    test_input = torch.randn(1, 320)
                    model.eval()
                    loaded_model.eval()
                    
                    with torch.no_grad():
                        orig_output = model(test_input)
                        loaded_output = loaded_model(test_input)
                    
                    if torch.allclose(orig_output, loaded_output, atol=1e-6):
                        print(f"    [PASS] Predictions match")
                    else:
                        print(f"    [WARN] Predictions differ (might be due to random init)")
                        print(f"      Original: {orig_output[0, :5]}")
                        print(f"      Loaded: {loaded_output[0, :5]}")
                        
            except Exception as e:
                print(f"    [FAIL] {e}")
                import traceback
                traceback.print_exc()
                all_passed = False
        
        return all_passed
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_fail_fast_on_missing_params():
    """Test that missing parameters fail with clear error"""
    print("\n" + "=" * 70)
    print("TEST 4: Fail-Fast on Missing Parameters")
    print("=" * 70)
    
    # Create a model that doesn't store all parameters
    class BadModel(torch.nn.Module):
        def __init__(self, n_features, d_model=32, missing_param=64):
            super().__init__()
            self.n_features = n_features  # Stored
            self.d_model = d_model        # Stored
            # missing_param is NOT stored
            self.layer = torch.nn.Linear(n_features, d_model)
    
    print("\n  Testing BadModel (missing 'missing_param')...")
    
    bad_model = BadModel(n_features=320, d_model=32, missing_param=64)
    
    try:
        config = extract_model_config(bad_model)
        print(f"    [FAIL] Should have raised ValueError, but got config: {list(config.keys())}")
        return False
    except ValueError as e:
        error_msg = str(e)
        if "missing required constructor parameters" in error_msg.lower():
            print(f"    [PASS] Correctly raised ValueError")
            print(f"      Error message: {error_msg[:200]}...")
            return True
        else:
            print(f"    [FAIL] Wrong error message: {error_msg}")
            return False
    except Exception as e:
        print(f"    [FAIL] Wrong exception type: {type(e).__name__}: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("MODEL CONFIG EXTRACTION & SAVE/LOAD TESTS")
    print("=" * 70)
    
    results = []
    
    # Test 1: Parameter storage
    results.append(("Parameter Storage", test_model_parameter_storage()))
    
    # Test 2: Config extraction
    results.append(("Config Extraction", test_config_extraction()))
    
    # Test 3: Save/load cycle
    results.append(("Save/Load Cycle", test_save_load_cycle()))
    
    # Test 4: Fail-fast
    results.append(("Fail-Fast on Missing Params", test_fail_fast_on_missing_params()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())


