"""
AQDnet Smoke Tests

Minimal test suite for AQDnet functionality:
- Feature extraction from sample structures
- Model parameter loading and prediction pipeline
- CLI interface availability
"""

import os
import sys
import tempfile
import glob
import json
import unittest
import pandas as pd
import numpy as np

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import aqdnet
from model import ModelByTensorflow
from structure import ElementwiseDNN


class TestFeatureExtraction(unittest.TestCase):
    """Test feature extraction pipeline"""
    
    def test_feature_generator_initialization(self):
        """Test that FeatureGenerator can be instantiated"""
        fg = aqdnet.FeatureGenerator(lig_code='LGD')
        self.assertIsNotNone(fg)
        self.assertEqual(fg.lig_code, 'LGD')
    
    def test_feature_extraction_from_sample(self):
        """Test feature extraction from sample structures"""
        sample_dir = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'sample_structures'
        )
        
        # Get PDB files
        pdb_files = sorted(glob.glob(os.path.join(sample_dir, '**', '*.pdb'), recursive=True))[:1]
        
        self.assertGreater(len(pdb_files), 0, f"No PDB files found in {sample_dir}")
        
        # Generate features
        with tempfile.NamedTemporaryFile(prefix='test_', suffix='.dat', delete=False) as temp_file:
            temp_name = temp_file.name
            with open(temp_name, mode='w') as f:
                f.write('\n'.join(pdb_files))
            
            try:
                fg = aqdnet.FeatureGenerator(lig_code='LGD')
                dataset = fg.generate(temp_name, mode='complex', num_cpu=1)
                
                # Verify output
                self.assertIsNotNone(dataset)
                self.assertGreater(len(dataset), 0, "Generated dataset is empty")
                self.assertGreater(dataset.shape[1], 0, "Generated dataset has no features")
                
            finally:
                if os.path.exists(temp_name):
                    os.remove(temp_name)


class TestModelPipeline(unittest.TestCase):
    """Test model loading and prediction pipeline"""
    
    def test_model_params_loading(self):
        """Test that model parameters can be loaded"""
        models_dir = os.path.join(
            os.path.dirname(__file__),
            '..',
            'models'
        )
        
        # Check Docking model params
        docking_params_file = os.path.join(
            models_dir,
            'Docking_Energy30RMSD2.5',
            'params_for_predict.json'
        )
        
        self.assertTrue(os.path.exists(docking_params_file), 
                       f"Model params not found: {docking_params_file}")
        
        # Load and verify params structure
        with open(docking_params_file, 'r') as f:
            params = json.load(f)
        
        self.assertIn('model_class_name', params, "Missing model_class_name in params")
        self.assertIn('fg_params', params, "Missing fg_params in params")
        self.assertIn('model_params', params, "Missing model_params in params")
        self.assertEqual(params['model_class_name'], 'ElementwiseDNN', 
                        "Expected ElementwiseDNN model class")
    
    def test_mock_model_params_loading(self):
        """Test loading mock model parameters for testing"""
        mock_model_dir = os.path.join(
            os.path.dirname(__file__),
            'mock_model'
        )
        
        params_file = os.path.join(mock_model_dir, 'params_for_predict.json')
        
        self.assertTrue(os.path.exists(params_file), 
                       f"Mock model params not found: {params_file}")
        
        # Load and verify params
        with open(params_file, 'r') as f:
            params = json.load(f)
        
        self.assertIn('model_class_name', params)
        self.assertIn('fg_params', params)
        self.assertIn('model_params', params)
    
    def test_model_class_instantiation(self):
        """Test that ElementwiseDNN model class can be instantiated with default params"""
        # Use minimal parameters - ElementwiseDNN has many but mostly have defaults
        try:
            model = ModelByTensorflow(network_cls=ElementwiseDNN)
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"Failed to instantiate ModelByTensorflow: {e}")


class TestCLIInterface(unittest.TestCase):
    """Test CLI interface availability"""
    
    def test_cli_import(self):
        """Test that CLI module can be imported"""
        try:
            import aqdnet_cli
            self.assertIsNotNone(aqdnet_cli)
        except ImportError:
            self.skipTest("aqdnet_cli not installed in test environment")
    
    def test_cli_commands_available(self):
        """Test that CLI commands are properly defined"""
        try:
            from aqdnet_cli import commands
            self.assertTrue(hasattr(commands, 'featurize_command'))
            self.assertTrue(hasattr(commands, 'predict_command'))
        except ImportError:
            self.skipTest("aqdnet_cli not installed in test environment")


if __name__ == '__main__':
    unittest.main()

    
    def test_cli_commands_available(self):
        """Test that CLI commands are properly defined"""
        try:
            from aqdnet_cli import commands
            self.assertTrue(hasattr(commands, 'featurize_command'))
            self.assertTrue(hasattr(commands, 'predict_command'))
            self.assertTrue(hasattr(commands, 'demo_command'))
        except ImportError:
            self.skipTest("aqdnet_cli not installed in test environment")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
