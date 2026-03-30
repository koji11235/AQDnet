"""
AQDnet Smoke Tests

Minimal test suite for AQDnet functionality:
- Feature extraction from sample structures
- Feature loading  
- Prediction pipeline availability
"""

import os
import sys
import tempfile
import glob
import unittest
import pandas as pd
import numpy as np

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import aqdnet


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


class TestFeatureLoading(unittest.TestCase):
    """Test feature file loading"""
    
    def test_sample_features_exist(self):
        """Check if sample features are available"""
        features_dir = os.path.join(
            os.path.dirname(__file__),
            '..',
            'features'
        )
        
        # Check for CSV features
        feature_files = glob.glob(os.path.join(features_dir, 'feature_trainset_*.csv'))
        self.assertGreater(len(feature_files), 0, f"No feature files found in {features_dir}")
    
    def test_feature_csv_loading(self):
        """Test loading feature CSV files"""
        features_dir = os.path.join(
            os.path.dirname(__file__),
            '..',
            'features'
        )
        
        feature_files = glob.glob(os.path.join(features_dir, 'feature_trainset_*.csv'))
        if len(feature_files) > 0:
            # Load first feature file
            df = pd.read_csv(feature_files[0])
            self.assertIsNotNone(df)
            self.assertGreater(len(df), 0, "Feature CSV is empty")
            self.assertGreater(df.shape[1], 0, "Feature CSV has no columns")
    
    def test_label_csv_loading(self):
        """Test loading label CSV files"""
        features_dir = os.path.join(
            os.path.dirname(__file__),
            '..',
            'features'
        )
        
        label_file = os.path.join(features_dir, 'label_trainset.csv')
        if os.path.exists(label_file):
            df = pd.read_csv(label_file)
            self.assertIsNotNone(df)
            self.assertGreater(len(df), 0, "Label CSV is empty")


class TestModelStructure(unittest.TestCase):
    """Test model infrastructure"""
    
    def test_model_params_accessible(self):
        """Test that model parameter files are accessible"""
        models_dir = os.path.join(
            os.path.dirname(__file__),
            '..',
            'models'
        )
        
        # Check Docking model
        docking_params = os.path.join(
            models_dir,
            'Docking_Energy30RMSD2.5',
            'params_for_predict.json'
        )
        self.assertTrue(os.path.exists(docking_params), f"Not found: {docking_params}")
        
        # Check Scoring model
        scoring_params = os.path.join(
            models_dir,
            'Scoring_Energy02RMSD2.0',
            'params_for_predict.json'
        )
        self.assertTrue(os.path.exists(scoring_params), f"Not found: {scoring_params}")


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
            self.assertTrue(hasattr(commands, 'demo_command'))
        except ImportError:
            self.skipTest("aqdnet_cli not installed in test environment")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
